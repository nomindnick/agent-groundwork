"""Tests for the prompted-mode fenced-JSON parser.

The parser is a state machine that handles streamed text and extracts
```json ...``` blocks as tool calls. State machines have bugs; the cost of
a parser bug is silently corrupted tool calls. So we test the cases.
"""

from __future__ import annotations

from agent_groundwork.providers.ollama import _PromptedFenceParser, _ParseEvent


TOOLS = {"write_file", "list_files"}


def _drain(parser: _PromptedFenceParser, deltas: list[str]) -> list[_ParseEvent]:
    events: list[_ParseEvent] = []
    for d in deltas:
        events.extend(parser.feed(d, TOOLS))
    events.extend(parser.flush())
    return events


def _texts(events: list[_ParseEvent]) -> str:
    return "".join(e.text for e in events if e.kind == "text")


def _calls(events: list[_ParseEvent]) -> list:
    return [e.call for e in events if e.kind == "tool_call"]


def _errors(events: list[_ParseEvent]) -> list:
    return [e.error for e in events if e.kind == "parse_error"]


# --------------------------- happy paths ---------------------------

def test_plain_text_only() -> None:
    parser = _PromptedFenceParser()
    events = _drain(parser, ["Hello world, no fences here."])
    assert _texts(events) == "Hello world, no fences here."
    assert _calls(events) == []
    assert _errors(events) == []


def test_single_fence_one_delta() -> None:
    parser = _PromptedFenceParser()
    payload = '```json\n{"tool": "write_file", "args": {"path": "x.md", "content": "hi"}}\n```'
    events = _drain(parser, [payload])
    calls = _calls(events)
    assert len(calls) == 1
    assert calls[0].name == "write_file"
    assert calls[0].args == {"path": "x.md", "content": "hi"}
    assert _errors(events) == []


def test_fence_split_across_two_deltas() -> None:
    parser = _PromptedFenceParser()
    deltas = [
        '```json\n{"tool": "write_file", "args": {"path": "y.md", ',
        '"content": "yo"}}\n```',
    ]
    events = _drain(parser, deltas)
    calls = _calls(events)
    assert len(calls) == 1
    assert calls[0].args == {"path": "y.md", "content": "yo"}


def test_fence_opener_split_across_delta_boundary() -> None:
    parser = _PromptedFenceParser()
    deltas = [
        "before ```js",
        'on\n{"tool": "list_files", "args": {}}\n```',
    ]
    events = _drain(parser, deltas)
    assert _texts(events) == "before "
    calls = _calls(events)
    assert len(calls) == 1
    assert calls[0].name == "list_files"
    assert calls[0].args == {}


def test_two_fences_in_one_stream() -> None:
    parser = _PromptedFenceParser()
    payload = (
        '```json\n{"tool": "list_files", "args": {}}\n```'
        " then "
        '```json\n{"tool": "write_file", "args": {"path": "a", "content": "b"}}\n```'
    )
    events = _drain(parser, [payload])
    calls = _calls(events)
    assert len(calls) == 2
    assert [c.name for c in calls] == ["list_files", "write_file"]
    assert " then " in _texts(events)


def test_text_call_text() -> None:
    parser = _PromptedFenceParser()
    payload = (
        "Sure thing! "
        '```json\n{"tool": "list_files", "args": {}}\n```'
        " Done."
    )
    events = _drain(parser, [payload])
    assert "Sure thing!" in _texts(events)
    assert "Done." in _texts(events)
    assert len(_calls(events)) == 1


def test_streaming_character_by_character() -> None:
    """Worst case: every character arrives in its own delta."""
    parser = _PromptedFenceParser()
    payload = 'pre ```json\n{"tool": "list_files", "args": {}}\n``` post'
    deltas = [c for c in payload]
    events = _drain(parser, deltas)
    assert "pre " in _texts(events)
    assert "post" in _texts(events) or " post" in _texts(events)
    assert len(_calls(events)) == 1


# --------------------------- parse errors ---------------------------

def test_bad_json_in_fence() -> None:
    parser = _PromptedFenceParser()
    events = _drain(parser, ['```json\n{tool: "list_files"}\n```'])
    errs = _errors(events)
    assert len(errs) == 1
    assert errs[0].stage == "json_invalid"


def test_missing_tool_key() -> None:
    parser = _PromptedFenceParser()
    events = _drain(parser, ['```json\n{"args": {}}\n```'])
    errs = _errors(events)
    assert len(errs) == 1
    assert errs[0].stage == "schema_invalid"


def test_unknown_tool_name() -> None:
    parser = _PromptedFenceParser()
    events = _drain(parser, ['```json\n{"tool": "delete_universe", "args": {}}\n```'])
    errs = _errors(events)
    assert len(errs) == 1
    assert errs[0].stage == "unknown_tool"
    assert "delete_universe" in errs[0].message


def test_unterminated_fence_flushed() -> None:
    parser = _PromptedFenceParser()
    events = _drain(parser, ['```json\n{"tool": "list_files", "args": {}'])
    errs = _errors(events)
    assert len(errs) == 1
    assert errs[0].stage == "fence_unterminated"
    assert _calls(events) == []


def test_args_not_dict() -> None:
    parser = _PromptedFenceParser()
    events = _drain(parser, ['```json\n{"tool": "list_files", "args": [1,2,3]}\n```'])
    errs = _errors(events)
    assert len(errs) == 1
    assert errs[0].stage == "schema_invalid"


def test_tool_not_string() -> None:
    parser = _PromptedFenceParser()
    events = _drain(parser, ['```json\n{"tool": 42, "args": {}}\n```'])
    errs = _errors(events)
    assert len(errs) == 1
    assert errs[0].stage == "schema_invalid"


def test_parse_error_followed_by_resumed_text() -> None:
    """After a bad fence, the parser returns to text mode and keeps emitting."""
    parser = _PromptedFenceParser()
    events = _drain(
        parser,
        ['```json\nnot json at all\n``` after-text'],
    )
    assert len(_errors(events)) == 1
    assert "after-text" in _texts(events)
