# Prompted-mode tool-call parser extracts fragments of text as tool names

Status: **Open** — reproducible, code-level bug, not a model issue.

## Context

- Date: 2026-04-08
- Model: `gemma4:e4b` (prompted)
- AGENT.md: `ef4fb1e`
- Tools: `list_files`, `read_file`, `write_file`, `edit_file`, `search_files`, `ask_user`
- Session: `cli-b8a3f3f2`
- Relevant code: the prompted-mode tool-call extractor in `agent_groundwork/providers/ollama.py` (or wherever the `<tool_call>`-style regex / JSON parser lives)

## What happened

Within the first five events of the session the prompted-mode parser
mis-tokenized the model's output twice, producing two dispatcher errors that
the model then had to work around.

At trace line 2, the model's `final_text` ended with an inline JSON blob
instead of a properly-delimited tool call:

```
{"tool": "list_files", "args": {}}
```

That is, the model wrote what it thought was a tool call as literal text in
its assistant message rather than in the channel the prompted-mode parser
expects. The parser then did two bad things on the following provider turn:

1. **Trace line 4**: extracted a `tool_call_started` event with `name = "\\"`
   (a literal backslash). Something in the parser matched a backslash as a
   tool name — almost certainly a regex eating escaped quotes from the
   serialized JSON.
2. **Trace line 5**: the dispatcher returned
   `unknown tool: '"read_file", "args":'` — i.e. the parser extracted
   `"read_file", "args":` (with embedded quotes and a trailing colon) as a
   tool name. This is a fragment of a different JSON tool-call that the
   parser latched onto mid-string.

Both failed at the dispatcher (correctly — these are not real tool names)
and surfaced to the model as tool errors, which the model then had to
recover from. The rest of the session eventually proceeded normally, but
the first two user turns burned provider calls on garbage.

## Evidence

Trace `traces/20260408-175105-cli-b8a3f3f2.jsonl`, first six events:

```jsonl
{"event_type": "provider_call", "model": "gemma4:e4b", "mode": "prompted", "prompt_tokens": 2213, "completion_tokens": 25}
{"event_type": "agent_event.done", "type": "done", "final_text": "{\"tool\": \"list_files\", \"args\": {}}"}
{"event_type": "provider_call", "model": "gemma4:e4b", "mode": "prompted", "prompt_tokens": 2238, "completion_tokens": 60}
{"event_type": "agent_event.tool_call_started", "type": "tool_call_started", "name": "\\", "args": {"path": "memory/identity.md"}, "call_id": "call_5c39d3ebbca7"}
{"event_type": "tool_call", "name": "\\", "result": {"ok": false, "data": null, "error": "unknown tool: '\"read_file\", \"args\":'", "latency_ms": 0.024934997782111168}}
{"event_type": "agent_event.tool_call_started", "type": "tool_call_started", "name": "read_file", "args": {"path": "memory/identity.md"}, "call_id": "call_db427ea68139"}
```

Note that the `args` on the broken call at line 4 are actually well-formed
(`{"path": "memory/identity.md"}`) — only the `name` field is corrupted.
This strongly suggests the parser is separating tool-name extraction from
argument extraction and one of them is using a looser pattern than the
other.

## Implications

1. **The parser is matching more liberally than it should.** The fact that
   the `args` extract cleanly but the `name` is a stray backslash (and that
   the dispatcher's own error message then contains another fragment,
   `'"read_file", "args":'`) tells us the extractor is doing
   field-by-field pattern matching rather than first isolating a complete
   tool-call JSON object and then deserializing it. The fix is probably to
   parse the full JSON object exactly once and pull fields out of the
   parsed dict, rather than running separate regexes for `name` and
   `args`.

2. **The model will sometimes write a tool call as bare JSON in text,
   not in whatever delimited format the prompted-mode prompt asks for.**
   This is a model-side quirk worth defending against: if the parser sees
   a naked `{"tool": ..., "args": ...}` at the end of a response with no
   wrapping delimiters, it should probably accept it as a tool call
   (optionally, with a trace warning) rather than leave it in the text
   and have the next turn's parser find fragments of it.

3. **This bug is independent of the hallucinated-tool-calls finding.**
   Different symptom, different fix, different reader. Fixing the parser
   will not make `gemma4:e4b` honest about narrated actions; curing the
   hallucination bug will not make the parser handle malformed output.
   They both happen to appear in the same session but they should stay
   as separate findings.

4. **Adjacent observation, left here for context:** in the same session
   (trace lines 80, 84) the model also invented the tool name
   `list_directory` and called it twice. That is a *model* hallucination,
   not a parser bug — the dispatcher's `unknown tool: 'list_directory'`
   error is the correct behavior. The model self-corrected on the next
   turn (*"My apologies. I incorrectly attempted to use a tool that does
   not exist..."*) and called `list_files` correctly. Noting it here
   because it shares the "unknown tool" error surface with the parser
   bug above, but it belongs to the hallucinated-tool-calls family, not
   this finding.
