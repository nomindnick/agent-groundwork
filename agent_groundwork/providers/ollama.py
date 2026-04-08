"""Ollama implementation of the Provider protocol.

Supports both `tool_call_mode="native"` (uses Ollama's `tools=` parameter and
the model's trained tool template) and `tool_call_mode="prompted"` (injects a
markdown tool block into the system prompt and parses fenced JSON code blocks
out of the model's free-form output).

Both modes flow through `stream()`, which yields a sequence of `ChatChunk`s
ending in a terminal `done=True` chunk carrying token counts and timings.
`chat()` is a thin wrapper that drains the stream into a `ChatResult`.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, ClassVar, Iterator, Literal

import ollama

from agent_groundwork.providers.base import (
    ChatChunk,
    ChatResult,
    Message,
    ParseError,
    ProviderError,
    ProviderToolModeUnsupportedError,
    ToolCall,
    ToolSchema,
    estimate_tokens,
)
from agent_groundwork.tools.base import render_prompted_block, to_native_tool_dict


# =============================== provider ===============================

class OllamaProvider:
    """Concrete `Provider` backed by a local Ollama daemon."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        keep_alive: str = "30m",
    ) -> None:
        self._client = ollama.AsyncClient(host=host)
        self._keep_alive = keep_alive

    async def chat(
        self,
        model: str,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        tool_call_mode: Literal["native", "prompted"] = "native",
    ) -> ChatResult:
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        parse_errors: list[ParseError] = []
        last: ChatChunk | None = None
        async for chunk in self.stream(model, messages, tools, tool_call_mode):
            if chunk.text:
                text_parts.append(chunk.text)
            if chunk.tool_call is not None:
                calls.append(chunk.tool_call)
            if chunk.parse_error is not None:
                parse_errors.append(chunk.parse_error)
            if chunk.done:
                last = chunk
        if last is None:
            raise ProviderError("provider stream did not emit a terminal chunk")
        return ChatResult(
            text="".join(text_parts),
            tool_calls=calls,
            parse_errors=parse_errors,
            prompt_tokens=last.prompt_tokens,
            completion_tokens=last.completion_tokens,
            total_latency_ms=last.total_latency_ms or 0.0,
            ttft_ms=last.ttft_ms,
            tokens_estimated=last.tokens_estimated,
        )

    async def stream(
        self,
        model: str,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        tool_call_mode: Literal["native", "prompted"] = "native",
    ) -> AsyncIterator[ChatChunk]:
        if tool_call_mode == "native":
            async for chunk in self._stream_native(model, messages, tools or []):
                yield chunk
        elif tool_call_mode == "prompted":
            async for chunk in self._stream_prompted(model, messages, tools or []):
                yield chunk
        else:
            raise ValueError(f"unknown tool_call_mode: {tool_call_mode!r}")

    # --------------------------- native mode ---------------------------

    async def _stream_native(
        self,
        model: str,
        messages: list[Message],
        tools: list[ToolSchema],
    ) -> AsyncIterator[ChatChunk]:
        native_tools = [to_native_tool_dict(t) for t in tools]
        native_msgs = [_to_ollama_message(m) for m in messages]

        start = time.monotonic()
        first_content_at: float | None = None
        completion_chars = 0
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        try:
            stream = await self._client.chat(
                model=model,
                messages=native_msgs,
                tools=native_tools or None,
                stream=True,
                keep_alive=self._keep_alive,
            )
        except ollama.ResponseError as e:
            if "tool" in str(e).lower():
                raise ProviderToolModeUnsupportedError(str(e)) from e
            raise ProviderError(str(e)) from e

        async for raw in stream:
            text, raw_tool_calls, done, p_tok, c_tok = _extract_chunk_fields(raw)

            if text:
                if first_content_at is None:
                    first_content_at = time.monotonic()
                completion_chars += len(text)
                yield ChatChunk(text=text)

            if raw_tool_calls:
                for tc in raw_tool_calls:
                    parsed = _parse_native_tool_call(tc)
                    if isinstance(parsed, ParseError):
                        yield ChatChunk(parse_error=parsed)
                        continue
                    if first_content_at is None:
                        first_content_at = time.monotonic()
                    yield ChatChunk(tool_call=parsed)

            if done:
                if p_tok is not None:
                    prompt_tokens = p_tok
                if c_tok is not None:
                    completion_tokens = c_tok

        end = time.monotonic()
        yield _build_done_chunk(
            messages=messages,
            completion_chars=completion_chars,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            start=start,
            end=end,
            first_content_at=first_content_at,
        )

    # --------------------------- prompted mode ---------------------------

    async def _stream_prompted(
        self,
        model: str,
        messages: list[Message],
        tools: list[ToolSchema],
    ) -> AsyncIterator[ChatChunk]:
        augmented = _inject_tool_block(messages, tools)
        native_msgs = [_to_ollama_message(m) for m in augmented]

        start = time.monotonic()
        first_content_at: float | None = None
        completion_chars = 0
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        try:
            stream = await self._client.chat(
                model=model,
                messages=native_msgs,
                stream=True,
                keep_alive=self._keep_alive,
            )
        except ollama.ResponseError as e:
            raise ProviderError(str(e)) from e

        parser = _PromptedFenceParser()
        tool_names = {t.name for t in tools}

        async for raw in stream:
            delta, raw_tool_calls, done, p_tok, c_tok = _extract_chunk_fields(raw)

            if delta:
                if first_content_at is None:
                    first_content_at = time.monotonic()
                completion_chars += len(delta)

            for event in parser.feed(delta, tool_names):
                if event.kind == "text" and event.text:
                    yield ChatChunk(text=event.text)
                elif event.kind == "tool_call" and event.call is not None:
                    yield ChatChunk(tool_call=event.call)
                elif event.kind == "parse_error" and event.error is not None:
                    yield ChatChunk(parse_error=event.error)

            # Models trained with native tool-call templates emit tool calls in
            # their trained format even when we didn't pass `tools=`. Ollama's
            # chat-template parser surfaces those as `message.tool_calls`. If
            # we ignored them here, the tokens would be silently dropped (zero
            # text, zero parsed calls, zero parse errors). Treat them as
            # successful tool calls — prompted mode in practice means "give the
            # model tool descriptions in the system prompt and accept whatever
            # call format it produces".
            if raw_tool_calls:
                for tc in raw_tool_calls:
                    parsed = _parse_native_tool_call(tc)
                    if isinstance(parsed, ParseError):
                        yield ChatChunk(parse_error=parsed)
                        continue
                    if first_content_at is None:
                        first_content_at = time.monotonic()
                    yield ChatChunk(tool_call=parsed)

            if done:
                if p_tok is not None:
                    prompt_tokens = p_tok
                if c_tok is not None:
                    completion_tokens = c_tok

        for event in parser.flush():
            if event.kind == "text" and event.text:
                yield ChatChunk(text=event.text)
            elif event.kind == "parse_error" and event.error is not None:
                yield ChatChunk(parse_error=event.error)

        end = time.monotonic()
        yield _build_done_chunk(
            messages=augmented,
            completion_chars=completion_chars,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            start=start,
            end=end,
            first_content_at=first_content_at,
        )


# =============================== helpers ===============================

def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Defensive accessor: works on dicts and objects with attributes.

    The ollama python package has gone back and forth on whether responses
    are TypedDicts or Pydantic models. Handle both shapes.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_chunk_fields(
    raw: Any,
) -> tuple[str, list[Any] | None, bool, int | None, int | None]:
    """Pull (text, tool_calls, done, prompt_tokens, completion_tokens) from
    a raw Ollama streaming chunk regardless of shape."""
    msg = _get(raw, "message", {})
    text = _get(msg, "content", "") or ""
    tcs = _get(msg, "tool_calls", None)
    done = bool(_get(raw, "done", False))
    p_tok = _get(raw, "prompt_eval_count", None)
    c_tok = _get(raw, "eval_count", None)
    return text, tcs, done, p_tok, c_tok


def _parse_native_tool_call(tc: Any) -> ToolCall | ParseError:
    """Convert an ollama-shaped tool_call entry into our `ToolCall` type."""
    fn = _get(tc, "function", None)
    if fn is None:
        return ParseError(
            stage="schema_invalid",
            message="native tool_call missing 'function' field",
            raw=str(tc),
        )
    name = _get(fn, "name", None)
    if not isinstance(name, str) or not name:
        return ParseError(
            stage="schema_invalid",
            message="native tool_call missing function name",
            raw=str(tc),
        )
    args_raw = _get(fn, "arguments", {})
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError as e:
            return ParseError(
                stage="json_invalid",
                message=f"native tool call arguments were not valid JSON: {e}",
                raw=args_raw,
            )
    elif isinstance(args_raw, dict):
        args = args_raw
    elif args_raw is None:
        args = {}
    else:
        return ParseError(
            stage="schema_invalid",
            message=f"native tool call arguments must be object, got {type(args_raw).__name__}",
            raw=str(args_raw),
        )

    tc_id = _get(tc, "id", None)
    if not isinstance(tc_id, str) or not tc_id:
        tc_id = f"call_{uuid.uuid4().hex[:12]}"
    return ToolCall(id=tc_id, name=name, args=args)


def _to_ollama_message(m: Message) -> dict[str, Any]:
    """Convert a `Message` into the dict shape `ollama.AsyncClient.chat` expects."""
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        out["tool_calls"] = [
            {"function": {"name": tc.name, "arguments": tc.args}}
            for tc in m.tool_calls
        ]
    if m.role == "tool":
        if m.tool_call_id:
            out["tool_call_id"] = m.tool_call_id
        if m.tool_name:
            out["name"] = m.tool_name
    return out


def _inject_tool_block(
    messages: list[Message], tools: list[ToolSchema]
) -> list[Message]:
    """Append a prompted-mode tool description block to the system message.

    If the first message is already a system message, append to it. Otherwise
    synthesize a new system message at position 0. Some models handle two
    consecutive system messages badly, so we never emit two.
    """
    if not tools:
        return messages
    block = render_prompted_block(tools)
    if messages and messages[0].role == "system":
        head = messages[0].model_copy(
            update={"content": (messages[0].content + "\n\n" + block).strip()}
        )
        return [head] + messages[1:]
    return [Message(role="system", content=block)] + list(messages)


def _build_done_chunk(
    *,
    messages: list[Message],
    completion_chars: int,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    start: float,
    end: float,
    first_content_at: float | None,
) -> ChatChunk:
    """Build the terminal `done=True` chunk, falling back to estimated
    token counts when the backend didn't report them."""
    estimated = False
    if prompt_tokens is None or prompt_tokens == 0:
        prompt_tokens = sum(estimate_tokens(m.content) for m in messages)
        estimated = True
    if completion_tokens is None or completion_tokens == 0:
        completion_tokens = max(1, completion_chars // 4) if completion_chars else 0
        estimated = True
    return ChatChunk(
        done=True,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_latency_ms=(end - start) * 1000.0,
        ttft_ms=((first_content_at - start) * 1000.0)
        if first_content_at is not None
        else None,
        tokens_estimated=estimated,
    )


# =============================== prompted-mode parser ===============================

@dataclass
class _ParseEvent:
    kind: Literal["text", "tool_call", "parse_error"]
    text: str = ""
    call: ToolCall | None = None
    error: ParseError | None = None


@dataclass
class _PromptedFenceParser:
    """Streams text and extracts ```json ...``` fenced blocks as tool calls.

    State machine with two states (outside/inside fence) and a hold-back
    buffer to handle a fence opener that straddles a delta boundary.
    """

    FENCE_OPEN: ClassVar[str] = "```json"
    FENCE_CLOSE: ClassVar[str] = "```"

    _buf: str = ""
    _inside_fence: bool = False
    _fence_buf: str = ""

    def feed(self, delta: str, tool_names: set[str]) -> Iterator[_ParseEvent]:
        if delta:
            self._buf += delta
        while True:
            if not self._inside_fence:
                idx = self._buf.find(self.FENCE_OPEN)
                if idx == -1:
                    # Hold back the last (len(FENCE_OPEN)-1) chars in case
                    # the next delta completes the opener.
                    safe = max(0, len(self._buf) - (len(self.FENCE_OPEN) - 1))
                    if safe > 0:
                        yield _ParseEvent(kind="text", text=self._buf[:safe])
                        self._buf = self._buf[safe:]
                    return
                else:
                    if idx > 0:
                        yield _ParseEvent(kind="text", text=self._buf[:idx])
                    self._buf = self._buf[idx + len(self.FENCE_OPEN):]
                    if self._buf.startswith("\n"):
                        self._buf = self._buf[1:]
                    self._inside_fence = True
                    self._fence_buf = ""
            else:
                idx = self._buf.find(self.FENCE_CLOSE)
                if idx == -1:
                    # Hold back the last (len(FENCE_CLOSE)-1) chars of _buf
                    # in case they're the start of a close marker that will
                    # complete in the next delta.
                    safe = max(0, len(self._buf) - (len(self.FENCE_CLOSE) - 1))
                    if safe > 0:
                        self._fence_buf += self._buf[:safe]
                        self._buf = self._buf[safe:]
                    return
                else:
                    self._fence_buf += self._buf[:idx]
                    self._buf = self._buf[idx + len(self.FENCE_CLOSE):]
                    self._inside_fence = False
                    yield from self._parse_fence(self._fence_buf, tool_names)
                    self._fence_buf = ""

    def flush(self) -> Iterator[_ParseEvent]:
        if self._inside_fence:
            yield _ParseEvent(
                kind="parse_error",
                error=ParseError(
                    stage="fence_unterminated",
                    message="stream ended inside an unterminated ```json block",
                    raw=self._fence_buf,
                ),
            )
            self._inside_fence = False
            self._fence_buf = ""
            return
        if self._buf:
            yield _ParseEvent(kind="text", text=self._buf)
            self._buf = ""

    def _parse_fence(
        self, content: str, tool_names: set[str]
    ) -> Iterator[_ParseEvent]:
        try:
            obj = json.loads(content.strip())
        except json.JSONDecodeError as e:
            yield _ParseEvent(
                kind="parse_error",
                error=ParseError(
                    stage="json_invalid", message=str(e), raw=content
                ),
            )
            return
        if not isinstance(obj, dict) or "tool" not in obj or "args" not in obj:
            yield _ParseEvent(
                kind="parse_error",
                error=ParseError(
                    stage="schema_invalid",
                    message="expected {'tool': ..., 'args': {...}}",
                    raw=content,
                ),
            )
            return
        name = obj["tool"]
        if not isinstance(name, str):
            yield _ParseEvent(
                kind="parse_error",
                error=ParseError(
                    stage="schema_invalid",
                    message=f"'tool' must be a string, got {type(name).__name__}",
                    raw=content,
                ),
            )
            return
        if name not in tool_names:
            yield _ParseEvent(
                kind="parse_error",
                error=ParseError(
                    stage="unknown_tool",
                    message=f"model called unknown tool '{name}'",
                    raw=content,
                ),
            )
            return
        args = obj["args"]
        if not isinstance(args, dict):
            yield _ParseEvent(
                kind="parse_error",
                error=ParseError(
                    stage="schema_invalid",
                    message=f"'args' must be an object, got {type(args).__name__}",
                    raw=content,
                ),
            )
            return
        yield _ParseEvent(
            kind="tool_call",
            call=ToolCall(
                id=f"call_{uuid.uuid4().hex[:12]}", name=name, args=args
            ),
        )
