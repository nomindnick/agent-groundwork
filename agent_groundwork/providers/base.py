"""Provider abstraction — types, Protocol, and shared utilities.

This module is pure types. No I/O, no Ollama-specific code. Concrete
providers (e.g. `providers/ollama.py`) implement the `Provider` Protocol
defined here.

Two tool-call modes are supported by every provider:
  - "native"   — uses the backend's built-in tool-calling mechanism
  - "prompted" — injects tool descriptions into the system prompt and
                 parses fenced JSON blocks out of the model's free-form output

Both modes return the same `ChatChunk` / `ChatResult` shapes so consumers
(agent loop, bakeoff) don't care which one was used.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Literal, Protocol

from pydantic import BaseModel, Field


# --------------------------- message types ---------------------------

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    """A model-emitted call to a tool. Provider-agnostic shape."""

    id: str
    name: str
    args: dict[str, Any]


class Message(BaseModel):
    """A single conversation turn.

    For role="tool", `tool_call_id` matches the id of the assistant tool_call
    being replied to, and `content` carries the tool's serialized result.
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None


# --------------------------- tool schema ---------------------------

class ToolSchema(BaseModel):
    """Provider-facing tool definition.

    `parameters` is a JSON Schema dict (Pydantic v2's `model_json_schema()`
    output, typically). The provider renders this into either the backend's
    native tool-call format or a prompted markdown block.
    """

    name: str
    description: str
    parameters: dict[str, Any]


# --------------------------- parse errors ---------------------------

class ParseError(BaseModel):
    """Surfaced when prompted-mode tool-call parsing fails.

    The bakeoff and the loop both want to know not just *that* parsing
    failed but *why* — `stage` is a small enum that supports clean
    aggregation in reports.
    """

    stage: Literal[
        "fence_unterminated",
        "json_invalid",
        "schema_invalid",
        "unknown_tool",
    ]
    message: str
    raw: str


# --------------------------- streaming I/O ---------------------------

class ChatChunk(BaseModel):
    """One incremental piece of a streaming chat response.

    A chunk carries one of:
      - `text`        : a text delta
      - `tool_call`   : a finalized tool call
      - `parse_error` : a prompted-mode parse failure
      - `done=True`   : the terminal chunk, carrying timing/token totals

    Multiple non-empty fields are allowed in principle but the provider
    implementations emit one fact per chunk to keep consumer code simple.
    """

    text: str = ""
    tool_call: ToolCall | None = None
    parse_error: ParseError | None = None
    done: bool = False

    # Populated only when done=True:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_latency_ms: float | None = None
    ttft_ms: float | None = None
    tokens_estimated: bool = False


class ChatResult(BaseModel):
    """Aggregate result returned by `Provider.chat()`.

    Equivalent to draining `Provider.stream()` and accumulating into one
    object. Use `stream()` directly if you need TTFT or want to render text
    as it arrives.
    """

    text: str
    tool_calls: list[ToolCall]
    parse_errors: list[ParseError]
    prompt_tokens: int | None
    completion_tokens: int | None
    total_latency_ms: float
    ttft_ms: float | None
    tokens_estimated: bool


# --------------------------- protocol ---------------------------

class Provider(Protocol):
    async def chat(
        self,
        model: str,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        tool_call_mode: Literal["native", "prompted"] = "native",
    ) -> ChatResult: ...

    def stream(
        self,
        model: str,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        tool_call_mode: Literal["native", "prompted"] = "native",
    ) -> AsyncIterator[ChatChunk]: ...


# --------------------------- exceptions ---------------------------

class ProviderError(Exception):
    """Base class for any provider-layer failure."""


class ProviderConnectionError(ProviderError):
    """Could not reach the backend."""


class ProviderModelNotFoundError(ProviderError):
    """The requested model is not installed/available on the backend."""


class ProviderToolModeUnsupportedError(ProviderError):
    """Native tool-call mode was requested against a model that does not
    support it. Bakeoff treats this as a clean cell-level failure rather
    than a crash."""


# --------------------------- shared utilities ---------------------------

def estimate_tokens(text: str) -> int:
    """Cheap token estimate used when the backend doesn't report counts.

    Famously inaccurate for non-English text and code, but adequate for
    relative comparisons in the bakeoff. Anywhere this estimate is used,
    the carrying `ChatChunk`/`ChatResult` sets `tokens_estimated=True` so
    consumers know not to treat it as ground truth.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)
