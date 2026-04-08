"""Agent event stream types.

The agent loop yields a sequence of `Event`s through its async generator.
Frontends (CLI in v1, web/Telegram in future phases) consume the stream and
render whatever is appropriate for their medium. The tracer also writes
each event to the JSONL trace.

Each event is a Pydantic `BaseModel` with a `type: Literal[...]` tag, and
`Event` is a discriminated `Union` over them. Rationale:
  - Free serialization (`event.model_dump()`) for the tracer.
  - `isinstance` dispatch still works for consumers.
  - Forward-compat with future SSE / wire formats: Pydantic round-trips
    the discriminator without custom code.

Note: this module's `ToolCallResult` is the *event* class. The payload it
carries (`ToolCallResult.result`) is `agent_groundwork.tools.base.ToolResult`,
a different type. Files that import both should alias on import.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

from agent_groundwork.tools.base import ToolResult


class TextChunk(BaseModel):
    """A streamed slice of model-generated text."""

    type: Literal["text_chunk"] = "text_chunk"
    text: str


class ToolCallStarted(BaseModel):
    """The agent is about to invoke a tool. Emitted before dispatch."""

    type: Literal["tool_call_started"] = "tool_call_started"
    name: str
    args: dict[str, Any]
    call_id: str


class ToolCallResult(BaseModel):
    """A tool call has finished (successfully or not).

    `result.ok` distinguishes success from failure; the model uses the same
    flag to decide whether to retry or report.
    """

    type: Literal["tool_call_result"] = "tool_call_result"
    name: str
    call_id: str
    result: ToolResult


class CompactionEvent(BaseModel):
    """The compactor just rewrote conversation history."""

    type: Literal["compaction"] = "compaction"
    summary_path: str
    pre_message_count: int
    post_message_count: int


class SystemPromptEdited(BaseModel):
    """The model edited AGENT.md (its own system prompt). Loud on purpose."""

    type: Literal["system_prompt_edited"] = "system_prompt_edited"
    diff: str


class Done(BaseModel):
    """The agent finished a turn cleanly with a final text response."""

    type: Literal["done"] = "done"
    final_text: str


class Error(BaseModel):
    """Something went wrong.

    `recoverable=True` means the user can sensibly try again (e.g. the
    model emitted only parse errors). `recoverable=False` means the loop
    has stopped because of a structural problem (e.g. iteration cap or
    provider exception).
    """

    type: Literal["error"] = "error"
    message: str
    recoverable: bool


Event = Annotated[
    Union[
        TextChunk,
        ToolCallStarted,
        ToolCallResult,
        CompactionEvent,
        SystemPromptEdited,
        Done,
        Error,
    ],
    Field(discriminator="type"),
]
