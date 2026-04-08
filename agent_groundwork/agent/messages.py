"""Message construction helpers for the agent loop.

This module deliberately does NOT define a parallel agent-side Message
type. The agent loop reuses `agent_groundwork.providers.base.Message`
directly — it already has every field the loop needs (`role`, `content`,
`tool_calls`, `tool_call_id`, `tool_name`) and `OllamaProvider` already
knows how to convert it to the wire format. A second type would be pure
ceremony.

What lives here are small constructor helpers that keep the loop free of
inline JSON and `Message(role=..., content=...)` boilerplate.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_groundwork.providers.base import Message, ToolCall
from agent_groundwork.tools.base import ToolResult


def load_system_prompt(path: Path) -> str:
    """Read AGENT.md (or any system-prompt file) into a string.

    Caller is responsible for path resolution. Raises whatever the
    underlying filesystem call raises (FileNotFoundError, PermissionError,
    UnicodeDecodeError, ...) — the agent constructor lets these surface
    so a broken config fails fast.
    """
    return path.read_text(encoding="utf-8")


def system_message(content: str) -> Message:
    return Message(role="system", content=content)


def user_message(content: str) -> Message:
    return Message(role="user", content=content)


def assistant_message(text: str, tool_calls: list[ToolCall]) -> Message:
    """Construct an assistant turn from streamed text + finalized tool calls."""
    return Message(role="assistant", content=text, tool_calls=tool_calls)


def tool_result_message(call: ToolCall, result: ToolResult) -> Message:
    """Wrap a `ToolResult` as a `role="tool"` Message ready for history.

    The content is a JSON-serialized `ToolResult.model_dump()`. The model
    sees a structured `{"ok": ..., "data": ..., "error": ...}` object,
    which preserves the `ok` flag the model needs to decide whether to
    retry or report an error.

    `default=str` is a safety net for any non-JSON-serializable payload
    that snuck into `data` (Pydantic's `Any` doesn't enforce serializability).
    `ensure_ascii=False` so non-ASCII content (filenames, note bodies) round-trips
    cleanly.
    """
    payload = json.dumps(
        result.model_dump(),
        default=str,
        ensure_ascii=False,
    )
    return Message(
        role="tool",
        content=payload,
        tool_call_id=call.id,
        tool_name=call.name,
    )


def build_initial_history(system_prompt: str, first_user_message: str) -> list[Message]:
    """Build a fresh two-message history (system + first user turn)."""
    return [
        system_message(system_prompt),
        user_message(first_user_message),
    ]
