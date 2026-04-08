"""Conversation compaction.

The compactor manages conversation context as it grows. As soon as the
history crosses a configurable threshold (message count or estimated
tokens), the compactor rewrites it: keep the system prompt, keep the most
recent K messages verbatim, and replace everything in between with a
synthetic-system summary message.

The summary itself is produced by a one-shot call to the configured
summarization model (defaults to the main loop model when empty in
config). The summary is also persisted to disk at
`<summary_dir>/<session_id>/summary.md` so it survives the process and
can be inspected by hand.

Subsequent compactions are *telescoping*: the previous summary text is
prepended as context to the new summarization call, so information
accumulates rather than resetting. The on-disk `summary.md` is
overwritten on each compaction; the trace JSONL captures every version
for audit.

Only one strategy ships in v1: `RollingSummaryCompactor`. The protocol is
defined separately so a future strategy can drop in without touching the
loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from agent_groundwork.providers.base import (
    Message,
    Provider,
    estimate_tokens,
)


# =============================== protocol ===============================


@dataclass
class CompactionResult:
    """Everything the loop needs after a compaction.

    The history is what gets fed into the next provider call. The
    snapshots and summary text feed the trace and the `CompactionEvent`.
    """

    history: list[Message]
    summary_path: Path
    pre_snapshot: list[Message]
    post_snapshot: list[Message]
    summary_text: str


class Compactor(Protocol):
    def should_compact(self, history: list[Message]) -> bool: ...
    async def compact(self, history: list[Message]) -> CompactionResult: ...


# =============================== summarizer prompt ===============================


SUMMARIZATION_SYSTEM_PROMPT = """You are summarizing the middle of a conversation so that an assistant can continue with limited context.

Produce a structured markdown summary with these sections (omit any that do not apply):

## User goals
Bullet list of what the user has asked for across the conversation.

## Decisions made
What the assistant committed to or did.

## Files created or modified
For each file, one line: `path` — short description.

## Open threads
Anything still in progress, pending, or unanswered.

Be dense. No preamble. No apologies. Markdown only."""


# =============================== token estimate ===============================


def _message_token_estimate(m: Message) -> int:
    """Rough token estimate for a full Message, including any tool calls.

    `Message.content` is just the text body — assistant turns that only
    emit tool calls have `content=""` and their payload lives in
    `tool_calls`. Counting content alone would systematically under-count
    tool-heavy conversations (the same class of "only looked at .content"
    bug that commit adf0483 fixed in the provider layer).

    The serialized tool_calls approximate what actually goes on the wire
    to Ollama's chat template; good enough for a compaction heuristic.
    """
    total = estimate_tokens(m.content)
    if m.tool_calls:
        for tc in m.tool_calls:
            total += estimate_tokens(tc.name)
            try:
                total += estimate_tokens(
                    json.dumps(tc.args, ensure_ascii=False, default=str)
                )
            except (TypeError, ValueError):
                total += estimate_tokens(str(tc.args))
    return total


# =============================== default impl ===============================


class RollingSummaryCompactor:
    """Default v1 compactor: keep system + last K, summarize the middle.

    Trigger: `len(history) > trigger_messages` OR estimated total tokens
    > `trigger_tokens` (whichever first).

    On compact:
        new_history = [system?, synthetic-system(summary), *history[-recent_window:]]

    Telescoping: the prior summary is fed back into the next summarization
    call as context, so information accumulates rather than resetting.
    """

    def __init__(
        self,
        *,
        provider: Provider,
        summarization_model: str,
        trigger_messages: int,
        trigger_tokens: int,
        recent_window: int,
        summary_dir: Path,
        session_id: str,
    ) -> None:
        self._provider = provider
        self._summarization_model = summarization_model
        self._trigger_messages = trigger_messages
        self._trigger_tokens = trigger_tokens
        self._recent_window = recent_window
        self._summary_dir = summary_dir
        self._session_id = session_id
        self._prior_summary: str = ""

    # --------------------------- gate ---------------------------

    def should_compact(self, history: list[Message]) -> bool:
        if len(history) > self._trigger_messages:
            return True
        total_tokens = sum(_message_token_estimate(m) for m in history)
        return total_tokens > self._trigger_tokens

    # --------------------------- main entry ---------------------------

    async def compact(self, history: list[Message]) -> CompactionResult:
        # Snapshot before mutation for trace audit.
        pre_snapshot = [m.model_copy(deep=True) for m in history]

        # Preserve the system message verbatim.
        if history and history[0].role == "system":
            system_msg: Message | None = history[0]
            body = history[1:]
        else:
            system_msg = None
            body = list(history)

        # Defensive no-op: if the body is shorter than the recent window,
        # there's nothing to summarize. Should not normally happen if
        # `should_compact` gated us, but be safe.
        if len(body) <= self._recent_window:
            return CompactionResult(
                history=history,
                summary_path=Path(""),  # sentinel: no file written
                pre_snapshot=pre_snapshot,
                post_snapshot=pre_snapshot,
                summary_text="",
            )

        to_summarize = body[: -self._recent_window]
        recent = body[-self._recent_window :]

        summary_text = await self._summarize(to_summarize)

        summary_path = self._write_summary_file(summary_text)

        # Update telescoping accumulator for the *next* compaction.
        self._prior_summary = summary_text

        new_history: list[Message] = []
        if system_msg is not None:
            new_history.append(system_msg)
        new_history.append(
            Message(
                role="system",
                content=(
                    "Earlier in this conversation (auto-summarized):\n\n"
                    + summary_text
                ),
            )
        )
        new_history.extend(recent)

        post_snapshot = [m.model_copy(deep=True) for m in new_history]

        return CompactionResult(
            history=new_history,
            summary_path=summary_path,
            pre_snapshot=pre_snapshot,
            post_snapshot=post_snapshot,
            summary_text=summary_text,
        )

    # --------------------------- internals ---------------------------

    async def _summarize(self, messages_to_summarize: list[Message]) -> str:
        """Render a transcript and ask the summarization model to summarize it."""
        transcript_lines: list[str] = []
        for m in messages_to_summarize:
            prefix = f"[{m.role}]"
            if m.tool_name:
                prefix += f" {m.tool_name}"
            transcript_lines.append(f"{prefix} {m.content}".rstrip())
        transcript = "\n\n".join(transcript_lines)

        user_content = ""
        if self._prior_summary:
            user_content += (
                "Previous summary of earlier conversation:\n\n"
                + self._prior_summary
                + "\n\n---\n\n"
            )
        user_content += "Summarize this continuation:\n\n" + transcript

        summarize_messages = [
            Message(role="system", content=SUMMARIZATION_SYSTEM_PROMPT),
            Message(role="user", content=user_content),
        ]

        result = await self._provider.chat(
            model=self._summarization_model,
            messages=summarize_messages,
            tools=None,
            tool_call_mode="native",
        )
        return result.text.strip()

    def _write_summary_file(self, summary_text: str) -> Path:
        session_dir = self._summary_dir / self._session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        path = session_dir / "summary.md"
        path.write_text(summary_text, encoding="utf-8")
        return path
