"""The agent loop.

Construct an `Agent` once per session. Call `run(user_message)` to
process one user turn; the method is an async generator yielding `Event`s
(see `agent.events`). Each `run()` call appends to the same in-memory
history, so the same `Agent` instance can drive a multi-turn REPL — the
CLI in Phase 4 will do exactly that.

The loop's contract:
  - All output flows through yielded events. No `print()` calls anywhere.
  - Tools never raise — the dispatcher catches every exception and wraps
    it in `ToolResult(ok=False, error=...)` so the model can recover.
  - Provider exceptions are fatal for the current turn and produce an
    `Error(recoverable=False)`.
  - Hitting the iteration cap also produces `Error(recoverable=False)`.
  - When the model emits ONLY parse errors (no text, no valid tool calls),
    the loop produces `Error(recoverable=True)` and returns. Phase 5 may
    revisit this if a retry-with-feedback turns out to help small models.
  - When the model edits AGENT.md (its own system prompt), the loop emits
    `SystemPromptEdited` AND reloads `history[0]` so the edit takes effect
    immediately within the current session.
"""

from __future__ import annotations

import difflib
import inspect
import time
from pathlib import Path
from typing import AsyncIterator, Literal

from pydantic import ValidationError

from agent_groundwork.agent.compaction import Compactor
from agent_groundwork.agent.events import (
    CompactionEvent,
    Done,
    Error,
    Event,
    SystemPromptEdited,
    TextChunk,
    ToolCallStarted,
)
from agent_groundwork.agent.events import ToolCallResult as ToolCallResultEvent
from agent_groundwork.agent.messages import (
    assistant_message,
    load_system_prompt,
    tool_result_message,
    user_message,
)
from agent_groundwork.paths import validate_path
from agent_groundwork.providers.base import (
    ChatChunk,
    Message,
    ParseError,
    Provider,
    ToolCall,
)
from agent_groundwork.tools.base import ToolRegistry, ToolResult
from agent_groundwork.tracing import Tracer


class Agent:
    """Drives the model -> tool -> model loop for one session."""

    def __init__(
        self,
        *,
        provider: Provider,
        tool_registry: ToolRegistry,
        compactor: Compactor,
        tracer: Tracer,
        system_prompt_path: Path,
        model: str,
        tool_call_mode: Literal["native", "prompted"],
        max_iterations: int,
        session_id: str,
    ) -> None:
        self._provider = provider
        self._registry = tool_registry
        self._compactor = compactor
        self._tracer = tracer
        self._system_prompt_path = system_prompt_path
        self._model = model
        self._tool_call_mode = tool_call_mode
        self._max_iterations = max_iterations
        self._session_id = session_id

        # Eagerly load the system prompt so a broken path fails fast at
        # construction rather than mid-conversation.
        self._system_prompt_cache = load_system_prompt(system_prompt_path)
        self._history: list[Message] = [
            Message(role="system", content=self._system_prompt_cache)
        ]

    # =============================== public ===============================

    @property
    def history(self) -> list[Message]:
        """Read-only view of current history (for the CLI/smoketest)."""
        return list(self._history)

    async def run(self, user_message_text: str) -> AsyncIterator[Event]:
        """Run one user turn. Yields events until Done or Error."""
        self._history.append(user_message(user_message_text))

        for iteration in range(self._max_iterations):
            # 1. Compaction gate.
            if self._compactor.should_compact(self._history):
                async for ev in self._maybe_compact():
                    yield ev

            # 2. Stream the provider.
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            parse_errors: list[ParseError] = []
            last: ChatChunk | None = None

            try:
                async for chunk in self._provider.stream(
                    model=self._model,
                    messages=self._history,
                    tools=self._registry.to_schemas(),
                    tool_call_mode=self._tool_call_mode,
                ):
                    if chunk.text:
                        text_parts.append(chunk.text)
                        yield TextChunk(text=chunk.text)
                    if chunk.tool_call is not None:
                        tool_calls.append(chunk.tool_call)
                    if chunk.parse_error is not None:
                        parse_errors.append(chunk.parse_error)
                        self._tracer.log_parse_error(
                            stage=chunk.parse_error.stage,
                            message=chunk.parse_error.message,
                            raw=chunk.parse_error.raw,
                        )
                    if chunk.done:
                        last = chunk
            except Exception as e:  # noqa: BLE001 — provider failures are fatal
                err = f"provider stream failed: {type(e).__name__}: {e}"
                self._tracer.write("provider_error", {"error": err})
                err_event = Error(message=err, recoverable=False)
                self._tracer.log_agent_event(err_event)
                yield err_event
                return

            # 3. Trace the provider call.
            self._tracer.log_provider_call(
                model=self._model,
                mode=self._tool_call_mode,
                message_count=len(self._history),
                prompt_tokens=last.prompt_tokens if last else None,
                completion_tokens=last.completion_tokens if last else None,
                total_latency_ms=last.total_latency_ms if last else None,
                ttft_ms=last.ttft_ms if last else None,
                tokens_estimated=bool(last.tokens_estimated) if last else False,
                tool_call_count=len(tool_calls),
                parse_error_count=len(parse_errors),
            )

            # 4. Append the assistant turn to history.
            assistant_text = "".join(text_parts)
            self._history.append(
                assistant_message(assistant_text, tool_calls)
            )

            # 5. Parse-error-only branch: model produced nothing salvageable.
            if not tool_calls and not assistant_text.strip() and parse_errors:
                first = parse_errors[0]
                msg = (
                    f"model emitted only parse errors "
                    f"(stage={first.stage}): {first.message}"
                )
                err_event = Error(message=msg, recoverable=True)
                self._tracer.log_agent_event(err_event)
                yield err_event
                return

            # 6. No tool calls: agent is done with this turn.
            if not tool_calls:
                done_event = Done(final_text=assistant_text)
                self._tracer.log_agent_event(done_event)
                yield done_event
                return

            # 7. Dispatch tool calls in order, appending each result to history.
            for call in tool_calls:
                started = ToolCallStarted(
                    name=call.name, args=call.args, call_id=call.id
                )
                self._tracer.log_agent_event(started)
                yield started

                # Snapshot AGENT.md if this edit_file targets it.
                pre_system_prompt: str | None = None
                if self._targets_system_prompt(call):
                    pre_system_prompt = self._read_system_prompt_file()

                t_start = time.monotonic()
                result = await self._dispatch(call)
                latency_ms = (time.monotonic() - t_start) * 1000.0

                self._tracer.log_tool_call(
                    name=call.name,
                    call_id=call.id,
                    args=call.args,
                    result=result,
                    latency_ms=latency_ms,
                )

                result_event = ToolCallResultEvent(
                    name=call.name, call_id=call.id, result=result
                )
                self._tracer.log_agent_event(result_event)
                yield result_event

                self._history.append(tool_result_message(call, result))

                # System prompt edit detection: only relevant if the edit
                # succeeded and we actually took a snapshot.
                if (
                    result.ok
                    and pre_system_prompt is not None
                    and call.name == "edit_file"
                ):
                    post = self._read_system_prompt_file()
                    if post != pre_system_prompt:
                        diff = self._render_diff(pre_system_prompt, post)
                        edited = SystemPromptEdited(diff=diff)
                        self._tracer.log_agent_event(edited)
                        yield edited
                        # Reload the in-memory system message so the next
                        # provider call sees the updated prompt.
                        self._system_prompt_cache = post
                        if self._history and self._history[0].role == "system":
                            self._history[0] = Message(
                                role="system", content=post
                            )

        # 8. Iteration cap exhausted.
        cap_msg = f"hit iteration cap ({self._max_iterations})"
        cap_event = Error(message=cap_msg, recoverable=False)
        self._tracer.log_agent_event(cap_event)
        yield cap_event

    # =============================== internals ===============================

    async def _maybe_compact(self) -> AsyncIterator[Event]:
        """Run compaction, log it, yield CompactionEvent. Tolerates failures."""
        try:
            result = await self._compactor.compact(self._history)
        except Exception as e:  # noqa: BLE001 — compaction is advisory
            self._tracer.write(
                "compaction_error",
                {"error": f"{type(e).__name__}: {e}"},
            )
            return

        # Sentinel: empty path means the compactor short-circuited (history
        # was too small to bother). Don't emit an event in that case.
        if str(result.summary_path) == "" or not result.summary_text:
            return

        self._history = result.history
        self._tracer.log_compaction(
            pre_snapshot=[m.model_dump() for m in result.pre_snapshot],
            post_snapshot=[m.model_dump() for m in result.post_snapshot],
            summary_text=result.summary_text,
            summary_path=str(result.summary_path),
        )
        evt = CompactionEvent(
            summary_path=str(result.summary_path),
            pre_message_count=len(result.pre_snapshot),
            post_message_count=len(result.post_snapshot),
        )
        self._tracer.log_agent_event(evt)
        yield evt

    async def _dispatch(self, call: ToolCall) -> ToolResult:
        """Look up + validate args + run a tool. Never raises."""
        tool = self._registry.get(call.name)
        if tool is None:
            return ToolResult(
                ok=False,
                error=f"unknown tool: {call.name!r}",
            )

        try:
            args_model = tool.args_schema.model_validate(call.args)
        except ValidationError as e:
            return ToolResult(
                ok=False,
                error=f"invalid arguments for {call.name}: {e}",
            )

        try:
            out = tool.run(args_model)
            if inspect.isawaitable(out):
                result = await out
            else:
                result = out  # tools should be async; tolerate sync
            if not isinstance(result, ToolResult):
                return ToolResult(
                    ok=False,
                    error=(
                        f"tool {call.name} returned non-ToolResult: "
                        f"{type(result).__name__}"
                    ),
                )
            return result
        except Exception as e:  # noqa: BLE001 — dispatcher contract
            return ToolResult(
                ok=False,
                error=f"tool {call.name} raised {type(e).__name__}: {e}",
            )

    def _targets_system_prompt(self, call: ToolCall) -> bool:
        """True iff this edit_file call is targeting AGENT.md."""
        if call.name != "edit_file":
            return False
        target = call.args.get("path")
        if not isinstance(target, str):
            return False
        try:
            sandbox_root = self._system_prompt_path.parent
            full = validate_path(sandbox_root, target)
        except Exception:  # noqa: BLE001 — any path failure → not targeting
            return False
        try:
            return full.resolve() == self._system_prompt_path.resolve()
        except Exception:  # noqa: BLE001
            return False

    def _read_system_prompt_file(self) -> str:
        try:
            return self._system_prompt_path.read_text(encoding="utf-8")
        except OSError:
            return ""

    @staticmethod
    def _render_diff(before: str, after: str) -> str:
        return "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile="AGENT.md (before)",
                tofile="AGENT.md (after)",
            )
        )
