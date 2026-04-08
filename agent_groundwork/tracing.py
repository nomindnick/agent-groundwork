"""Append-only JSONL session trace writer.

The Tracer is the primary debugging artifact for the agent. Every notable
event during a session — provider calls, tool calls, compactions, agent
events, parse errors — gets one JSON line in
`<trace_dir>/<UTC-timestamp>-<session_id>.jsonl`.

Append-only, line-oriented, no buffering games. Flushes after every write
so a crashed process still has a complete trace up to the moment of
failure. Re-readable with `jq` and zero project dependencies.

The class is a context manager. Typical usage:

    with Tracer(cfg.tracing.trace_dir, session_id) as tracer:
        agent = Agent(..., tracer=tracer, ...)
        async for event in agent.run(prompt):
            ...

The tracer's lifecycle is owned by the *caller* of the agent loop, not by
the loop itself. The smoketest and the future CLI both wrap construction
in a `with` block.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from pydantic import BaseModel


class Tracer:
    """JSONL session trace writer."""

    def __init__(self, trace_dir: Path, session_id: str) -> None:
        self._session_id = session_id
        trace_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self._path = trace_dir / f"{ts}-{session_id}.jsonl"
        self._fp: TextIO = self._path.open("a", encoding="utf-8")
        self._closed = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def session_id(self) -> str:
        return self._session_id

    # =============================== core ===============================

    def write(self, event_type: str, payload: dict[str, Any]) -> None:
        """Write one row. `payload` is merged into the row at top level.

        Every row carries `ts`, `session_id`, `event_type` plus the
        payload's keys. Nothing buffered; every call flushes.
        """
        if self._closed:
            return
        row: dict[str, Any] = {
            "ts": time.time(),
            "session_id": self._session_id,
            "event_type": event_type,
            **payload,
        }
        self._fp.write(
            json.dumps(row, default=str, ensure_ascii=False) + "\n"
        )
        self._fp.flush()

    # =============================== convenience ===============================

    def log_provider_call(
        self,
        *,
        model: str,
        mode: str,
        message_count: int,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_latency_ms: float | None,
        ttft_ms: float | None,
        tokens_estimated: bool,
        tool_call_count: int,
        parse_error_count: int,
    ) -> None:
        self.write(
            "provider_call",
            {
                "model": model,
                "mode": mode,
                "message_count": message_count,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_latency_ms": total_latency_ms,
                "ttft_ms": ttft_ms,
                "tokens_estimated": tokens_estimated,
                "tool_call_count": tool_call_count,
                "parse_error_count": parse_error_count,
            },
        )

    def log_tool_call(
        self,
        *,
        name: str,
        call_id: str,
        args: dict[str, Any],
        result: BaseModel,
        latency_ms: float,
    ) -> None:
        """Log a tool dispatch + result. `result` is typed as BaseModel to
        avoid a circular import on `tools.base.ToolResult`."""
        self.write(
            "tool_call",
            {
                "name": name,
                "call_id": call_id,
                "args": args,
                "result": result.model_dump(),
                "latency_ms": latency_ms,
            },
        )

    def log_compaction(
        self,
        *,
        pre_snapshot: list[dict[str, Any]],
        post_snapshot: list[dict[str, Any]],
        summary_text: str,
        summary_path: str,
    ) -> None:
        self.write(
            "compaction",
            {
                "pre_snapshot": pre_snapshot,
                "post_snapshot": post_snapshot,
                "summary_text": summary_text,
                "summary_path": summary_path,
            },
        )

    def log_agent_event(self, event: BaseModel) -> None:
        """Log an Event from `agent_groundwork.agent.events`.

        The event_type is derived from the event's `type` discriminator
        and prefixed with `agent_event.` so it groups cleanly in jq queries.
        """
        dumped = event.model_dump()
        kind = dumped.get("type", "unknown")
        self.write(f"agent_event.{kind}", dumped)

    def log_parse_error(self, stage: str, message: str, raw: str) -> None:
        self.write(
            "parse_error",
            {"stage": stage, "message": message, "raw": raw},
        )

    # =============================== lifecycle ===============================

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._fp.close()
        finally:
            self._closed = True

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
