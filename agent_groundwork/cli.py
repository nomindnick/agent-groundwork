"""Phase 4 CLI frontend for agent-groundwork.

`python -m agent_groundwork` lands here. The CLI:

  - loads config and constructs the provider, tools, compactor, tracer, agent
  - runs an interactive REPL: prompt -> `agent.run()` -> render events
  - renders the agent's event stream with ANSI colors (auto-disabled when
    stdout is not a TTY, when `NO_COLOR` is set, or when `--no-color` is passed)
  - wires `ask_user` to a stdin-backed input provider so the agent can pause
    mid-turn and ask the user for clarification
  - handles Ctrl-C cleanly: cancels the in-flight turn and returns to the
    prompt instead of exiting the process
  - prints the trace file path on exit

`print()` is intentional in this module — the no-print rule applies only to
the agent core, not to CLI / scripts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from typing import Any, Awaitable, Callable

# Side-effect import: enables line editing and arrow-key history at the
# REPL prompt with zero dependencies.
import readline  # noqa: F401

from agent_groundwork.agent.compaction import RollingSummaryCompactor
from agent_groundwork.agent.events import (
    CompactionEvent,
    Done,
    Error,
    Event,
    SystemPromptEdited,
    TextChunk,
    ToolCallResult,
    ToolCallStarted,
)
from agent_groundwork.agent.loop import Agent
from agent_groundwork.config import load_config
from agent_groundwork.providers.ollama import OllamaProvider
from agent_groundwork.tools import build_default_registry
from agent_groundwork.tracing import Tracer


# ============================== ANSI helpers ==============================

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"

# Module-level toggle, set during `main()` based on --no-color, NO_COLOR, isatty.
_USE_COLOR = True


def _set_color_enabled(enabled: bool) -> None:
    global _USE_COLOR
    _USE_COLOR = enabled


def c(text: str, *codes: str) -> str:
    """Wrap `text` in ANSI codes (or return it unchanged if colors are off)."""
    if not _USE_COLOR or not codes:
        return text
    return "".join(codes) + text + RESET


# ============================== formatting helpers ==============================


def _truncate(s: str, limit: int = 80) -> str:
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def _format_args(args: dict[str, Any]) -> str:
    """Render a tool's args dict as a compact `k=v, k=v` string.

    Each value is JSON-encoded so nested dicts/lists print sensibly, then
    truncated to 80 chars so a `write_file` with a 5KB body doesn't drown
    the terminal.
    """
    parts: list[str] = []
    for k, v in args.items():
        try:
            v_str = json.dumps(v, ensure_ascii=False)
        except (TypeError, ValueError):
            v_str = repr(v)
        parts.append(f"{k}={_truncate(v_str)}")
    return ", ".join(parts)


def _format_tool_data(data: Any) -> str:
    if data is None:
        return "None"
    try:
        return _truncate(json.dumps(data, ensure_ascii=False))
    except (TypeError, ValueError):
        return _truncate(repr(data))


# ============================== event renderer ==============================


class _EventRenderer:
    """Renders the agent's event stream to the terminal.

    Stateful: tracks whether the last byte we wrote was a newline so that
    non-text events (which need their own line) can prepend `\\n` only when
    needed. Streaming `TextChunk`s flow through `sys.stdout.write` directly
    so the user sees tokens appear as they arrive.
    """

    def __init__(self) -> None:
        self._at_line_start = True

    # ----- public surface used by the REPL after a cancel -----

    def ensure_newline(self) -> None:
        if not self._at_line_start:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._at_line_start = True

    def cancel_warning(self) -> None:
        """Print a visible '[cancelled]' notice after Ctrl-C interrupts a turn."""
        self.ensure_newline()
        print(
            c(
                "[cancelled — previous turn left unfinished in history]",
                DIM,
                YELLOW,
            ),
            flush=True,
        )
        self._at_line_start = True

    # ----- main dispatch -----

    def render(self, event: Event) -> None:
        if isinstance(event, TextChunk):
            self._render_text(event)
            return
        if isinstance(event, ToolCallStarted):
            self._render_tool_started(event)
            return
        if isinstance(event, ToolCallResult):
            self._render_tool_result(event)
            return
        if isinstance(event, CompactionEvent):
            self._render_compaction(event)
            return
        if isinstance(event, SystemPromptEdited):
            self._render_system_prompt_edited(event)
            return
        if isinstance(event, Done):
            self._render_done(event)
            return
        if isinstance(event, Error):
            self._render_error(event)
            return

    # ----- per-event renderers -----

    def _render_text(self, event: TextChunk) -> None:
        if not event.text:
            return
        sys.stdout.write(event.text)
        sys.stdout.flush()
        self._at_line_start = event.text.endswith("\n")

    def _render_tool_started(self, event: ToolCallStarted) -> None:
        self.ensure_newline()
        line = c(
            f"[tool_call_started] {event.name}({_format_args(event.args)})",
            CYAN,
        )
        print(line, flush=True)
        self._at_line_start = True

    def _render_tool_result(self, event: ToolCallResult) -> None:
        self.ensure_newline()
        ok = event.result.ok
        status = c("ok", GREEN) if ok else c("FAIL", RED)
        data_repr = _format_tool_data(event.result.data)
        print(
            f"[tool_call_result] {event.name} {status} data={data_repr}",
            flush=True,
        )
        if event.result.error:
            print(c(f"  error: {event.result.error}", RED), flush=True)
        self._at_line_start = True

    def _render_compaction(self, event: CompactionEvent) -> None:
        self.ensure_newline()
        print(
            c(
                f"[compaction] {event.summary_path} "
                f"({event.pre_message_count}->{event.post_message_count})",
                DIM,
            ),
            flush=True,
        )
        self._at_line_start = True

    def _render_system_prompt_edited(self, event: SystemPromptEdited) -> None:
        self.ensure_newline()
        print(c("[system_prompt_edited]", BOLD, YELLOW), flush=True)
        # Print the FULL diff (not truncated — loud on purpose).
        for line in event.diff.splitlines():
            print(c(line, YELLOW), flush=True)
        self._at_line_start = True

    def _render_done(self, event: Done) -> None:
        self.ensure_newline()
        print(c(f"[done] ({len(event.final_text)} chars)", DIM), flush=True)
        self._at_line_start = True

    def _render_error(self, event: Error) -> None:
        self.ensure_newline()
        tag = "error" if event.recoverable else "ERROR"
        print(
            c(
                f"[{tag}] recoverable={event.recoverable} {event.message}",
                BOLD,
                RED,
            ),
            flush=True,
        )
        self._at_line_start = True


# ============================== ask_user provider ==============================


def _make_user_input_provider() -> Callable[[str], Awaitable[str]]:
    """Build the stdin-backed `ask_user` provider for the CLI.

    Notes / known caveats (Phase 5 may revisit):
      - We use `asyncio.to_thread(input, ...)` because the agent loop awaits
        this provider; the blocking read happens on a worker thread so the
        event loop stays responsive.
      - Edge case: if the user hits Ctrl-C while we're blocked on `input()`,
        the SIGINT lands on the main thread (which raises KeyboardInterrupt
        and unwinds the agent generator) but the worker thread stays
        blocked on stdin until something lands on it. The next Enter wakes
        the orphaned reader. Survivable; not perfect. If this bites in
        Phase 5, switch to a `loop.add_reader()` POSIX path.
    """

    async def cli_user_input_provider(question: str) -> str:
        sys.stdout.write("\n" + c("[ask_user] ", CYAN) + question + "\n")
        sys.stdout.flush()
        return await asyncio.to_thread(input, c("answer> ", DIM))

    return cli_user_input_provider


# ============================== argparse ==============================


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m agent_groundwork",
        description="Phase 4 CLI for agent-groundwork.",
    )
    p.add_argument(
        "--config",
        default="config.toml",
        help="Path to the config TOML (default: ./config.toml).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override config.model.name (e.g. for swapping finalists).",
    )
    p.add_argument(
        "--mode",
        choices=["native", "prompted"],
        default=None,
        help="Override config.model.tool_call_mode.",
    )
    p.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors (also auto-disabled when stdout isn't a TTY or NO_COLOR is set).",
    )
    p.add_argument(
        "--session-id",
        default=None,
        help="Override the auto-generated session id (useful for reproducible traces).",
    )
    return p


def _resolve_color_enabled(args: argparse.Namespace) -> bool:
    if args.no_color:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


# ============================== REPL loop ==============================


_HELP_TEXT = """\
Available commands:
  /quit, /exit   exit the CLI
  /help          show this message
  /trace         print the current trace file path

Keys:
  Enter on empty line   re-prompts (skips the turn)
  Ctrl-C during a turn  cancels the in-flight turn
  Ctrl-D at the prompt  exits
"""


async def _run_repl(
    agent: Agent,
    renderer: _EventRenderer,
    tracer: Tracer,
) -> None:
    print(c("Type /help for commands. Ctrl-D or /quit to exit.", DIM), flush=True)
    print(flush=True)

    while True:
        # Bare input() (NOT asyncio.to_thread) at the prompt: this lets
        # Ctrl-C raise KeyboardInterrupt synchronously and lets EOF (Ctrl-D)
        # raise EOFError, both of which we handle here. The REPL is
        # sequential anyway — there is no concurrent work to keep the
        # event loop responsive for.
        try:
            line = input(c("you> ", BOLD))
        except EOFError:
            sys.stdout.write("\n")
            return
        except KeyboardInterrupt:
            sys.stdout.write("\n")
            continue

        line = line.strip()
        if not line:
            continue

        # Slash commands.
        if line in ("/quit", "/exit"):
            return
        if line == "/help":
            sys.stdout.write(_HELP_TEXT)
            sys.stdout.flush()
            continue
        if line == "/trace":
            print(f"trace: {tracer.path}", flush=True)
            continue
        if line.startswith("/"):
            print(c(f"unknown command: {line}", RED), flush=True)
            continue

        # Run a single agent turn. KeyboardInterrupt cancels in-flight and
        # returns to the prompt; everything else (Done, Error, recoverable
        # or not) is handled by the renderer and we continue the REPL.
        gen = agent.run(line)
        try:
            async for event in gen:
                renderer.render(event)
        except KeyboardInterrupt:
            # Close the generator gracefully so the underlying Ollama HTTP
            # stream is cancelled and we don't leak a task at shutdown.
            await gen.aclose()
            renderer.cancel_warning()
            # Note: agent._history now contains the cancelled user message
            # without an assistant reply. The model can usually cope on the
            # next turn. Phase 5 may add Agent.rollback_last_turn() if this
            # becomes a real friction point.


# ============================== main ==============================


async def async_main(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    model = args.model or cfg.model.name
    mode = args.mode or cfg.model.tool_call_mode
    session_id = args.session_id or ("cli-" + uuid.uuid4().hex[:8])

    provider = OllamaProvider(
        host=cfg.provider.host,
        keep_alive=cfg.provider.ollama.keep_alive,
    )
    registry = build_default_registry(
        sandbox_root=cfg.agent.sandbox_root,
        user_input_provider=_make_user_input_provider(),
    )
    summarization_model = cfg.compaction.summarization_model or model
    compactor = RollingSummaryCompactor(
        provider=provider,
        summarization_model=summarization_model,
        trigger_messages=cfg.compaction.trigger_messages,
        trigger_tokens=cfg.compaction.trigger_tokens,
        recent_window=cfg.compaction.recent_window,
        summary_dir=cfg.compaction.summary_dir,
        session_id=session_id,
    )

    with Tracer(cfg.tracing.trace_dir, session_id) as tracer:
        agent = Agent(
            provider=provider,
            tool_registry=registry,
            compactor=compactor,
            tracer=tracer,
            system_prompt_path=cfg.agent.system_prompt_path,
            model=model,
            tool_call_mode=mode,
            max_iterations=cfg.agent.max_iterations,
            session_id=session_id,
        )
        renderer = _EventRenderer()

        # Startup banner.
        print(c("agent-groundwork CLI", BOLD), flush=True)
        print(f"  model:    {model} ({mode})", flush=True)
        print(f"  sandbox:  {cfg.agent.sandbox_root}", flush=True)
        print(f"  session:  {session_id}", flush=True)
        print(f"  trace:    {tracer.path}", flush=True)
        print(flush=True)

        try:
            await _run_repl(agent, renderer, tracer)
        finally:
            # Always remind the user where the trace landed, regardless of
            # how the REPL ended (clean exit, EOF, /quit, exception).
            print(flush=True)
            print(c(f"trace: {tracer.path}", DIM), flush=True)

    return 0


def main() -> int:
    args = build_arg_parser().parse_args()
    _set_color_enabled(_resolve_color_enabled(args))
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        # Top-level Ctrl-C (e.g. during shutdown). Conventional exit code.
        return 130
