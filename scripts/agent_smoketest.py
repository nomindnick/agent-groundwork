"""Phase 3 smoketest. Runs one user message through the full agent loop.

`print` is allowed here because scripts/CLI are exempt from the no-print
rule that applies to the agent core.

Usage examples:

  # Default: write hello.md and stop
  python scripts/agent_smoketest.py

  # Verify Criterion 2 (iteration cap)
  python scripts/agent_smoketest.py --max-iterations 2 \
      --prompt "List the files, then read every one, then write a summary"

  # Verify Criterion 3 (compaction)
  python scripts/agent_smoketest.py --trigger-messages 4 \
      --prompt "List the files, read each one, then write summary.md"

Uses the real sandbox from `config.toml` so the trace lands in the
configured location and you can inspect `./sandbox/hello.md` after the run.
The `ask_user` tool is wired to the stub provider — this smoketest does
not pause for input.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid

from agent_groundwork.agent.compaction import RollingSummaryCompactor
from agent_groundwork.agent.events import (
    CompactionEvent,
    Done,
    Error,
    SystemPromptEdited,
    TextChunk,
    ToolCallResult,
    ToolCallStarted,
)
from agent_groundwork.agent.loop import Agent
from agent_groundwork.config import load_config
from agent_groundwork.providers.ollama import OllamaProvider
from agent_groundwork.tools import build_default_registry, stub_user_input_provider
from agent_groundwork.tracing import Tracer


DEFAULT_PROMPT = (
    "Write a file called hello.md containing just the word 'hi', then stop."
)


async def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3 agent smoketest.")
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--model", default=None, help="Override config.model.name")
    ap.add_argument(
        "--mode",
        choices=["native", "prompted"],
        default=None,
        help="Override config.model.tool_call_mode",
    )
    ap.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="User message to send to the agent",
    )
    ap.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override config.agent.max_iterations (use to verify iteration cap)",
    )
    ap.add_argument(
        "--trigger-messages",
        type=int,
        default=None,
        help="Override config.compaction.trigger_messages (use to force compaction)",
    )
    ap.add_argument(
        "--trigger-tokens",
        type=int,
        default=None,
        help="Override config.compaction.trigger_tokens",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = args.model or cfg.model.name
    mode = args.mode or cfg.model.tool_call_mode
    max_iterations = (
        args.max_iterations
        if args.max_iterations is not None
        else cfg.agent.max_iterations
    )
    trigger_messages = (
        args.trigger_messages
        if args.trigger_messages is not None
        else cfg.compaction.trigger_messages
    )
    trigger_tokens = (
        args.trigger_tokens
        if args.trigger_tokens is not None
        else cfg.compaction.trigger_tokens
    )
    session_id = "smoketest-" + uuid.uuid4().hex[:8]

    print(f"[smoketest] session_id={session_id}")
    print(f"[smoketest] model={model} mode={mode}")
    print(f"[smoketest] sandbox={cfg.agent.sandbox_root}")
    print(f"[smoketest] max_iterations={max_iterations} trigger_messages={trigger_messages}")

    provider = OllamaProvider(
        host=cfg.provider.host,
        keep_alive=cfg.provider.ollama.keep_alive,
    )
    registry = build_default_registry(
        sandbox_root=cfg.agent.sandbox_root,
        user_input_provider=stub_user_input_provider,
    )
    summarization_model = cfg.compaction.summarization_model or model
    compactor = RollingSummaryCompactor(
        provider=provider,
        summarization_model=summarization_model,
        trigger_messages=trigger_messages,
        trigger_tokens=trigger_tokens,
        recent_window=cfg.compaction.recent_window,
        summary_dir=cfg.compaction.summary_dir,
        session_id=session_id,
    )

    exit_code = 0
    with Tracer(cfg.tracing.trace_dir, session_id) as tracer:
        print(f"[smoketest] trace={tracer.path}")
        agent = Agent(
            provider=provider,
            tool_registry=registry,
            compactor=compactor,
            tracer=tracer,
            system_prompt_path=cfg.agent.system_prompt_path,
            model=model,
            tool_call_mode=mode,
            max_iterations=max_iterations,
            session_id=session_id,
        )

        print(f"[smoketest] user: {args.prompt}")
        print("[smoketest] ---- events ----")

        try:
            async for event in agent.run(args.prompt):
                if isinstance(event, TextChunk):
                    sys.stdout.write(event.text)
                    sys.stdout.flush()
                elif isinstance(event, ToolCallStarted):
                    print(
                        f"\n[tool_call_started] {event.name}({event.args})"
                    )
                elif isinstance(event, ToolCallResult):
                    status = "ok" if event.result.ok else "FAIL"
                    print(
                        f"[tool_call_result] {event.name} {status} "
                        f"data={event.result.data} error={event.result.error}"
                    )
                elif isinstance(event, CompactionEvent):
                    print(
                        f"\n[compaction] {event.summary_path} "
                        f"({event.pre_message_count}->{event.post_message_count})"
                    )
                elif isinstance(event, SystemPromptEdited):
                    print(f"\n[system_prompt_edited]\n{event.diff}")
                elif isinstance(event, Done):
                    print(
                        f"\n[done] final_text_len={len(event.final_text)} chars"
                    )
                elif isinstance(event, Error):
                    print(
                        f"\n[error] recoverable={event.recoverable} "
                        f"message={event.message}"
                    )
                    exit_code = 1
        except KeyboardInterrupt:
            print("\n[smoketest] interrupted")
            exit_code = 130

    print(f"[smoketest] complete — trace at {tracer.path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
