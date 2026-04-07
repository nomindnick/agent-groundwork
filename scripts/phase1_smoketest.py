"""Phase 1 smoketest. Not a pytest test — a human-run demo.

Demonstrates the IMPLEMENTATION_PLAN's "done when" criteria for Phase 1:
  1. The sandbox guard rejects path escape.
  2. A registry-driven write_file inside the sandbox works.
  3. The provider can call any installed Ollama model in either tool-call mode.

Usage:
    python scripts/phase1_smoketest.py [--mode native|prompted] [--model NAME]

`print` is allowed here because scripts/CLI are exempt from the no-print rule
that applies to the agent core.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from agent_groundwork.config import load_config
from agent_groundwork.paths import PathEscapeError, validate_path
from agent_groundwork.providers.base import Message
from agent_groundwork.providers.ollama import OllamaProvider
from agent_groundwork.tools import build_default_registry, stub_user_input_provider
from agent_groundwork.tools.files import WriteFileArgs


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["native", "prompted"], default="native")
    ap.add_argument("--model", default=None, help="Override config.model.name")
    ap.add_argument("--config", default="config.toml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = args.model or cfg.model.name

    # 1. Sandbox containment.
    print("=" * 60)
    print("[1] sandbox containment")
    print("=" * 60)
    try:
        validate_path(cfg.agent.sandbox_root, "../escaped.txt")
        print("FAIL: path escape was not caught")
        return 1
    except PathEscapeError as e:
        print(f"  ok: blocked escape → {e}")

    # 2. Registry + write_file inside the sandbox.
    print()
    print("=" * 60)
    print("[2] write_file via registry")
    print("=" * 60)
    registry = build_default_registry(
        sandbox_root=cfg.agent.sandbox_root,
        user_input_provider=stub_user_input_provider,
    )
    write_tool = registry.get("write_file")
    assert write_tool is not None
    result = await write_tool.run(
        WriteFileArgs(path="smoketest.md", content="hello phase 1\n")
    )
    print(f"  ok={result.ok} data={result.data} error={result.error}")
    if not result.ok:
        return 1
    if not (cfg.agent.sandbox_root / "smoketest.md").exists():
        print("FAIL: smoketest.md was not actually written")
        return 1

    # 3. Provider round trip.
    print()
    print("=" * 60)
    print(f"[3] provider stream — model={model} mode={args.mode}")
    print("=" * 60)
    provider = OllamaProvider(
        host=cfg.provider.host,
        keep_alive=cfg.provider.ollama.keep_alive,
    )

    messages = [
        Message(
            role="system",
            content=(
                "You are a helpful assistant with access to file tools. "
                "When the user asks you to write a file, call the write_file "
                "tool. Keep responses short."
            ),
        ),
        Message(
            role="user",
            content="Write the word 'hello' to greeting.md.",
        ),
    ]

    text_buf: list[str] = []
    saw_call = False
    saw_parse_error = False
    try:
        async for chunk in provider.stream(
            model=model,
            messages=messages,
            tools=registry.to_schemas(),
            tool_call_mode=args.mode,
        ):
            if chunk.text:
                text_buf.append(chunk.text)
                print(chunk.text, end="", flush=True)
            if chunk.tool_call is not None:
                saw_call = True
                print(
                    f"\n[tool_call] {chunk.tool_call.name}({chunk.tool_call.args})"
                )
            if chunk.parse_error is not None:
                saw_parse_error = True
                print(f"\n[parse_error] stage={chunk.parse_error.stage} {chunk.parse_error.message}")
            if chunk.done:
                print(
                    "\n[done] "
                    f"prompt={chunk.prompt_tokens} "
                    f"completion={chunk.completion_tokens} "
                    f"ttft_ms={chunk.ttft_ms} "
                    f"total_ms={chunk.total_latency_ms} "
                    f"estimated={chunk.tokens_estimated}"
                )
    except Exception as e:
        print(f"\n[ERROR] provider stream failed: {type(e).__name__}: {e}")
        return 1

    print()
    print("Summary:")
    print(f"  text bytes={sum(len(t) for t in text_buf)}")
    print(f"  saw tool_call={saw_call}")
    print(f"  saw parse_error={saw_parse_error}")
    print()
    print(
        "Note: this script does not dispatch tool calls — Phase 1 only "
        "parses them. Phase 3's loop dispatches."
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
