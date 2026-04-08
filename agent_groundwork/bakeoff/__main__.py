"""Entry point: `python -m agent_groundwork.bakeoff [options]`.

By default, runs the full candidate list from `config.toml` into a fresh
timestamped directory under `bakeoff.result_dir`.

Common one-model-at-a-time workflow:

    # First model, fresh result dir
    python -m agent_groundwork.bakeoff --model gemma4:e4b

    # Note the result dir from the run above, then append more models
    python -m agent_groundwork.bakeoff \
        --model qwen3:8b \
        --result-dir bakeoff_results/20260407-180000

Cells already present in the target dir are skipped, so re-running with the
same `--result-dir` is safe and incremental. The report regenerates from the
union of all cells each time.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from agent_groundwork.bakeoff.harness import run_bakeoff
from agent_groundwork.config import load_config


def _parse_models(values: list[str] | None) -> list[str] | None:
    """Flatten a repeatable comma-tolerant --model arg into a clean list."""
    if not values:
        return None
    out: list[str] = []
    for v in values:
        for piece in v.split(","):
            piece = piece.strip()
            if piece and piece not in out:
                out.append(piece)
    return out or None


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m agent_groundwork.bakeoff",
        description="Run the bakeoff harness against one or more Ollama models.",
    )
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Path to config.toml (default: ./config.toml)",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Model name to run. Repeatable; values may also be comma-separated. "
            "Overrides bakeoff.candidate_models from config.toml."
        ),
    )
    parser.add_argument(
        "--result-dir",
        default=None,
        help=(
            "Append to an existing result directory instead of creating a fresh "
            "timestamped one. Cells already present are skipped; the report "
            "regenerates from the union of all cells."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    models_override = _parse_models(args.model)
    result_dir_override = Path(args.result_dir) if args.result_dir else None

    result_dir = asyncio.run(
        run_bakeoff(
            config,
            models_override=models_override,
            result_dir_override=result_dir_override,
        )
    )
    print(f"bakeoff complete: {result_dir}")


if __name__ == "__main__":
    main()
