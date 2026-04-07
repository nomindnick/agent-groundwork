"""Entry point: `python -m agent_groundwork.bakeoff [config.toml]`."""

from __future__ import annotations

import asyncio
import sys

from agent_groundwork.bakeoff.harness import run_bakeoff
from agent_groundwork.config import load_config


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.toml"
    config = load_config(config_path)
    result_dir = asyncio.run(run_bakeoff(config))
    print(f"bakeoff complete: {result_dir}")


if __name__ == "__main__":
    main()
