"""Bakeoff harness — runs (model, mode, scenario) cells against Ollama.

The bakeoff is intentionally one-shot per scenario: we send the user prompt
once and capture whatever the model produces (text, tool calls, parse errors)
plus latency and token counts. Tool calls are NOT executed — the bakeoff
scores them as *attempts* against a per-scenario rubric. The agent loop
(Phase 3) is what actually dispatches tools.

Each cell runs in a fresh temp sandbox. Seed files (declared in the scenario
YAML) are materialized before the run via `validate_path` to keep sandbox
escapes impossible even at setup time.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import ollama
import yaml
from pydantic import BaseModel, Field

from agent_groundwork.config import Config
from agent_groundwork.paths import validate_path
from agent_groundwork.providers.base import (
    ChatChunk,
    Message,
    ParseError,
    ToolCall,
    ToolSchema,
)
from agent_groundwork.providers.ollama import OllamaProvider
from agent_groundwork.tools.base import ToolRegistry
from agent_groundwork.tools.files import (
    EditFileTool,
    ListFilesTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from agent_groundwork.tools.interaction import AskUserTool, stub_user_input_provider
from agent_groundwork.bakeoff.scoring import score_cell


# =============================== scenario types ===============================

class Rubric(BaseModel):
    """Per-scenario scoring expectations. All fields optional."""

    expected_tool_called: str | None = None
    expected_no_tool_calls: bool = False
    expected_to_ask_user: bool = False
    arg_contains: dict[str, Any] = Field(default_factory=dict)
    expect_refusal_text: bool = False


class Scenario(BaseModel):
    """One bakeoff scenario, loaded from a YAML file."""

    name: str
    description: str
    prompt: str
    seed_files: dict[str, str] = Field(default_factory=dict)
    available_tools: list[str] | None = None
    seed_messages: list[Message] = Field(default_factory=list)
    rubric: Rubric


def load_scenarios(scenario_dir: str | Path) -> list[Scenario]:
    """Load all `*.yaml` files in `scenario_dir` into Scenario models.

    Files are sorted by name for stable bakeoff output ordering.
    """
    root = Path(scenario_dir)
    scenarios: list[Scenario] = []
    for path in sorted(root.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        scenarios.append(Scenario.model_validate(raw))
    return scenarios


# =============================== cell result type ===============================

class CellResult(BaseModel):
    """Outcome of a single (model, mode, scenario) bakeoff cell."""

    model: str
    mode: Literal["native", "prompted"]
    scenario: str

    # Timing / token counts (zeros when error is set)
    ttft_ms: float | None = None
    total_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_estimated: bool = False
    tokens_per_sec: float | None = None

    # Scoring axes
    tool_call_attempted: bool = False
    tool_call_parseable: bool = False
    parse_error_stage: str | None = None
    correct_tool: bool = False
    correct_args: bool = False
    stopped_correctly: bool = False

    # Raw artifacts
    response_text: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    parse_errors: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


# =============================== sandbox / registry helpers ===============================

def _seed_sandbox(sandbox_root: Path, files: dict[str, str]) -> None:
    """Materialize the scenario's seed files into a fresh sandbox.

    Routes every path through `validate_path` so a malicious scenario can't
    escape the sandbox via `..` or absolute paths.
    """
    for rel, content in files.items():
        full = validate_path(sandbox_root, rel)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")


def build_tool_registry(sandbox_root: Path) -> ToolRegistry:
    """Build a fresh registry of all six v1 tools bound to a sandbox root.

    `ask_user` uses the stub provider — the bakeoff is one-shot so it should
    never actually be invoked, but its schema must be present for scenarios
    that test the model's ability to call it.
    """
    reg = ToolRegistry()
    reg.register(ListFilesTool(sandbox_root))
    reg.register(ReadFileTool(sandbox_root))
    reg.register(WriteFileTool(sandbox_root))
    reg.register(EditFileTool(sandbox_root))
    reg.register(SearchFilesTool(sandbox_root))
    reg.register(AskUserTool(stub_user_input_provider))
    return reg


def _filtered_schemas(
    registry: ToolRegistry, available: list[str] | None
) -> list[ToolSchema]:
    """Return tool schemas, optionally filtered to a scenario-specific subset."""
    schemas = registry.to_schemas()
    if available is None:
        return schemas
    keep = set(available)
    return [s for s in schemas if s.name in keep]


# =============================== cell runner ===============================

async def run_cell(
    provider: OllamaProvider,
    model: str,
    mode: Literal["native", "prompted"],
    scenario: Scenario,
    sandbox_root: Path,
) -> CellResult:
    """Execute a single bakeoff cell. Never raises — all failure modes are
    captured into `CellResult.error` so the run continues."""
    _seed_sandbox(sandbox_root, scenario.seed_files)
    registry = build_tool_registry(sandbox_root)
    schemas = _filtered_schemas(registry, scenario.available_tools)

    messages: list[Message] = list(scenario.seed_messages) + [
        Message(role="user", content=scenario.prompt)
    ]

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    parse_errors: list[ParseError] = []
    last: ChatChunk | None = None

    try:
        async for chunk in provider.stream(model, messages, schemas, mode):
            if chunk.text:
                text_parts.append(chunk.text)
            if chunk.tool_call is not None:
                tool_calls.append(chunk.tool_call)
            if chunk.parse_error is not None:
                parse_errors.append(chunk.parse_error)
            if chunk.done:
                last = chunk
    except Exception as e:  # noqa: BLE001 — bakeoff captures every failure mode
        return CellResult(
            model=model,
            mode=mode,
            scenario=scenario.name,
            response_text="".join(text_parts),
            tool_calls=[tc.model_dump() for tc in tool_calls],
            parse_errors=[pe.model_dump() for pe in parse_errors],
            error=f"{type(e).__name__}: {e}",
        )

    response_text = "".join(text_parts)
    total_ms = (last.total_latency_ms if last else None) or 0.0
    completion_tokens = (last.completion_tokens if last else 0) or 0
    tps: float | None = None
    if total_ms > 0 and completion_tokens > 0:
        tps = completion_tokens / (total_ms / 1000.0)

    scoring = score_cell(scenario.rubric, response_text, tool_calls, parse_errors)

    return CellResult(
        model=model,
        mode=mode,
        scenario=scenario.name,
        ttft_ms=last.ttft_ms if last else None,
        total_latency_ms=total_ms,
        prompt_tokens=(last.prompt_tokens if last else 0) or 0,
        completion_tokens=completion_tokens,
        tokens_estimated=bool(last.tokens_estimated) if last else False,
        tokens_per_sec=tps,
        response_text=response_text,
        tool_calls=[tc.model_dump() for tc in tool_calls],
        parse_errors=[pe.model_dump() for pe in parse_errors],
        **scoring,
    )


# =============================== cold-load measurement ===============================

async def measure_cold_load(
    host: str,
    model: str,
) -> dict[str, Any]:
    """Force-unload a model and time the next cold call.

    Uses the ollama python client directly (not via OllamaProvider) so we can
    issue an unload (`keep_alive=0`) without polluting the provider abstraction.
    The cold call is a tiny prompt with no tools — we want to measure model
    load time, not generation time.
    """
    client = ollama.AsyncClient(host=host)
    cold_load = {
        "model": model,
        "ttft_ms": None,
        "total_latency_ms": 0.0,
        "error": None,
    }
    try:
        # Step 1: unload by issuing a no-op chat with keep_alive=0.
        try:
            await client.chat(
                model=model,
                messages=[{"role": "user", "content": "ok"}],
                stream=False,
                keep_alive=0,
            )
        except Exception:  # noqa: BLE001 — best effort; even unload failures shouldn't block cold timing
            pass

        # Step 2: time a cold call.
        start = time.monotonic()
        first_at: float | None = None
        stream = await client.chat(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            stream=True,
            keep_alive="30m",
        )
        async for raw in stream:
            content = ""
            msg = raw.get("message") if isinstance(raw, dict) else getattr(raw, "message", None)
            if msg is not None:
                content = (msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")) or ""
            if content and first_at is None:
                first_at = time.monotonic()
        end = time.monotonic()
        cold_load["total_latency_ms"] = (end - start) * 1000.0
        if first_at is not None:
            cold_load["ttft_ms"] = (first_at - start) * 1000.0
    except Exception as e:  # noqa: BLE001
        cold_load["error"] = f"{type(e).__name__}: {e}"
    return cold_load


# =============================== resume helpers ===============================

def _existing_cells(cells_path: Path) -> set[tuple[str, str, str]]:
    """Return the set of (model, mode, scenario) triples already recorded.

    Used to skip cells that have already run when appending to a result dir.
    Errors and partial rows are silently treated as "not done" so they get re-run.
    """
    if not cells_path.exists():
        return set()
    done: set[tuple[str, str, str]] = set()
    with cells_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                done.add((row["model"], row["mode"], row["scenario"]))
            except KeyError:
                continue
    return done


def _existing_cold_loads(cold_path: Path) -> set[str]:
    """Return the set of model names that already have a cold-load row."""
    if not cold_path.exists():
        return set()
    done: set[str] = set()
    with cold_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = row.get("model")
            if isinstance(name, str):
                done.add(name)
    return done


# =============================== top-level runner ===============================

async def run_bakeoff(
    config: Config,
    *,
    models_override: list[str] | None = None,
    result_dir_override: Path | None = None,
) -> Path:
    """Run the bakeoff and write JSONL + report into a result dir.

    By default, creates a fresh timestamped dir under `config.bakeoff.result_dir`
    and runs every model in `config.bakeoff.candidate_models`. Both behaviors
    can be overridden:

    - `models_override`: run only these model names instead of the config list.
      Lets you bake one model at a time without editing config.toml.
    - `result_dir_override`: append to an existing result dir instead of making
      a new one. Cells already present (matched by (model, mode, scenario))
      are skipped, so re-invoking with the same dir is safe and incremental.
      The report regenerates from the union of all cells each time.

    Returns the result directory path.
    """
    scenarios = load_scenarios(config.bakeoff.scenario_dir)
    models = list(models_override) if models_override else list(config.bakeoff.candidate_models)
    if not models:
        raise ValueError("no models to run — pass models_override or set candidate_models")
    modes: list[Literal["native", "prompted"]] = ["native", "prompted"]

    if result_dir_override is not None:
        result_dir = Path(result_dir_override)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        result_dir = Path(config.bakeoff.result_dir) / timestamp
    result_dir.mkdir(parents=True, exist_ok=True)

    cells_path = result_dir / "cells.jsonl"
    cold_path = result_dir / "cold_load.jsonl"

    done_cells = _existing_cells(cells_path)
    done_cold = _existing_cold_loads(cold_path)

    provider = OllamaProvider(
        host=config.provider.host,
        keep_alive=config.provider.ollama.keep_alive,
    )

    planned_cells = [
        (m, mode, s)
        for m in models
        for mode in modes
        for s in scenarios
        if (m, mode, s.name) not in done_cells
    ]
    skipped = (len(models) * len(modes) * len(scenarios)) - len(planned_cells)
    print(
        f"[bakeoff] {len(models)} model(s) x {len(modes)} modes x {len(scenarios)} scenarios "
        f"-> {len(planned_cells)} cells to run "
        f"({skipped} already done) -> {result_dir}",
        file=sys.stderr,
    )

    # Cold-load pass — skip models we've already measured.
    cold_todo = [m for m in models if m not in done_cold]
    if cold_todo:
        print(
            f"[bakeoff] cold-load pass ({len(cold_todo)} model(s); "
            f"{len(models) - len(cold_todo)} already measured)",
            file=sys.stderr,
        )
        with cold_path.open("a", encoding="utf-8") as cold_f:
            for model in cold_todo:
                print(f"[bakeoff]   cold load: {model}", file=sys.stderr)
                cold = await measure_cold_load(config.provider.host, model)
                cold_f.write(json.dumps(cold) + "\n")
                cold_f.flush()
    else:
        print("[bakeoff] cold-load pass: nothing to do", file=sys.stderr)

    # Main pass.
    total_to_run = len(planned_cells)
    with cells_path.open("a", encoding="utf-8") as cells_f:
        for cell_idx, (model, mode, scenario) in enumerate(planned_cells, start=1):
            print(
                f"[bakeoff] cell {cell_idx}/{total_to_run}: "
                f"{model} | {mode} | {scenario.name}",
                file=sys.stderr,
            )
            with TemporaryDirectory(prefix="bakeoff_") as tmp:
                sandbox_root = Path(tmp)
                result = await run_cell(
                    provider, model, mode, scenario, sandbox_root
                )
            cells_f.write(json.dumps(result.model_dump()) + "\n")
            cells_f.flush()

    # Generate the markdown report from the union of all cells in this dir.
    from agent_groundwork.bakeoff.report import generate as generate_report

    report_path = generate_report(result_dir)
    print(f"[bakeoff] report: {report_path}", file=sys.stderr)
    return result_dir


__all__ = [
    "Rubric",
    "Scenario",
    "CellResult",
    "load_scenarios",
    "build_tool_registry",
    "run_cell",
    "measure_cold_load",
    "run_bakeoff",
]
