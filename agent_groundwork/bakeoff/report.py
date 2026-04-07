"""Markdown report generator for a bakeoff run.

Reads `cells.jsonl` + `cold_load.jsonl` from a result directory and writes
`report.md` next to them. Pure stdlib string assembly — no jinja, no extras.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _fmt_ms(value: float | int | None) -> str:
    if value is None:
        return "—"
    return f"{value:.0f}"


def _fmt_float(value: float | int | None, places: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:.{places}f}"


def _bool_mark(value: bool) -> str:
    return "yes" if value else "no"


def _is_success(cell: dict[str, Any]) -> bool:
    """Cell-level success: tool choice + args + stop behavior all correct
    and no parse errors. (parse_error_stage being set means the model emitted
    something unparseable, which is a failure regardless of correct_tool.)"""
    if cell.get("error"):
        return False
    if cell.get("parse_error_stage"):
        return False
    return bool(
        cell.get("correct_tool")
        and cell.get("correct_args")
        and cell.get("stopped_correctly")
    )


def _avg(values: list[float | int | None]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


# =============================== sections ===============================

def _header(
    cells: list[dict[str, Any]], cold: list[dict[str, Any]], result_dir: Path
) -> str:
    models = sorted({c["model"] for c in cells})
    scenarios = sorted({c["scenario"] for c in cells})
    lines = [
        f"# Bakeoff report — {result_dir.name}",
        "",
        f"- Result dir: `{result_dir}`",
        f"- Models tested ({len(models)}): " + ", ".join(f"`{m}`" for m in models),
        f"- Scenarios ({len(scenarios)}): " + ", ".join(f"`{s}`" for s in scenarios),
        f"- Total cells: {len(cells)}",
        f"- Cold-load measurements: {len(cold)}",
        "",
    ]
    return "\n".join(lines)


def _per_model_summary(
    cells: list[dict[str, Any]], cold: list[dict[str, Any]]
) -> str:
    cold_by_model = {c["model"]: c for c in cold}

    # Group by (model, mode).
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for c in cells:
        groups.setdefault((c["model"], c["mode"]), []).append(c)

    rows: list[str] = []
    rows.append("## Per-model summary")
    rows.append("")
    rows.append(
        "| model | mode | success | avg ttft (ms) | avg total (ms) | tok/s | parse fail | provider err | cold load (ms) |"
    )
    rows.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    for (model, mode), cs in sorted(groups.items()):
        n = len(cs)
        successes = sum(1 for c in cs if _is_success(c))
        success_rate = f"{successes}/{n}"
        avg_ttft = _avg([c.get("ttft_ms") for c in cs])
        avg_total = _avg([c.get("total_latency_ms") for c in cs])
        avg_tps = _avg([c.get("tokens_per_sec") for c in cs])
        parse_fail = sum(1 for c in cs if c.get("parse_error_stage"))
        prov_err = sum(1 for c in cs if c.get("error"))
        cold_ms: float | None = None
        cm = cold_by_model.get(model)
        if cm and cm.get("error") is None:
            cold_ms = cm.get("total_latency_ms")
        rows.append(
            f"| `{model}` | {mode} | {success_rate} | {_fmt_ms(avg_ttft)} | "
            f"{_fmt_ms(avg_total)} | {_fmt_float(avg_tps)} | {parse_fail} | {prov_err} | {_fmt_ms(cold_ms)} |"
        )
    rows.append("")
    return "\n".join(rows)


def _per_scenario_breakdown(cells: list[dict[str, Any]]) -> str:
    scenarios = sorted({c["scenario"] for c in cells})
    out: list[str] = ["## Per-scenario breakdown", ""]
    for scenario in scenarios:
        rows = [c for c in cells if c["scenario"] == scenario]
        out.append(f"### `{scenario}`")
        out.append("")
        out.append(
            "| model | mode | tool? | parse? | correct tool | correct args | stopped | total (ms) |"
        )
        out.append("|---|---|:---:|:---:|:---:|:---:|:---:|---:|")
        for c in sorted(rows, key=lambda r: (r["model"], r["mode"])):
            err_marker = " ⚠" if c.get("error") else ""
            out.append(
                f"| `{c['model']}`{err_marker} | {c['mode']} | "
                f"{_bool_mark(c.get('tool_call_attempted', False))} | "
                f"{_bool_mark(c.get('tool_call_parseable', False))} | "
                f"{_bool_mark(c.get('correct_tool', False))} | "
                f"{_bool_mark(c.get('correct_args', False))} | "
                f"{_bool_mark(c.get('stopped_correctly', False))} | "
                f"{_fmt_ms(c.get('total_latency_ms'))} |"
            )
        out.append("")
    return "\n".join(out)


def _cold_load_section(cold: list[dict[str, Any]]) -> str:
    if not cold:
        return ""
    out = ["## Cold-load timings", ""]
    out.append("| model | ttft (ms) | total (ms) | error |")
    out.append("|---|---:|---:|---|")
    for c in sorted(cold, key=lambda r: r["model"]):
        out.append(
            f"| `{c['model']}` | {_fmt_ms(c.get('ttft_ms'))} | "
            f"{_fmt_ms(c.get('total_latency_ms'))} | {c.get('error') or ''} |"
        )
    out.append("")
    return "\n".join(out)


def _notable_failures(cells: list[dict[str, Any]]) -> str:
    interesting = [
        c
        for c in cells
        if c.get("error")
        or c.get("parse_error_stage")
        or not c.get("correct_tool")
        or not c.get("correct_args")
    ]
    if not interesting:
        return "## Notable failures\n\n_None — every cell passed cleanly._\n"
    out = ["## Notable failures", ""]
    out.append(
        f"_{len(interesting)} cell(s) had a parse error, provider error, "
        f"or scoring miss. Raw outputs below for hand review._"
    )
    out.append("")
    for c in interesting:
        out.append(
            f"### `{c['model']}` / {c['mode']} / `{c['scenario']}`"
        )
        out.append("")
        if c.get("error"):
            out.append(f"- **provider error:** `{c['error']}`")
        if c.get("parse_error_stage"):
            out.append(f"- **parse error stage:** `{c['parse_error_stage']}`")
        out.append(
            f"- correct_tool: {_bool_mark(c.get('correct_tool', False))}, "
            f"correct_args: {_bool_mark(c.get('correct_args', False))}, "
            f"stopped_correctly: {_bool_mark(c.get('stopped_correctly', False))}"
        )
        if c.get("response_text"):
            # Use a 4-backtick fence so any inner ```json blocks (common in
            # prompted-mode failures) don't terminate the outer block.
            out.append("")
            out.append("**response text:**")
            out.append("")
            out.append("````")
            out.append(c["response_text"])
            out.append("````")
        if c.get("tool_calls"):
            out.append("")
            out.append("**tool calls:**")
            out.append("")
            out.append("```json")
            out.append(json.dumps(c["tool_calls"], indent=2))
            out.append("```")
        if c.get("parse_errors"):
            out.append("")
            out.append("**parse errors:**")
            out.append("")
            out.append("```json")
            out.append(json.dumps(c["parse_errors"], indent=2))
            out.append("```")
        out.append("")
    return "\n".join(out)


# =============================== entry point ===============================

def generate(result_dir: str | Path) -> Path:
    """Read JSONL inputs from `result_dir` and write `report.md`.

    Returns the path to the generated report.
    """
    result_dir = Path(result_dir)
    cells = _load_jsonl(result_dir / "cells.jsonl")
    cold = _load_jsonl(result_dir / "cold_load.jsonl")

    sections = [
        _header(cells, cold, result_dir),
        _per_model_summary(cells, cold),
        _per_scenario_breakdown(cells),
        _cold_load_section(cold),
        _notable_failures(cells),
    ]
    report = "\n".join(s for s in sections if s)

    out_path = result_dir / "report.md"
    out_path.write_text(report, encoding="utf-8")
    return out_path


__all__ = ["generate"]
