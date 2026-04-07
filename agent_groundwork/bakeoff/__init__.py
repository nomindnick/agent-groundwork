"""Bakeoff harness package — model evaluation across scenarios.

See `harness.py` for the runner and `report.py` for the markdown generator.
Scenarios live as YAML files in `scenarios/`.
"""

from agent_groundwork.bakeoff.harness import (
    CellResult,
    Rubric,
    Scenario,
    build_tool_registry,
    load_scenarios,
    measure_cold_load,
    run_bakeoff,
    run_cell,
)
from agent_groundwork.bakeoff.report import generate as generate_report
from agent_groundwork.bakeoff.scoring import score_cell

__all__ = [
    "CellResult",
    "Rubric",
    "Scenario",
    "build_tool_registry",
    "generate_report",
    "load_scenarios",
    "measure_cold_load",
    "run_bakeoff",
    "run_cell",
    "score_cell",
]
