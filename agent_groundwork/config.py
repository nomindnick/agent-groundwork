"""Configuration loading.

Reads `config.toml` (TOML, via stdlib `tomllib`) into a Pydantic model. Path
fields in the config are resolved relative to the config file's directory, so
`./sandbox` works regardless of the current working directory.

The loader does NOT create any directories. Tracer/compactor (Phase 3) create
their own output dirs on first use.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    sandbox_root: Path
    system_prompt_path: Path
    max_iterations: int = 8


class OllamaProviderConfig(BaseModel):
    keep_alive: str = "30m"


class ProviderConfig(BaseModel):
    backend: Literal["ollama"] = "ollama"
    host: str = "http://localhost:11434"
    ollama: OllamaProviderConfig = Field(default_factory=OllamaProviderConfig)


class ModelConfig(BaseModel):
    name: str
    tool_call_mode: Literal["native", "prompted"] = "native"


class CompactionConfig(BaseModel):
    trigger_messages: int = 10
    trigger_tokens: int = 4000
    recent_window: int = 6
    summarization_model: str = ""  # empty = use main model
    summary_dir: Path


class TracingConfig(BaseModel):
    trace_dir: Path


class BakeoffConfig(BaseModel):
    result_dir: Path
    scenario_dir: Path
    candidate_models: list[str]


class Config(BaseModel):
    agent: AgentConfig
    provider: ProviderConfig
    model: ModelConfig
    compaction: CompactionConfig
    tracing: TracingConfig
    bakeoff: BakeoffConfig


# Whitelist of (section, field) pairs whose string values should be resolved
# as paths relative to the config file's directory at load time.
_PATH_FIELDS: tuple[tuple[str, str], ...] = (
    ("agent", "sandbox_root"),
    ("agent", "system_prompt_path"),
    ("compaction", "summary_dir"),
    ("tracing", "trace_dir"),
    ("bakeoff", "result_dir"),
    ("bakeoff", "scenario_dir"),
)


def _resolve_path_fields(raw: dict[str, Any], base: Path) -> dict[str, Any]:
    """Resolve known path fields against `base` (the config file's dir).

    `~` is expanded. Absolute paths are left as-is. Relative paths are
    rooted at `base`. Returns a new dict; does not mutate input.
    """
    out: dict[str, Any] = {k: dict(v) if isinstance(v, dict) else v for k, v in raw.items()}
    for section, field in _PATH_FIELDS:
        if section not in out or not isinstance(out[section], dict):
            continue
        value = out[section].get(field)
        if not isinstance(value, str):
            continue
        expanded = Path(value).expanduser()
        if not expanded.is_absolute():
            expanded = (base / expanded).resolve()
        out[section][field] = expanded
    return out


def load_config(path: str | Path = "config.toml") -> Config:
    """Load and validate config from a TOML file.

    Path fields in the config are resolved relative to the config file's
    directory, so values like `./sandbox` work regardless of cwd.
    """
    config_path = Path(path).resolve()
    with config_path.open("rb") as f:
        raw = tomllib.load(f)
    resolved = _resolve_path_fields(raw, config_path.parent)
    return Config.model_validate(resolved)
