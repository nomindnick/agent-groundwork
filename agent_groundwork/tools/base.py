"""Tool framework — types, registry, and schema rendering.

A `Tool` is a typed callable the model can invoke. Each tool declares:
  - a unique `name`
  - a one-line `description` (the model reads this)
  - an `args_schema` (a Pydantic model — drives both validation and the
    JSON Schema we hand to the provider)
  - an async `run(args) -> ToolResult` method

Tools NEVER raise. Any exception inside `run` is caught by the dispatcher
(Phase 3) and converted to `ToolResult(ok=False, error=...)`.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel

from agent_groundwork.providers.base import ToolSchema


# --------------------------- result type ---------------------------

class ToolResult(BaseModel):
    """The only allowed return shape from a tool's `run` method.

    `data` should be JSON-serializable in practice (strings, numbers, bools,
    lists/dicts of the same). The trace writer falls back to `default=str`
    as a safety net but normal operation should never hit it.
    """

    ok: bool
    data: Any = None
    error: str | None = None


# --------------------------- tool protocol ---------------------------

@runtime_checkable
class Tool(Protocol):
    name: ClassVar[str]
    description: ClassVar[str]
    args_schema: ClassVar[type[BaseModel]]

    async def run(self, args: BaseModel) -> ToolResult: ...


# --------------------------- schema rendering ---------------------------

def tool_to_schema(tool: Tool) -> ToolSchema:
    """Build a provider-facing `ToolSchema` from a tool.

    `parameters` is the Pydantic-derived JSON Schema dict.
    """
    return ToolSchema(
        name=tool.name,
        description=tool.description,
        parameters=tool.args_schema.model_json_schema(),
    )


def render_prompted_block(schemas: list[ToolSchema]) -> str:
    """Render a markdown block describing the tools, suitable for
    appending to a system prompt in prompted mode.

    The block tells the model exactly how to invoke a tool — emit a fenced
    `\u0060\u0060\u0060json` block containing `{"tool": "<name>", "args": {...}}`.
    The matching parser in `providers/ollama.py` extracts these blocks
    from the streamed output.
    """
    lines: list[str] = []
    lines.append("## Available tools")
    lines.append("")
    lines.append(
        "You may call one tool per response. To call a tool, output a fenced "
        "JSON block in exactly this format:"
    )
    lines.append("")
    lines.append("```json")
    lines.append('{"tool": "<name>", "args": {<arguments>}}')
    lines.append("```")
    lines.append("")
    lines.append(
        "Do not put prose inside the fenced block. After the tool runs you "
        "will receive a tool message with its result and may continue."
    )
    lines.append("")
    lines.append("### Tools")
    lines.append("")
    for schema in schemas:
        lines.append(f"- `{schema.name}` — {schema.description}")
        lines.append("  Arguments (JSON Schema):")
        lines.append("  ```json")
        pretty = json.dumps(schema.parameters, indent=2, sort_keys=True)
        for ln in pretty.splitlines():
            lines.append("  " + ln)
        lines.append("  ```")
    return "\n".join(lines)


def to_native_tool_dict(schema: ToolSchema) -> dict[str, Any]:
    """Render a single `ToolSchema` into Ollama's native tool format."""
    return {
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": schema.parameters,
        },
    }


# --------------------------- registry ---------------------------

class ToolRegistry:
    """Holds the set of tools available to the agent.

    A simple ordered dict wrapper. The registry knows how to render its
    contents into the two formats the provider needs (native or prompted).
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool name already registered: {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def to_schemas(self) -> list[ToolSchema]:
        return [tool_to_schema(t) for t in self._tools.values()]

    def to_native_schemas(self) -> list[dict[str, Any]]:
        return [to_native_tool_dict(s) for s in self.to_schemas()]

    def to_prompted_block(self) -> str:
        return render_prompted_block(self.to_schemas())
