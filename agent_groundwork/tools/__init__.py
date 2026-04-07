"""Tool framework public surface.

Re-exports the framework types and provides a `build_default_registry`
convenience that wires up every v1 tool against a sandbox root and a
user-input provider.
"""

from __future__ import annotations

from pathlib import Path

from agent_groundwork.tools.base import (
    Tool,
    ToolRegistry,
    ToolResult,
    render_prompted_block,
    tool_to_schema,
)
from agent_groundwork.tools.files import (
    EditFileTool,
    ListFilesTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from agent_groundwork.tools.interaction import (
    AskUserTool,
    UserInputProvider,
    stub_user_input_provider,
)


def build_default_registry(
    sandbox_root: Path,
    user_input_provider: UserInputProvider = stub_user_input_provider,
) -> ToolRegistry:
    """Construct a registry populated with all v1 tools."""
    registry = ToolRegistry()
    registry.register(ListFilesTool(sandbox_root))
    registry.register(ReadFileTool(sandbox_root))
    registry.register(WriteFileTool(sandbox_root))
    registry.register(EditFileTool(sandbox_root))
    registry.register(SearchFilesTool(sandbox_root))
    registry.register(AskUserTool(user_input_provider))
    return registry


__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "tool_to_schema",
    "render_prompted_block",
    "build_default_registry",
    "stub_user_input_provider",
    "UserInputProvider",
]
