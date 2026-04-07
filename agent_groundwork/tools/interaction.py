"""Human-in-the-loop tools.

`ask_user` is the bridge that lets the agent pause and request a clarifying
answer from the human. The actual input source is injected at construction
time via `UserInputProvider`, so the same tool implementation works for
the CLI (stdin), a future web UI (HTTP request/response), or a Telegram bot.

Phase 1 ships a stub provider so scripts and tests can construct the
registry without a real frontend.
"""

from __future__ import annotations

import inspect
from typing import Awaitable, Callable, ClassVar, Union

from pydantic import BaseModel, Field

from agent_groundwork.tools.base import ToolResult


UserInputProvider = Callable[[str], Union[str, Awaitable[str]]]


def stub_user_input_provider(question: str) -> str:
    """Phase 1 placeholder. Returns a canned response.

    The CLI (Phase 4) will replace this with a real stdin reader wrapped
    in `asyncio.to_thread`.
    """
    return "(stubbed user response — replace with real CLI provider in Phase 4)"


class AskUserArgs(BaseModel):
    question: str = Field(..., description="The clarifying question to ask the user.")


class AskUserTool:
    name: ClassVar[str] = "ask_user"
    description: ClassVar[str] = (
        "Pause and ask the user a clarifying question. Use this when the "
        "request is ambiguous instead of guessing."
    )
    args_schema = AskUserArgs

    def __init__(self, provider: UserInputProvider) -> None:
        self._provider = provider

    async def run(self, args: AskUserArgs) -> ToolResult:
        out = self._provider(args.question)
        if inspect.isawaitable(out):
            out = await out
        if not isinstance(out, str):
            return ToolResult(
                ok=False,
                error=f"user input provider returned non-str: {type(out).__name__}",
            )
        return ToolResult(ok=True, data=out)
