"""Per-cell rubric scoring.

Deliberately small and dumb. The rubrics in the scenario YAMLs are intentionally
lenient — small models phrase things many ways, and the goal of the bakeoff is
to surface raw outputs for manual review, not to gatekeep on rigid pattern
matching. Tweak this file freely after the first run.
"""

from __future__ import annotations

from typing import Any

from agent_groundwork.providers.base import ParseError, ToolCall


REFUSAL_KEYWORDS = (
    "can't",
    "cannot",
    "unable",
    "no such tool",
    "don't have",
    "do not have",
    "not allowed",
    "not available",
    "no tool",
    "refuse",
)


def _arg_value_matches(actual: Any, expected: Any) -> bool:
    """Equality check, but None on the actual side never matches."""
    return actual == expected


def _arg_contains_matches(actual: Any, expected_substr: Any) -> bool:
    """Substring check on stringified actual value."""
    if actual is None:
        return False
    return str(expected_substr) in str(actual)


def _check_arg_contains(
    rubric_args: dict[str, Any], call_args: dict[str, Any]
) -> bool:
    """Validate `arg_contains` against a tool call's args.

    A key suffixed with `_contains` triggers a substring check on the field
    obtained by stripping that suffix; bare keys are equality checks.
    Returns True iff every requirement is satisfied.
    """
    for key, expected in rubric_args.items():
        if key.endswith("_contains"):
            field = key[: -len("_contains")]
            if not _arg_contains_matches(call_args.get(field), expected):
                return False
        else:
            if not _arg_value_matches(call_args.get(key), expected):
                return False
    return True


def _looks_like_refusal(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in REFUSAL_KEYWORDS)


def score_cell(
    rubric: Any,  # bakeoff.harness.Rubric — typed as Any to avoid circular import
    text: str,
    tool_calls: list[ToolCall],
    parse_errors: list[ParseError],
) -> dict[str, Any]:
    """Compute the scoring fields for a CellResult.

    Returns a dict with: tool_call_attempted, tool_call_parseable,
    parse_error_stage, correct_tool, correct_args, stopped_correctly.
    """
    attempted = bool(tool_calls) or bool(parse_errors)
    parseable = bool(tool_calls) and not parse_errors
    parse_error_stage = parse_errors[0].stage if parse_errors else None

    first_call: ToolCall | None = tool_calls[0] if tool_calls else None

    # --- correct_tool ---
    if rubric.expect_refusal_text:
        # Repurpose: "correct" means refused in text and didn't call any tool.
        correct_tool = _looks_like_refusal(text) and not tool_calls
    elif rubric.expected_to_ask_user:
        correct_tool = first_call is not None and first_call.name == "ask_user"
    elif rubric.expected_no_tool_calls:
        correct_tool = not tool_calls
    elif rubric.expected_tool_called:
        correct_tool = (
            first_call is not None and first_call.name == rubric.expected_tool_called
        )
    else:
        correct_tool = True  # nothing asserted -> trivially satisfied

    # --- correct_args ---
    if rubric.arg_contains and first_call is not None:
        correct_args = _check_arg_contains(rubric.arg_contains, first_call.args)
    elif rubric.arg_contains and first_call is None:
        correct_args = False
    else:
        correct_args = True  # nothing asserted

    # --- stopped_correctly (one-shot proxy) ---
    if rubric.expected_no_tool_calls or rubric.expect_refusal_text:
        stopped_correctly = len(tool_calls) == 0
    elif rubric.expected_to_ask_user:
        stopped_correctly = (
            len(tool_calls) == 1 and tool_calls[0].name == "ask_user"
        )
    else:
        # Default: most one-shot scenarios want exactly one tool call.
        stopped_correctly = len(tool_calls) == 1

    return {
        "tool_call_attempted": attempted,
        "tool_call_parseable": parseable,
        "parse_error_stage": parse_error_stage,
        "correct_tool": correct_tool,
        "correct_args": correct_args,
        "stopped_correctly": stopped_correctly,
    }


__all__ = ["score_cell"]
