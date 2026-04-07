"""Sandbox path validation.

The single chokepoint for every file operation in the project. Any tool that
touches the filesystem MUST resolve its path through `validate_path` first.

The function resolves a candidate path against a sandbox root and asserts the
result lies inside that root after symlink resolution. Symlinks pointing
outside the sandbox are rejected; relative paths that would walk out are
rejected; absolute paths outside the root are rejected.
"""

from __future__ import annotations

from pathlib import Path


class PathEscapeError(ValueError):
    """Raised when a candidate path resolves outside the sandbox root."""

    def __init__(self, candidate: str, reason: str) -> None:
        self.candidate = candidate
        self.reason = reason
        super().__init__(f"path escape rejected ({reason}): {candidate!r}")


def validate_path(root: Path, candidate: str) -> Path:
    """Resolve `candidate` relative to `root` and assert containment.

    Returns the resolved absolute path on success. Raises `PathEscapeError` on
    any containment violation. Raises `TypeError` if `candidate` is not a str.

    `root` must exist. `candidate` may name a file that does not yet exist
    (this is necessary for write_file); resolution still follows existing
    symlinks in any prefix of the path, so a symlink-out-of-sandbox in the
    parent chain will correctly fail containment.
    """
    if not isinstance(candidate, str):
        raise TypeError(
            f"validate_path candidate must be str, got {type(candidate).__name__}"
        )
    if "\x00" in candidate:
        raise PathEscapeError(candidate, "contains NUL byte")

    root_resolved = root.resolve(strict=True)
    full = (root_resolved / candidate).resolve(strict=False)

    if not full.is_relative_to(root_resolved):
        raise PathEscapeError(candidate, "resolves outside sandbox root")

    return full
