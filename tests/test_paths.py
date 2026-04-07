"""Tests for the sandbox path guard.

This is the one place in v1 where unit tests are non-optional. A sandbox
escape would be a real bug. Coverage targets every kind of escape attempt
plus the happy paths the file tools rely on.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_groundwork.paths import PathEscapeError, validate_path


# --------------------------- happy paths ---------------------------

def test_empty_string_returns_root(tmp_path: Path) -> None:
    assert validate_path(tmp_path, "") == tmp_path.resolve()


def test_dot_returns_root(tmp_path: Path) -> None:
    assert validate_path(tmp_path, ".") == tmp_path.resolve()


def test_simple_filename(tmp_path: Path) -> None:
    assert validate_path(tmp_path, "foo.md") == (tmp_path / "foo.md").resolve()


def test_nested_path(tmp_path: Path) -> None:
    assert (
        validate_path(tmp_path, "a/b/c.md")
        == (tmp_path / "a" / "b" / "c.md").resolve()
    )


def test_dotdot_walking_back_inside(tmp_path: Path) -> None:
    # Resolves to root/b.md — still inside root, so allowed.
    assert validate_path(tmp_path, "a/../b.md") == (tmp_path / "b.md").resolve()


def test_leading_dot_slash(tmp_path: Path) -> None:
    assert validate_path(tmp_path, "./foo") == (tmp_path / "foo").resolve()


def test_existing_file_resolves(tmp_path: Path) -> None:
    f = tmp_path / "exists.txt"
    f.write_text("hi")
    assert validate_path(tmp_path, "exists.txt") == f.resolve()


def test_nonexistent_file_still_validates(tmp_path: Path) -> None:
    """write_file needs this — the target doesn't exist yet."""
    result = validate_path(tmp_path, "new/sub/file.md")
    assert result == (tmp_path / "new" / "sub" / "file.md").resolve()


# --------------------------- absolute escape ---------------------------

def test_absolute_outside_root_rejected(tmp_path: Path) -> None:
    with pytest.raises(PathEscapeError):
        validate_path(tmp_path, "/etc/passwd")


def test_absolute_sibling_directory_rejected(tmp_path: Path) -> None:
    sibling = tmp_path.parent / "evil"
    with pytest.raises(PathEscapeError):
        validate_path(tmp_path, str(sibling))


# --------------------------- dot-dot escape ---------------------------

def test_dotdot_escape_rejected(tmp_path: Path) -> None:
    with pytest.raises(PathEscapeError):
        validate_path(tmp_path, "../evil")


def test_double_dotdot_escape_rejected(tmp_path: Path) -> None:
    with pytest.raises(PathEscapeError):
        validate_path(tmp_path, "a/../../evil")


def test_many_dotdots_rejected(tmp_path: Path) -> None:
    with pytest.raises(PathEscapeError):
        validate_path(tmp_path, "../" * 20 + "etc/passwd")


# --------------------------- symlinks ---------------------------

def test_symlink_inside_pointing_outside_rejected(tmp_path: Path) -> None:
    outside_target = tmp_path.parent / "external_target"
    outside_target.mkdir(exist_ok=True)
    link = tmp_path / "evil_link"
    os.symlink(outside_target, link)
    with pytest.raises(PathEscapeError):
        validate_path(tmp_path, "evil_link")


def test_symlink_inside_pointing_outside_via_subpath_rejected(tmp_path: Path) -> None:
    outside_target = tmp_path.parent / "external_target_2"
    outside_target.mkdir(exist_ok=True)
    (outside_target / "secrets.txt").write_text("hunter2")
    link = tmp_path / "evil_link"
    os.symlink(outside_target, link)
    with pytest.raises(PathEscapeError):
        validate_path(tmp_path, "evil_link/secrets.txt")


def test_symlink_inside_pointing_inside_allowed(tmp_path: Path) -> None:
    real = tmp_path / "real.md"
    real.write_text("ok")
    link = tmp_path / "alias.md"
    os.symlink(real, link)
    # Resolves to real.md, which is inside root.
    assert validate_path(tmp_path, "alias.md") == real.resolve()


# --------------------------- bad input ---------------------------

def test_nul_byte_rejected(tmp_path: Path) -> None:
    with pytest.raises(PathEscapeError):
        validate_path(tmp_path, "foo\x00bar")


def test_bytes_rejected(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        validate_path(tmp_path, b"foo")  # type: ignore[arg-type]


def test_none_rejected(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        validate_path(tmp_path, None)  # type: ignore[arg-type]
