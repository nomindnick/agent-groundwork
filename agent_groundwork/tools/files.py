"""File-system tools the agent uses to interact with its sandbox.

Five tools:
  - list_files   : list (recursively) what's in the sandbox
  - read_file    : read a UTF-8 text file
  - write_file   : create or overwrite a file (atomic, with parent mkdir)
  - edit_file    : literal-string targeted edit, with ambiguity detection
  - search_files : ripgrep wrapper with a Python fallback

All paths route through `agent_groundwork.paths.validate_path`. Errors are
returned as `ToolResult(ok=False, error=...)`, never raised. Error messages
report relativized paths so the model never sees the host filesystem layout.
"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from agent_groundwork.paths import PathEscapeError, validate_path
from agent_groundwork.tools.base import ToolResult


# --------------------------- shared helpers ---------------------------

MAX_READ_BYTES = 256 * 1024
MAX_LIST_ENTRIES = 1000
MAX_SEARCH_MATCHES = 200
RG_TIMEOUT_SECONDS = 10
_SKIP_DIR_NAMES = {"__pycache__", ".git"}


def _rel(sandbox_root: Path, abs_path: Path) -> str:
    """Return a POSIX relative path from sandbox root to abs_path.

    Both paths are resolved to handle the case where sandbox_root contains
    symlinks (uncommon, but possible during tests).
    """
    return abs_path.resolve().relative_to(sandbox_root.resolve()).as_posix()


def _atomic_write(path: Path, content: str) -> int:
    """Write `content` atomically to `path`. Returns bytes written."""
    encoded = content.encode("utf-8")
    suffix = ".tmp." + uuid.uuid4().hex[:6]
    tmp = path.with_name(path.name + suffix)
    tmp.write_bytes(encoded)
    tmp.replace(path)
    return len(encoded)


def _is_hidden_or_skip(p: Path) -> bool:
    """Skip dotfiles and well-known noise dirs at any depth."""
    parts = p.parts
    return any(part.startswith(".") or part in _SKIP_DIR_NAMES for part in parts)


# =============================== list_files ===============================

class ListFilesArgs(BaseModel):
    subdir: str = Field(
        default="",
        description="Subdirectory under the sandbox to list. Empty = whole sandbox.",
    )


class ListFilesTool:
    name: ClassVar[str] = "list_files"
    description: ClassVar[str] = (
        "List files in the sandbox, optionally under a subdirectory. "
        "Returns relative POSIX paths."
    )
    args_schema = ListFilesArgs

    def __init__(self, sandbox_root: Path) -> None:
        self._root = sandbox_root

    async def run(self, args: ListFilesArgs) -> ToolResult:
        try:
            full = validate_path(self._root, args.subdir)
        except PathEscapeError as e:
            return ToolResult(ok=False, error=str(e))

        if not full.exists():
            return ToolResult(
                ok=False,
                error=f"path does not exist: {args.subdir or '.'}",
            )
        if not full.is_dir():
            return ToolResult(
                ok=False,
                error=f"not a directory: {args.subdir or '.'}",
            )

        paths: list[str] = []
        truncated = False
        for entry in sorted(full.rglob("*")):
            if entry.is_dir():
                continue
            rel_to_root = entry.relative_to(self._root.resolve())
            if _is_hidden_or_skip(rel_to_root):
                continue
            paths.append(rel_to_root.as_posix())
            if len(paths) >= MAX_LIST_ENTRIES:
                truncated = True
                break

        return ToolResult(
            ok=True,
            data={"paths": paths, "count": len(paths), "truncated": truncated},
        )


# =============================== read_file ===============================

class ReadFileArgs(BaseModel):
    path: str = Field(..., description="Path to the file inside the sandbox.")


class ReadFileTool:
    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = (
        "Read a text file from the sandbox and return its contents."
    )
    args_schema = ReadFileArgs

    def __init__(self, sandbox_root: Path) -> None:
        self._root = sandbox_root

    async def run(self, args: ReadFileArgs) -> ToolResult:
        try:
            full = validate_path(self._root, args.path)
        except PathEscapeError as e:
            return ToolResult(ok=False, error=str(e))

        if not full.exists():
            return ToolResult(ok=False, error=f"file not found: {args.path}")
        if full.is_dir():
            return ToolResult(ok=False, error=f"is a directory, not a file: {args.path}")

        size = full.stat().st_size
        truncated = False
        if size > MAX_READ_BYTES:
            data = full.read_bytes()[:MAX_READ_BYTES]
            text = data.decode("utf-8", errors="replace")
            truncated = True
        else:
            text = full.read_text(encoding="utf-8", errors="replace")

        return ToolResult(
            ok=True,
            data={
                "path": _rel(self._root, full),
                "content": text,
                "bytes": size,
                "truncated": truncated,
            },
        )


# =============================== write_file ===============================

class WriteFileArgs(BaseModel):
    path: str = Field(..., description="Path inside the sandbox. Parent dirs are created if missing.")
    content: str = Field(..., description="Full file contents.")


class WriteFileTool:
    name: ClassVar[str] = "write_file"
    description: ClassVar[str] = (
        "Create or overwrite a text file in the sandbox. Parent directories "
        "are created automatically. Use edit_file for small targeted changes."
    )
    args_schema = WriteFileArgs

    def __init__(self, sandbox_root: Path) -> None:
        self._root = sandbox_root

    async def run(self, args: WriteFileArgs) -> ToolResult:
        try:
            full = validate_path(self._root, args.path)
        except PathEscapeError as e:
            return ToolResult(ok=False, error=str(e))

        if full.exists() and full.is_dir():
            return ToolResult(
                ok=False,
                error=f"path is a directory, refusing to overwrite: {args.path}",
            )

        overwrote = full.exists()
        created_dirs = not full.parent.exists()
        full.parent.mkdir(parents=True, exist_ok=True)

        try:
            bytes_written = _atomic_write(full, args.content)
        except OSError as e:
            return ToolResult(ok=False, error=f"write failed: {e}")

        return ToolResult(
            ok=True,
            data={
                "path": _rel(self._root, full),
                "bytes_written": bytes_written,
                "overwrote": overwrote,
                "created_dirs": created_dirs,
            },
        )


# =============================== edit_file ===============================

class EditFileArgs(BaseModel):
    path: str = Field(..., description="Path inside the sandbox.")
    old: str = Field(..., description="Literal string to find. Must match exactly once.")
    new: str = Field(..., description="String to replace `old` with.")


class EditFileTool:
    name: ClassVar[str] = "edit_file"
    description: ClassVar[str] = (
        "Make a targeted edit to a file using literal string replacement. "
        "Errors if `old` is not found or matches more than once."
    )
    args_schema = EditFileArgs

    def __init__(self, sandbox_root: Path) -> None:
        self._root = sandbox_root

    async def run(self, args: EditFileArgs) -> ToolResult:
        try:
            full = validate_path(self._root, args.path)
        except PathEscapeError as e:
            return ToolResult(ok=False, error=str(e))

        if not full.exists():
            return ToolResult(ok=False, error=f"file not found: {args.path}")
        if full.is_dir():
            return ToolResult(ok=False, error=f"is a directory, not a file: {args.path}")
        if args.old == "":
            return ToolResult(ok=False, error="empty 'old' string is not allowed")

        try:
            text = full.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(ok=False, error=f"file is not valid UTF-8: {args.path}")

        rel = _rel(self._root, full)
        count = text.count(args.old)
        if count == 0:
            return ToolResult(ok=False, error=f"'old' not found in {rel}")
        if count > 1:
            return ToolResult(
                ok=False,
                error=f"'old' matched {count} times in {rel} — narrow the match",
            )

        new_text = text.replace(args.old, args.new, 1)
        try:
            _atomic_write(full, new_text)
        except OSError as e:
            return ToolResult(ok=False, error=f"write failed: {e}")

        return ToolResult(
            ok=True,
            data={"path": rel, "replacements": 1},
        )


# =============================== search_files ===============================

class SearchFilesArgs(BaseModel):
    query: str = Field(..., description="Literal string or regex to search for.")
    subdir: str = Field(default="", description="Subdirectory under the sandbox to scope the search.")


class SearchFilesTool:
    name: ClassVar[str] = "search_files"
    description: ClassVar[str] = (
        "Search file contents in the sandbox and return matching lines as "
        "{path, line, text} entries. Uses ripgrep when available."
    )
    args_schema = SearchFilesArgs

    def __init__(self, sandbox_root: Path) -> None:
        self._root = sandbox_root

    async def run(self, args: SearchFilesArgs) -> ToolResult:
        if not args.query:
            return ToolResult(ok=False, error="empty query")

        try:
            full = validate_path(self._root, args.subdir)
        except PathEscapeError as e:
            return ToolResult(ok=False, error=str(e))

        if not full.exists():
            return ToolResult(ok=False, error=f"path does not exist: {args.subdir or '.'}")
        if not full.is_dir():
            return ToolResult(ok=False, error=f"not a directory: {args.subdir or '.'}")

        try:
            matches, truncated = self._search_with_ripgrep(args.query, full)
        except FileNotFoundError:
            matches, truncated = self._search_python(args.query, full)

        return ToolResult(
            ok=True,
            data={"matches": matches, "truncated": truncated},
        )

    def _search_with_ripgrep(
        self, query: str, full: Path
    ) -> tuple[list[dict[str, object]], bool]:
        # --no-ignore: the sandbox is the agent's data dir, not source code,
        # so .gitignore should not hide files from the agent's own search.
        result = subprocess.run(
            [
                "rg",
                "--vimgrep",
                "--no-heading",
                "--no-ignore",
                "--",
                query,
                str(full),
            ],
            capture_output=True,
            text=True,
            timeout=RG_TIMEOUT_SECONDS,
            check=False,
        )
        # rg exits 1 on no matches; treat that as empty result.
        if result.returncode not in (0, 1):
            # Bubble up as a non-fatal Python fallback trigger.
            raise FileNotFoundError(
                f"ripgrep returned {result.returncode}: {result.stderr.strip()}"
            )

        matches: list[dict[str, object]] = []
        truncated = False
        for line in result.stdout.splitlines():
            # vimgrep format: "path:line:col:text"
            parts = line.split(":", 3)
            if len(parts) < 4:
                continue
            abs_path_str, line_no_str, _col, text = parts
            try:
                line_no = int(line_no_str)
            except ValueError:
                continue
            try:
                rel = _rel(self._root, Path(abs_path_str))
            except (ValueError, OSError):
                continue
            matches.append({"path": rel, "line": line_no, "text": text})
            if len(matches) >= MAX_SEARCH_MATCHES:
                truncated = True
                break
        return matches, truncated

    def _search_python(
        self, query: str, full: Path
    ) -> tuple[list[dict[str, object]], bool]:
        matches: list[dict[str, object]] = []
        truncated = False
        for entry in sorted(full.rglob("*")):
            if entry.is_dir():
                continue
            rel_to_root = entry.relative_to(self._root.resolve())
            if _is_hidden_or_skip(rel_to_root):
                continue
            try:
                text = entry.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if query in line:
                    matches.append(
                        {
                            "path": rel_to_root.as_posix(),
                            "line": line_no,
                            "text": line,
                        }
                    )
                    if len(matches) >= MAX_SEARCH_MATCHES:
                        truncated = True
                        return matches, truncated
        return matches, truncated
