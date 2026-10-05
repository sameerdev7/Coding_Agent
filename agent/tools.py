import subprocess
from pathlib import Path

from agent.sandbox import LocalSandbox, Sandbox

ALLOWED_COMMANDS = {"pytest", "python", "python3"}
MAX_COMMAND_TIMEOUT = 120


def _safe_path(repo_root: Path, user_path: str) -> Path:
    resolved = (repo_root / user_path).resolve()
    repo_root_resolved = repo_root.resolve()
    if not resolved.is_relative_to(repo_root_resolved):
        raise ValueError(f"path escapes repo root: {user_path}")
    return resolved


def read_file(repo_root: Path, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    """Read a file, or a 1-indexed inclusive line range of it."""
    target = _safe_path(repo_root, path)
    if not target.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    text = target.read_text()
    if start_line is None and end_line is None:
        return text
    lines = text.splitlines(keepends=True)
    start = max(start_line or 1, 1)
    end = end_line if end_line is not None else len(lines)
    if start > end:
        raise ValueError(f"start_line ({start}) is after end_line ({end})")
    return "".join(lines[start - 1 : end])


def write_file(repo_root: Path, path: str, content: str) -> str:
    target = _safe_path(repo_root, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"wrote {len(content.splitlines())} lines to {path}"


def edit_file(repo_root: Path, path: str, old_text: str, new_text: str) -> str:
    """Replace exactly one occurrence of `old_text` — cheaper and safer than rewriting a whole file."""
    target = _safe_path(repo_root, path)
    if not target.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    if not old_text:
        raise ValueError("old_text must not be empty")
    content = target.read_text()
    occurrences = content.count(old_text)
    if occurrences == 0:
        raise ValueError("old_text was not found in the file; re-read it and copy the text exactly")
    if occurrences > 1:
        raise ValueError(f"old_text appears {occurrences} times; include more surrounding lines to make it unique")
    target.write_text(content.replace(old_text, new_text, 1))
    return f"edited {path}: replaced 1 occurrence"


def list_dir(repo_root: Path, path: str = ".") -> str:
    target = _safe_path(repo_root, path)
    if not target.is_dir():
        raise NotADirectoryError(f"not a directory: {path}")
    entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
    return "\n".join(entries)


def grep(repo_root: Path, pattern: str, path: str = ".") -> str:
    target = _safe_path(repo_root, path)
    # -e + -- so a pattern or path starting with "-" can never be parsed as a grep option.
    result = subprocess.run(
        ["grep", "-rnI", "-e", pattern, "--", str(target)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        errors="replace",
        shell=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"grep failed: {result.stderr}")
    return result.stdout or "(no matches)"


def run_command(repo_root: Path, args: list[str], timeout: int = 30, sandbox: Sandbox | None = None) -> str:
    if not args:
        raise ValueError("empty command")
    if args[0] not in ALLOWED_COMMANDS:
        raise ValueError(f"command not allowlisted: {args[0]}")
    runner = sandbox or LocalSandbox()
    return runner.run(repo_root, args, min(timeout, MAX_COMMAND_TIMEOUT))
