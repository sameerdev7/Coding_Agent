import subprocess
from pathlib import Path

ALLOWED_COMMANDS = {"pytest", "python", "python3"}


def _safe_path(repo_root: Path, user_path: str) -> Path:
    resolved = (repo_root / user_path).resolve()
    repo_root_resolved = repo_root.resolve()
    if not resolved.is_relative_to(repo_root_resolved):
        raise ValueError(f"path escapes repo root: {user_path}")
    return resolved


def read_file(repo_root: Path, path: str) -> str:
    target = _safe_path(repo_root, path)
    if not target.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    return target.read_text()


def write_file(repo_root: Path, path: str, content: str) -> str:
    target = _safe_path(repo_root, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"wrote {len(content.splitlines())} lines to {path}"


def list_dir(repo_root: Path, path: str = ".") -> str:
    target = _safe_path(repo_root, path)
    if not target.is_dir():
        raise NotADirectoryError(f"not a directory: {path}")
    entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
    return "\n".join(entries)


def grep(repo_root: Path, pattern: str, path: str = ".") -> str:
    target = _safe_path(repo_root, path)
    result = subprocess.run(
        ["grep", "-rn", pattern, str(target)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        shell=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"grep failed: {result.stderr}")
    return result.stdout or "(no matches)"


def run_command(repo_root: Path, args: list[str], timeout: int = 30) -> str:
    if not args:
        raise ValueError("empty command")
    if args[0] not in ALLOWED_COMMANDS:
        raise ValueError(f"command not allowlisted: {args[0]}")
    try:
        result = subprocess.run(
            args,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return f"exit=timeout after {timeout}s"
    return f"exit={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
