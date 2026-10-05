import subprocess
from datetime import UTC, datetime
from pathlib import Path


def _run(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        shell=False,
    )


def is_git_repo(repo_root: Path) -> bool:
    result = _run(repo_root, ["rev-parse", "--is-inside-work-tree"])
    return result.returncode == 0 and result.stdout.strip() == "true"


def is_clean(repo_root: Path) -> bool:
    result = _run(repo_root, ["status", "--porcelain"])
    return result.returncode == 0 and result.stdout.strip() == ""


def current_branch(repo_root: Path) -> str:
    result = _run(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    return result.stdout.strip()


def create_agent_branch(repo_root: Path, task: str) -> str:
    """Create and check out a fresh branch for this run. Never touches the branch you were on."""
    slug = "".join(c if c.isalnum() else "-" for c in task.lower())[:40].strip("-") or "task"
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    branch = f"agent/{slug}-{timestamp}"
    result = _run(repo_root, ["checkout", "-b", branch])
    if result.returncode != 0:
        raise RuntimeError(f"failed to create branch {branch}: {result.stderr}")
    return branch


def commit_all(repo_root: Path, message: str) -> str:
    _run(repo_root, ["add", "-A"])
    result = _run(repo_root, ["commit", "-m", message])
    if result.returncode != 0:
        raise RuntimeError(f"failed to commit: {result.stderr}")
    return result.stdout.strip()


def diff_for_path(repo_root: Path, path: str) -> str:
    """Unstaged diff for a single path, for display only — never fed back to the LLM."""
    result = _run(repo_root, ["diff", "--", path])
    return result.stdout
