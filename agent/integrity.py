"""Guards against the cheapest way to "make tests pass": editing the tests."""

import hashlib
import os
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

_IGNORED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def is_test_path(rel_path: str) -> bool:
    parts = PurePosixPath(rel_path.replace("\\", "/")).parts
    if not parts or any(part in _IGNORED_DIRS for part in parts):
        return False
    name = parts[-1]
    in_test_dir = "tests" in parts[:-1] or "test" in parts[:-1]
    return in_test_dir or name == "conftest.py" or fnmatch(name, "test_*.py") or fnmatch(name, "*_test.py")


def repo_relative(repo_root: Path, path: str) -> str | None:
    """Repo-relative POSIX path after resolving `..` and symlinks, or None if it escapes the repo."""
    resolved = (repo_root / path).resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return None


def is_protected_write(repo_root: Path, path: str) -> bool:
    rel = repo_relative(repo_root, path)
    return rel is not None and is_test_path(rel)


def snapshot_tests(repo_root: Path) -> dict[str, str]:
    """sha256 of every test file (anything matching `is_test_path`) under the repo."""
    root = repo_root.resolve()
    snapshot: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _IGNORED_DIRS]
        for filename in filenames:
            full = Path(dirpath) / filename
            rel = full.relative_to(root).as_posix()
            if is_test_path(rel) and not rel.endswith(".pyc"):
                snapshot[rel] = hashlib.sha256(full.read_bytes()).hexdigest()
    return snapshot


def changed_tests(repo_root: Path, snapshot: dict[str, str]) -> list[str]:
    current = snapshot_tests(repo_root)
    changes = []
    for rel in sorted(set(snapshot) | set(current)):
        if rel not in current:
            changes.append(f"{rel} (deleted)")
        elif rel not in snapshot:
            changes.append(f"{rel} (added)")
        elif snapshot[rel] != current[rel]:
            changes.append(f"{rel} (modified)")
    return changes
