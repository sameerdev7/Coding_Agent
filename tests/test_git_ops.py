import subprocess
from pathlib import Path

import pytest

from agent import git_ops


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "file.txt").write_text("hello\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "initial")
    return tmp_path


def test_is_git_repo_true_for_real_repo(repo: Path):
    assert git_ops.is_git_repo(repo) is True


def test_is_git_repo_false_for_plain_directory(tmp_path: Path):
    assert git_ops.is_git_repo(tmp_path) is False


def test_is_clean_true_before_changes(repo: Path):
    assert git_ops.is_clean(repo) is True


def test_is_clean_false_after_changes(repo: Path):
    (repo / "file.txt").write_text("changed\n")
    assert git_ops.is_clean(repo) is False


def test_create_agent_branch_switches_branch_without_touching_original(repo: Path):
    original = git_ops.current_branch(repo)
    branch = git_ops.create_agent_branch(repo, "Make all tests pass!")
    assert git_ops.current_branch(repo) == branch
    assert branch.startswith("agent/make-all-tests-pass")
    assert branch != original


def test_commit_all_commits_working_tree_changes(repo: Path):
    (repo / "file.txt").write_text("changed\n")
    (repo / "new.txt").write_text("new\n")
    git_ops.commit_all(repo, "agent: fix the thing")
    assert git_ops.is_clean(repo) is True


def test_diff_for_path_shows_the_change(repo: Path):
    (repo / "file.txt").write_text("changed\n")
    diff = git_ops.diff_for_path(repo, "file.txt")
    assert "-hello" in diff
    assert "+changed" in diff
