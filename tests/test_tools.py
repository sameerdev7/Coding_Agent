import time
from pathlib import Path

import pytest

from agent import tools


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "file.txt").write_text("hello")
    return tmp_path


def test_read_write_roundtrip(repo: Path):
    tools.write_file(repo, "new.txt", "content here")
    assert tools.read_file(repo, "new.txt") == "content here"


def test_write_creates_parent_dirs(repo: Path):
    tools.write_file(repo, "nested/dir/file.txt", "x")
    assert tools.read_file(repo, "nested/dir/file.txt") == "x"


def test_read_file_missing_raises(repo: Path):
    with pytest.raises(FileNotFoundError):
        tools.read_file(repo, "does_not_exist.txt")


@pytest.mark.parametrize("escape_path", ["../../etc/passwd", "/etc/passwd", "sub/../../outside.txt"])
def test_path_escape_raises(repo: Path, escape_path: str):
    with pytest.raises(ValueError):
        tools.read_file(repo, escape_path)


def test_list_dir(repo: Path):
    entries = tools.list_dir(repo, ".")
    assert "sub/" in entries.splitlines()


def test_grep_finds_match(repo: Path):
    result = tools.grep(repo, "hello", ".")
    assert "file.txt" in result


def test_grep_no_match_does_not_raise(repo: Path):
    result = tools.grep(repo, "no_such_pattern_xyz", ".")
    assert "no matches" in result


def test_run_command_rejects_non_allowlisted(repo: Path):
    with pytest.raises(ValueError):
        tools.run_command(repo, ["rm", "-rf", "."])


def test_run_command_runs_python(repo: Path):
    result = tools.run_command(repo, ["python3", "-c", "print('hi')"])
    assert "exit=0" in result
    assert "hi" in result


def test_run_command_enforces_timeout(repo: Path):
    start = time.monotonic()
    result = tools.run_command(repo, ["python3", "-c", "import time; time.sleep(5)"], timeout=1)
    elapsed = time.monotonic() - start
    assert "timeout" in result
    assert elapsed < 4
