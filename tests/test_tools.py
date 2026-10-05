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


# --- read ranges, edit_file, grep hardening, run_command guards -------------------------------------


def test_read_file_line_range(repo: Path):
    tools.write_file(repo, "lines.txt", "one\ntwo\nthree\nfour\n")
    assert tools.read_file(repo, "lines.txt", start_line=2, end_line=3) == "two\nthree\n"
    assert tools.read_file(repo, "lines.txt", start_line=3) == "three\nfour\n"
    assert tools.read_file(repo, "lines.txt", end_line=1) == "one\n"


def test_read_file_bad_range_raises(repo: Path):
    tools.write_file(repo, "lines.txt", "one\ntwo\n")
    with pytest.raises(ValueError):
        tools.read_file(repo, "lines.txt", start_line=5, end_line=2)


def test_edit_file_replaces_a_unique_snippet(repo: Path):
    tools.write_file(repo, "code.py", "a = 1\nb = 2\n")
    assert "replaced 1 occurrence" in tools.edit_file(repo, "code.py", "b = 2", "b = 3")
    assert tools.read_file(repo, "code.py") == "a = 1\nb = 3\n"


def test_edit_file_rejects_missing_ambiguous_and_empty(repo: Path):
    tools.write_file(repo, "code.py", "x = 1\nx = 1\n")
    with pytest.raises(ValueError, match="not found"):
        tools.edit_file(repo, "code.py", "nope", "y")
    with pytest.raises(ValueError, match="2 times"):
        tools.edit_file(repo, "code.py", "x = 1", "x = 2")
    with pytest.raises(ValueError, match="must not be empty"):
        tools.edit_file(repo, "code.py", "", "y")
    assert tools.read_file(repo, "code.py") == "x = 1\nx = 1\n"  # untouched after failures


def test_edit_file_is_path_jailed(repo: Path):
    with pytest.raises(ValueError, match="escapes"):
        tools.edit_file(repo, "../../etc/passwd", "root", "evil")


def test_grep_pattern_starting_with_dash_is_not_an_option(repo: Path):
    tools.write_file(repo, "flags.txt", "--help is here\n")
    assert "flags.txt" in tools.grep(repo, "--help", ".")


def test_run_command_clamps_timeout(repo: Path):
    seen = {}

    class SpySandbox:
        name = "spy"

        def run(self, repo_root, args, timeout):
            seen["timeout"] = timeout
            return "exit=0"

    tools.run_command(repo, ["pytest"], timeout=10_000, sandbox=SpySandbox())
    assert seen["timeout"] == tools.MAX_COMMAND_TIMEOUT


def test_run_command_uses_the_given_sandbox(repo: Path):
    class SpySandbox:
        name = "spy"

        def run(self, repo_root, args, timeout):
            return f"ran {args} in {repo_root.name}"

    assert tools.run_command(repo, ["pytest", "-q"], sandbox=SpySandbox()).startswith("ran ['pytest', '-q']")
