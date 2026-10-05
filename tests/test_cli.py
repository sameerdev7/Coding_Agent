from pathlib import Path

import pytest

from agent import cli
from agent.config import Settings


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch):
    """Settings() reads ./.env — keep the developer's real keys out of these tests."""
    monkeypatch.chdir(tmp_path)
    for var in ("GROQ_API_KEY", "GEMINI_API_KEY", "LLM_PROVIDER", "SANDBOX", "PROTECT_TESTS"):
        monkeypatch.delenv(var, raising=False)


def test_parser_defaults_come_from_settings():
    parser = cli.build_parser(Settings(_env_file=None))
    args = parser.parse_args(["run", "--repo", ".", "--task", "x"])
    assert args.provider == "groq"
    assert args.sandbox == "auto"
    assert args.allow_test_edits is False
    assert args.no_git is False


def test_env_overrides_flow_through_to_flag_defaults(monkeypatch):
    monkeypatch.setenv("SANDBOX", "local")
    monkeypatch.setenv("PROTECT_TESTS", "false")
    args = cli.build_parser(Settings(_env_file=None)).parse_args(["tui", "--repo", "."])
    assert args.sandbox == "local"
    assert args.allow_test_edits is True


def test_every_command_is_registered():
    parser = cli.build_parser(Settings(_env_file=None))
    for argv in (["run", "--repo", ".", "--task", "t"], ["chat", "--repo", "."], ["tui", "--repo", "."], ["eval"]):
        assert parser.parse_args(argv).command == argv[0]


def test_missing_repo_exits_1(capsys):
    code = cli.main(["run", "--repo", "/definitely/not/here", "--task", "x", "--sandbox", "local"])
    assert code == 1
    assert "does not exist" in capsys.readouterr().err


def test_missing_api_key_is_a_clear_error_not_a_traceback(tmp_path: Path, capsys):
    code = cli.main(["run", "--repo", str(tmp_path), "--task", "x", "--sandbox", "local"])
    assert code == 2
    assert "GROQ_API_KEY" in capsys.readouterr().err


def test_requesting_docker_without_docker_fails_loudly(tmp_path: Path, monkeypatch, capsys):
    from agent import sandbox as sandbox_module

    monkeypatch.setattr(sandbox_module, "docker_available", lambda: False)
    code = cli.main(["run", "--repo", str(tmp_path), "--task", "x", "--sandbox", "docker"])
    assert code == 2
    assert "docker" in capsys.readouterr().err


def test_eval_rejects_missing_repos(capsys):
    code = cli.main(["eval", "--repos", "/nope", "--sandbox", "local"])
    assert code == 1
    assert "do not exist" in capsys.readouterr().err


# --- end-to-end: real CLI, real git, real (local) sandbox, scripted model ---------------------------

import json  # noqa: E402
import subprocess  # noqa: E402

from agent.llm.base import ChatResponse  # noqa: E402
from agent.llm.fake_client import FakeLLMClient  # noqa: E402

BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def bug_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "bug"
    (repo / "tests").mkdir(parents=True)
    (repo / "calc.py").write_text(BUGGY)
    (repo / "tests" / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    )
    (repo / "pytest.ini").write_text("[pytest]\npythonpath = .\n")
    return repo


@pytest.fixture
def git_bug_repo(bug_repo: Path) -> Path:
    _git(bug_repo, "init", "-q")
    _git(bug_repo, "config", "user.email", "t@example.com")
    _git(bug_repo, "config", "user.name", "T")
    (bug_repo / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    _git(bug_repo, "add", "-A")
    _git(bug_repo, "commit", "-q", "-m", "initial")
    return bug_repo


def fixing_agent() -> FakeLLMClient:
    return FakeLLMClient(
        [
            ChatResponse(tool_call=None, content="1. fix 2. verify", total_tokens=10),
            ChatResponse(
                tool_call={"name": "edit_file", "args": {"path": "calc.py", "old_text": "a - b", "new_text": "a + b"}},
                total_tokens=20,
            ),
            ChatResponse(tool_call={"name": "run_command", "args": {"args": ["pytest", "-q"]}}, total_tokens=20),
            ChatResponse(tool_call={"name": "done", "args": {"summary": "fixed add"}}, total_tokens=20),
        ]
    )


def lazy_agent() -> FakeLLMClient:
    return FakeLLMClient(
        [ChatResponse(tool_call=None, content="plan")] + [ChatResponse(tool_call={"name": "done", "args": {}})] * 3
    )


def test_run_end_to_end_fixes_the_bug_and_writes_log_and_transcript(
    bug_repo: Path, tmp_path: Path, monkeypatch, capsys
):
    monkeypatch.setattr(cli, "build_llm", lambda settings, provider: fixing_agent())
    log, transcript = tmp_path / "run.jsonl", tmp_path / "transcript.json"

    code = cli.main(
        [
            "run",
            "--repo",
            str(bug_repo),
            "--task",
            "fix it",
            "--sandbox",
            "local",
            "--no-git",
            "--run-log",
            str(log),
            "--transcript-out",
            str(transcript),
        ]
    )

    assert code == 0
    assert (bug_repo / "calc.py").read_text() == FIXED
    out = capsys.readouterr().out
    assert "Task complete in 3 iterations (70 tokens)" in out
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert [e["event"] for e in events] == ["run_start", "plan", "step", "step", "step", "run_end"]
    assert events[-1]["success"] is True and events[-1]["total_tokens"] == 70
    assert any(m["role"] == "tool" for m in json.loads(transcript.read_text()))


def test_run_reports_failure_with_exit_code_1(bug_repo: Path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_llm", lambda settings, provider: lazy_agent())
    code = cli.main(
        ["run", "--repo", str(bug_repo), "--task", "x", "--sandbox", "local", "--no-git", "--max-iterations", "2"]
    )
    assert code == 1
    assert "Task failed" in capsys.readouterr().out
    assert (bug_repo / "calc.py").read_text() == BUGGY


def test_git_workflow_branches_and_commits_only_on_success(git_bug_repo: Path, monkeypatch):
    original = _git(git_bug_repo, "rev-parse", "--abbrev-ref", "HEAD")
    monkeypatch.setattr(cli, "build_llm", lambda settings, provider: fixing_agent())

    assert cli.main(["run", "--repo", str(git_bug_repo), "--task", "fix add", "--sandbox", "local"]) == 0

    branch = _git(git_bug_repo, "rev-parse", "--abbrev-ref", "HEAD")
    assert branch.startswith("agent/fix-add-") and branch != original
    assert _git(git_bug_repo, "log", "-1", "--format=%s") == "agent: fixed add"
    assert _git(git_bug_repo, "status", "--porcelain") == ""
    assert _git(git_bug_repo, "show", f"{original}:calc.py") == BUGGY.strip()  # original branch untouched


def test_git_workflow_leaves_failed_attempts_uncommitted(git_bug_repo: Path, monkeypatch):
    monkeypatch.setattr(cli, "build_llm", lambda settings, provider: lazy_agent())
    cli.main(["run", "--repo", str(git_bug_repo), "--task", "x", "--sandbox", "local", "--max-iterations", "2"])
    assert _git(git_bug_repo, "log", "--format=%s") == "initial"


def test_git_workflow_is_skipped_on_a_dirty_repo(git_bug_repo: Path, monkeypatch, capsys):
    original = _git(git_bug_repo, "rev-parse", "--abbrev-ref", "HEAD")
    (git_bug_repo / "notes.txt").write_text("my uncommitted work\n")
    monkeypatch.setattr(cli, "build_llm", lambda settings, provider: fixing_agent())

    assert cli.main(["run", "--repo", str(git_bug_repo), "--task", "fix", "--sandbox", "local"]) == 0

    assert _git(git_bug_repo, "rev-parse", "--abbrev-ref", "HEAD") == original
    assert "uncommitted changes" in capsys.readouterr().out
    assert (git_bug_repo / "notes.txt").read_text() == "my uncommitted work\n"


def test_agent_cannot_cheat_by_editing_tests_end_to_end(bug_repo: Path, monkeypatch):
    cheating = FakeLLMClient(
        [
            ChatResponse(tool_call=None, content="plan"),
            ChatResponse(
                tool_call={
                    "name": "write_file",
                    "args": {"path": "tests/test_calc.py", "content": "def test_add():\n    assert True\n"},
                }
            ),
            ChatResponse(tool_call={"name": "run_command", "args": {"args": ["pytest", "-q"]}}),
            ChatResponse(tool_call={"name": "done", "args": {}}),
        ]
    )
    monkeypatch.setattr(cli, "build_llm", lambda settings, provider: cheating)

    code = cli.main(
        ["run", "--repo", str(bug_repo), "--task", "x", "--sandbox", "local", "--no-git", "--max-iterations", "3"]
    )

    assert code == 1
    assert "assert add(1, 2) == 3" in (bug_repo / "tests" / "test_calc.py").read_text()


def test_allow_test_edits_flag_lifts_the_protection(bug_repo: Path, monkeypatch):
    rewriting = FakeLLMClient(
        [
            ChatResponse(tool_call=None, content="plan"),
            ChatResponse(
                tool_call={"name": "write_file", "args": {"path": "tests/test_calc.py", "content": "# rewritten\n"}}
            ),
        ]
    )
    monkeypatch.setattr(cli, "build_llm", lambda settings, provider: rewriting)
    cli.main(
        [
            "run",
            "--repo",
            str(bug_repo),
            "--task",
            "x",
            "--sandbox",
            "local",
            "--no-git",
            "--max-iterations",
            "1",
            "--allow-test-edits",
        ]
    )
    assert (bug_repo / "tests" / "test_calc.py").read_text() == "# rewritten\n"


def test_chat_handles_multiple_tasks_in_one_session(bug_repo: Path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_llm", lambda settings, provider: fixing_agent())
    typed = iter(["fix it", "", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(typed))

    code = cli.main(["chat", "--repo", str(bug_repo), "--sandbox", "local", "--no-git"])

    assert code == 0
    out = capsys.readouterr().out
    assert "Task complete" in out and "Goodbye." in out
    assert (bug_repo / "calc.py").read_text() == FIXED


def test_eval_command_end_to_end(bug_repo: Path, tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_llm", lambda settings, provider: fixing_agent())
    out = tmp_path / "eval.json"

    code = cli.main(
        [
            "eval",
            "--repos",
            str(bug_repo),
            "--runs",
            "2",
            "--sandbox",
            "local",
            "--out",
            str(out),
            "--min-success-rate",
            "1.0",
        ]
    )

    assert code == 0
    printed = capsys.readouterr().out
    assert "2/2 (100%)" in printed
    payload = json.loads(out.read_text())
    assert payload["meta"]["runs_per_repo"] == 2 and len(payload["results"]) == 2
    assert (bug_repo / "calc.py").read_text() == BUGGY  # evals never touch the original


def test_eval_exits_nonzero_below_min_success_rate(bug_repo: Path, monkeypatch):
    monkeypatch.setattr(cli, "build_llm", lambda settings, provider: lazy_agent())
    code = cli.main(
        [
            "eval",
            "--repos",
            str(bug_repo),
            "--runs",
            "1",
            "--sandbox",
            "local",
            "--max-iterations",
            "2",
            "--min-success-rate",
            "0.5",
        ]
    )
    assert code == 1
