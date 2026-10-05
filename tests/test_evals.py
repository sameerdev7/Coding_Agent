import json
from pathlib import Path

import pytest

from agent.evals import EvalResult, run_eval, save_results, summarize, to_markdown
from agent.llm.base import ChatResponse
from agent.llm.fake_client import FakeLLMClient

BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
TEST = "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"


@pytest.fixture
def buggy_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "buggy"
    (repo / "tests").mkdir(parents=True)
    (repo / "calc.py").write_text(BUGGY)
    (repo / "tests" / "test_calc.py").write_text(TEST)
    (repo / "pytest.ini").write_text("[pytest]\npythonpath = .\n")
    return repo


def good_agent() -> FakeLLMClient:
    return FakeLLMClient(
        [
            ChatResponse(tool_call=None, content="plan", total_tokens=10),
            ChatResponse(
                tool_call={"name": "write_file", "args": {"path": "calc.py", "content": FIXED}}, total_tokens=20
            ),
            ChatResponse(tool_call={"name": "run_command", "args": {"args": ["pytest", "-q"]}}, total_tokens=20),
            ChatResponse(tool_call={"name": "done", "args": {"summary": "fixed"}}, total_tokens=20),
        ]
    )


def lazy_agent() -> FakeLLMClient:
    """Declares done immediately without fixing anything."""
    return FakeLLMClient(
        [ChatResponse(tool_call=None, content="plan")]
        + [ChatResponse(tool_call={"name": "done", "args": {}}) for _ in range(5)]
    )


def test_successful_runs_are_counted_and_the_original_repo_is_untouched(buggy_repo: Path):
    results = run_eval([buggy_repo], runs=2, llm_factory=good_agent, max_iterations=6)

    assert [r.success for r in results] == [True, True]
    assert all(r.iterations == 3 and r.total_tokens == 70 for r in results)
    assert (buggy_repo / "calc.py").read_text() == BUGGY  # evals run on copies


def test_a_run_that_claims_done_but_did_not_fix_it_is_a_failure(buggy_repo: Path):
    results = run_eval([buggy_repo], runs=1, llm_factory=lazy_agent, max_iterations=3)
    assert results[0].success is False


def test_an_already_passing_repo_is_flagged_not_counted_as_a_win(tmp_path: Path, buggy_repo: Path):
    (buggy_repo / "calc.py").write_text(FIXED)
    results = run_eval([buggy_repo], runs=1, llm_factory=good_agent)
    assert results[0].success is False
    assert "baseline already passes" in results[0].error


def test_a_crashing_run_is_recorded_as_an_error_not_raised(buggy_repo: Path):
    def exploding_factory():
        raise ConnectionError("provider down")

    results = run_eval([buggy_repo], runs=1, llm_factory=exploding_factory)

    assert results[0].success is False
    assert "ConnectionError: provider down" in results[0].error


def test_on_result_callback_fires_per_run(buggy_repo: Path):
    seen: list[EvalResult] = []
    run_eval([buggy_repo], runs=3, llm_factory=good_agent, max_iterations=6, on_result=seen.append)
    assert len(seen) == 3


def test_summary_math_and_markdown():
    results = [
        EvalResult("a", 1, True, 4, 100, 10.0),
        EvalResult("a", 2, False, 15, 300, 30.0),
        EvalResult("b", 1, True, 6, 200, 20.0),
    ]
    rows = summarize(results)
    a, b, overall = rows

    assert a["successes"] == 1 and a["success_rate"] == 0.5
    assert a["avg_iterations_on_success"] == 4.0 and a["avg_tokens"] == 200
    assert b["success_rate"] == 1.0
    assert overall["repo"] == "**overall**" and overall["runs"] == 3 and overall["successes"] == 2

    table = to_markdown(rows)
    assert "| a | 2 | 1/2 (50%) | 4.0 | 200 | 20.0 | 0 |" in table


def test_save_results_round_trips(tmp_path: Path):
    results = [EvalResult("a", 1, True, 4, 100, 10.0)]
    out = tmp_path / "eval.json"
    save_results(out, results, {"provider": "fake"})
    payload = json.loads(out.read_text())
    assert payload["meta"] == {"provider": "fake"}
    assert payload["results"][0]["repo"] == "a"
    assert payload["summary"][0]["successes"] == 1
