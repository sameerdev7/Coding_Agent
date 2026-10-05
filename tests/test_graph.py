from pathlib import Path

import pytest

import agent.graph as graph_module
from agent.graph import build_graph, continue_state, initial_state
from agent.llm.base import ChatResponse
from agent.llm.fake_client import FakeLLMClient

PLAN_OVERVIEW_RESPONSE = ChatResponse(
    tool_call=None, content="1. Run tests. 2. Read the file. 3. Fix it. 4. Verify.", total_tokens=10
)


def call(name: str, **args) -> ChatResponse:
    return ChatResponse(tool_call={"name": name, "args": args}, total_tokens=5)


def make_repo(tmp_path: Path) -> Path:
    (tmp_path / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_calculator.py").write_text("def test_add():\n    assert True\n")
    return tmp_path


def run_graph(llm, repo_root: Path, max_iterations: int = 15, **graph_kwargs):
    graph = build_graph(llm, **graph_kwargs)
    return graph.invoke(initial_state("fix it", repo_root, max_iterations=max_iterations))


def test_graph_succeeds_on_scripted_fix(tmp_path: Path, monkeypatch):
    repo_root = make_repo(tmp_path)
    llm = FakeLLMClient(
        [
            PLAN_OVERVIEW_RESPONSE,
            call("run_command", args=["pytest", "-q"]),
            call("read_file", path="calculator.py"),
            call("write_file", path="calculator.py", content="fixed"),
            call("run_command", args=["pytest", "-q"]),
            call("done"),
        ]
    )
    runs = {"n": 0}

    def fake_run_command(repo_root, args, timeout=30):
        runs["n"] += 1
        return "exit=1\nstdout:\n1 failed\n" if runs["n"] == 1 else "exit=0\nstdout:\n1 passed\n"

    monkeypatch.setitem(graph_module.TOOL_DISPATCH, "run_command", fake_run_command)

    final_state = run_graph(llm, repo_root)

    assert final_state["finished"] is True
    assert final_state["tests_passing"] is True
    assert final_state["iterations"] == 5


def test_graph_fails_when_never_done(tmp_path: Path):
    repo_root = make_repo(tmp_path)
    llm = FakeLLMClient([PLAN_OVERVIEW_RESPONSE] + [call("list_dir", path=".") for _ in range(20)])

    final_state = run_graph(llm, repo_root, max_iterations=5)

    assert final_state["finished"] is False
    assert final_state["iterations"] == 5


def test_graph_rejects_premature_done_without_test_run(tmp_path: Path):
    repo_root = make_repo(tmp_path)
    llm = FakeLLMClient([PLAN_OVERVIEW_RESPONSE, call("done"), call("list_dir", path="."), call("list_dir", path=".")])

    final_state = run_graph(llm, repo_root, max_iterations=3)

    assert final_state["tests_passing"] is None
    assert final_state["iterations"] == 3


def test_plan_overview_runs_once_and_does_not_count_as_an_iteration(tmp_path: Path):
    repo_root = make_repo(tmp_path)
    llm = FakeLLMClient([PLAN_OVERVIEW_RESPONSE, call("done", summary="nothing to do")])

    final_state = run_graph(llm, repo_root, max_iterations=1)

    assert final_state["plan"] == PLAN_OVERVIEW_RESPONSE.content
    assert final_state["iterations"] == 1
    assert final_state["summary"] == "nothing to do"
    assert llm.calls[0]["require_tool"] is False
    assert llm.calls[1]["require_tool"] is True


def test_token_usage_accumulates_across_calls(tmp_path: Path):
    repo_root = make_repo(tmp_path)
    llm = FakeLLMClient([PLAN_OVERVIEW_RESPONSE, call("list_dir", path="."), call("list_dir", path=".")])

    final_state = run_graph(llm, repo_root, max_iterations=2)

    assert final_state["total_tokens"] == 10 + 5 + 5


# --- test-file protection -------------------------------------------------------------------------


def test_writes_to_test_files_are_rejected(tmp_path: Path):
    repo_root = make_repo(tmp_path)
    original = (repo_root / "tests" / "test_calculator.py").read_text()
    llm = FakeLLMClient(
        [
            PLAN_OVERVIEW_RESPONSE,
            call("write_file", path="tests/test_calculator.py", content="def test_add():\n    pass\n"),
            call("edit_file", path="./tests/../tests/test_calculator.py", old_text="True", new_text="False"),
        ]
    )

    final_state = run_graph(llm, repo_root, max_iterations=2)

    assert (repo_root / "tests" / "test_calculator.py").read_text() == original
    tool_results = [m["content"] for m in final_state["messages"] if m.get("role") == "tool"]
    assert all("read-only" in result for result in tool_results)


def test_protection_can_be_disabled(tmp_path: Path):
    repo_root = make_repo(tmp_path)
    llm = FakeLLMClient(
        [
            PLAN_OVERVIEW_RESPONSE,
            call("write_file", path="tests/test_calculator.py", content="# rewritten\n"),
        ]
    )

    run_graph(llm, repo_root, max_iterations=1, protect_tests=False)

    assert (repo_root / "tests" / "test_calculator.py").read_text() == "# rewritten\n"


def test_done_is_refused_if_tests_were_modified_out_of_band(tmp_path: Path, monkeypatch):
    """`python -c ...` can write anywhere in the repo, so protection also checks at `done`."""
    repo_root = make_repo(tmp_path)

    def sneaky_run_command(repo_root, args, timeout=30):
        (repo_root / "tests" / "test_calculator.py").write_text("def test_add():\n    assert 1\n")
        return "exit=0\nstdout:\n1 passed\n"

    monkeypatch.setitem(graph_module.TOOL_DISPATCH, "run_command", sneaky_run_command)
    llm = FakeLLMClient([PLAN_OVERVIEW_RESPONSE, call("run_command", args=["pytest"]), call("done")])

    final_state = run_graph(llm, repo_root, max_iterations=2)

    assert final_state["finished"] is False
    last_tool_result = [m["content"] for m in final_state["messages"] if m.get("role") == "tool"][-1]
    assert "test files were modified" in last_tool_result
    assert "tests/test_calculator.py (modified)" in last_tool_result


# --- context control ------------------------------------------------------------------------------


def test_tool_output_is_truncated(tmp_path: Path):
    repo_root = make_repo(tmp_path)
    (repo_root / "big.txt").write_text("x" * 50_000)
    llm = FakeLLMClient([PLAN_OVERVIEW_RESPONSE, call("read_file", path="big.txt")])

    final_state = run_graph(llm, repo_root, max_iterations=1)

    assert len(final_state["last_tool_result"]) < 7_000
    assert "characters truncated" in final_state["last_tool_result"]


def test_llm_sees_compacted_history_but_state_keeps_everything(tmp_path: Path):
    repo_root = make_repo(tmp_path)
    (repo_root / "big.txt").write_text("y" * 2_000)
    llm = FakeLLMClient(
        [PLAN_OVERVIEW_RESPONSE] + [call("read_file", path="big.txt") for _ in range(10)] + [call("list_dir", path=".")]
    )

    final_state = run_graph(llm, repo_root, max_iterations=11)

    full_tool_messages = [m["content"] for m in final_state["messages"] if m.get("role") == "tool"]
    assert all(len(c) >= 2_000 for c in full_tool_messages[:10])  # state is untouched

    last_seen = llm.calls[-1]["messages"]
    seen_tool_messages = [m["content"] for m in last_seen if m.get("role") == "tool"]
    assert "elided" in seen_tool_messages[0]  # oldest exchange shrunk for the model
    assert "elided" not in seen_tool_messages[-1]  # newest exchange verbatim


# --- edit_file, chat continuation -----------------------------------------------------------------


def test_edit_file_applies_and_invalidates_test_status(tmp_path: Path):
    repo_root = make_repo(tmp_path)
    llm = FakeLLMClient(
        [
            PLAN_OVERVIEW_RESPONSE,
            call("edit_file", path="calculator.py", old_text="a + b", new_text="a - b"),
        ]
    )

    final_state = run_graph(llm, repo_root, max_iterations=1)

    assert "a - b" in (repo_root / "calculator.py").read_text()
    assert final_state["tests_passing"] is False


def test_continue_state_keeps_history_and_resets_progress(tmp_path: Path):
    repo_root = make_repo(tmp_path)
    llm = FakeLLMClient([PLAN_OVERVIEW_RESPONSE, call("list_dir", path=".")])
    first = run_graph(llm, repo_root, max_iterations=1)

    second = continue_state(first, "now do something else")

    assert second["messages"][: len(first["messages"])] == first["messages"]
    assert second["messages"][-1] == {"role": "user", "content": "now do something else"}
    assert second["iterations"] == 0
    assert second["total_tokens"] == 0
    assert second["finished"] is False


def test_unknown_tool_arguments_become_an_observation_not_a_crash(tmp_path: Path):
    repo_root = make_repo(tmp_path)
    llm = FakeLLMClient([PLAN_OVERVIEW_RESPONSE, call("read_file", path="calculator.py", line_end=400)])

    final_state = run_graph(llm, repo_root, max_iterations=1)

    assert final_state["last_tool_result"].startswith("Error:")
    assert "line_end" in final_state["last_tool_result"]


@pytest.mark.parametrize("bad", ["rm", "bash", "sh"])
def test_non_allowlisted_command_is_an_observation_not_a_crash(tmp_path: Path, bad: str):
    repo_root = make_repo(tmp_path)
    llm = FakeLLMClient([PLAN_OVERVIEW_RESPONSE, call("run_command", args=[bad, "-rf", "."])])

    final_state = run_graph(llm, repo_root, max_iterations=1)

    assert "not allowlisted" in final_state["last_tool_result"]
