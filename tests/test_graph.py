from pathlib import Path

from agent.graph import build_graph, initial_state
from agent.llm.base import ChatResponse
from agent.llm.fake_client import FakeLLMClient


def make_repo(tmp_path: Path) -> Path:
    (tmp_path / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
    return tmp_path


def test_graph_succeeds_on_scripted_fix(tmp_path: Path):
    repo_root = make_repo(tmp_path)
    responses = [
        ChatResponse(tool_call={"name": "run_command", "args": {"args": ["pytest", "-q"]}}),
        ChatResponse(tool_call={"name": "read_file", "args": {"path": "calculator.py"}}),
        ChatResponse(tool_call={"name": "write_file", "args": {"path": "calculator.py", "content": "fixed"}}),
        ChatResponse(tool_call={"name": "run_command", "args": {"args": ["pytest", "-q"]}}),
        ChatResponse(tool_call={"name": "done", "args": {}}),
    ]
    llm = FakeLLMClient(responses)

    call_count = {"n": 0}

    def fake_run_command(repo_root, args, timeout=30):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "exit=1\nstdout:\n1 failed\nstderr:\n"
        return "exit=0\nstdout:\n1 passed\nstderr:\n"

    import agent.graph as graph_module

    graph_module.TOOL_DISPATCH["run_command"] = fake_run_command

    graph = build_graph(llm)
    state = initial_state("fix it", repo_root, max_iterations=15)
    final_state = graph.invoke(state)

    assert final_state["finished"] is True
    assert final_state["tests_passing"] is True
    assert final_state["iterations"] == 5


def test_graph_fails_when_never_done(tmp_path: Path):
    repo_root = make_repo(tmp_path)
    responses = [
        ChatResponse(tool_call={"name": "list_dir", "args": {"path": "."}}) for _ in range(20)
    ]
    llm = FakeLLMClient(responses)

    graph = build_graph(llm)
    state = initial_state("fix it", repo_root, max_iterations=5)
    final_state = graph.invoke(state)

    assert final_state["finished"] is False
    assert final_state["iterations"] == 5


def test_graph_rejects_premature_done_without_test_run(tmp_path: Path):
    repo_root = make_repo(tmp_path)
    responses = [
        ChatResponse(tool_call={"name": "done", "args": {}}),
        ChatResponse(tool_call={"name": "list_dir", "args": {"path": "."}}),
        ChatResponse(tool_call={"name": "list_dir", "args": {"path": "."}}),
    ]
    llm = FakeLLMClient(responses)

    graph = build_graph(llm)
    state = initial_state("fix it", repo_root, max_iterations=3)
    final_state = graph.invoke(state)

    assert final_state["tests_passing"] is None
    assert final_state["iterations"] == 3
