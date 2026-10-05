from pathlib import Path

import pytest
from textual.widgets import Input, RichLog

from agent.llm.base import ChatResponse
from agent.llm.fake_client import FakeLLMClient
from agent.tui import CodingAgentTUI

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def make_repo(tmp_path: Path) -> Path:
    (tmp_path / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
    return tmp_path


async def test_app_mounts_with_expected_widgets(tmp_path: Path):
    repo_root = make_repo(tmp_path)
    llm = FakeLLMClient([])
    app = CodingAgentTUI(llm, repo_root, max_iterations=5, use_git=False)

    async with app.run_test():
        assert app.query_one("#log", RichLog) is not None
        assert app.query_one("#task_input", Input) is not None
        assert str(repo_root) in app.sub_title or repo_root.name in app.sub_title


async def test_submitting_a_task_runs_the_graph_to_completion(tmp_path: Path, monkeypatch):
    repo_root = make_repo(tmp_path)
    responses = [
        ChatResponse(tool_call=None, content="1. Check tests. 2. Fix. 3. Verify."),
        ChatResponse(tool_call={"name": "run_command", "args": {"args": ["pytest", "-q"]}}),
        ChatResponse(tool_call={"name": "done", "args": {"summary": "nothing to fix"}}),
    ]
    llm = FakeLLMClient(responses)
    app = CodingAgentTUI(llm, repo_root, max_iterations=5, use_git=False)

    import agent.graph as graph_module

    monkeypatch.setitem(graph_module.TOOL_DISPATCH, "run_command", lambda repo_root, args, timeout=30: "exit=0\n")

    async with app.run_test() as pilot:
        task_input = app.query_one("#task_input", Input)
        task_input.value = "make tests pass"
        await pilot.press("enter")
        await pilot.pause(0.2)

        assert app.state is not None
        assert app.state["finished"] is True
        assert "📋 plan" in "".join(str(line) for line in app.query_one("#log", RichLog).lines)
