from pathlib import Path

from agent.events import AgentRun, PlanEvent, StepEvent
from agent.graph import build_graph, initial_state
from agent.llm.base import ChatResponse
from agent.llm.fake_client import FakeLLMClient


def _run(tmp_path: Path, responses, max_iterations=3) -> AgentRun:
    (tmp_path / "a.py").write_text("x = 1\n")
    graph = build_graph(FakeLLMClient(responses))
    return AgentRun(graph, initial_state("task", tmp_path, max_iterations))


def test_events_are_one_plan_then_one_step_per_iteration(tmp_path: Path):
    run = _run(
        tmp_path,
        [
            ChatResponse(tool_call=None, content="the plan", total_tokens=7),
            ChatResponse(tool_call={"name": "list_dir", "args": {"path": "."}}, total_tokens=3),
            ChatResponse(tool_call={"name": "read_file", "args": {"path": "a.py"}}, total_tokens=4),
        ],
        max_iterations=2,
    )

    events = list(run.events())

    assert events[0] == PlanEvent("the plan")
    steps = [e for e in events if isinstance(e, StepEvent)]
    assert [s.step for s in steps] == [1, 2]
    assert [s.tool for s in steps] == ["list_dir", "read_file"]
    assert steps[-1].total_tokens == 7 + 3 + 4
    assert steps[1].result == "x = 1\n"


def test_final_state_is_available_after_iteration(tmp_path: Path):
    run = _run(
        tmp_path,
        [ChatResponse(tool_call=None, content="p"), ChatResponse(tool_call={"name": "list_dir", "args": {}})],
        max_iterations=1,
    )
    list(run.events())
    assert run.final_state["iterations"] == 1


def test_a_response_without_a_tool_call_still_produces_a_step(tmp_path: Path):
    run = _run(
        tmp_path,
        [ChatResponse(tool_call=None, content="p"), ChatResponse(tool_call=None, content="I refuse")],
        max_iterations=1,
    )
    steps = [e for e in run.events() if isinstance(e, StepEvent)]
    assert steps[0].tool is None
    assert "must call a tool" in steps[0].result
