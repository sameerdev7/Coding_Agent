"""One structured event stream over a graph run — consumed by the CLI, the TUI, the run log and the evals."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from agent.state import AgentState


@dataclass(frozen=True)
class PlanEvent:
    plan: str


@dataclass(frozen=True)
class StepEvent:
    step: int
    tool: str | None
    args: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    tests_passing: bool | None = None
    total_tokens: int = 0


class AgentRun:
    """Drives `graph.stream` and exposes it as PlanEvent / StepEvent objects.

    After `events()` is exhausted, `final_state` holds the last state the graph produced.
    """

    def __init__(self, graph: Any, state: AgentState):
        self._graph = graph
        self._state = state
        self.final_state: AgentState = state

    def events(self) -> Iterator[PlanEvent | StepEvent]:
        step = 0
        plan_emitted = False
        for snapshot in self._graph.stream(self._state, stream_mode="values"):
            self.final_state = snapshot

            if snapshot.get("plan") and not plan_emitted:
                plan_emitted = True
                yield PlanEvent(snapshot["plan"])

            iterations = snapshot["iterations"]
            if iterations != step:
                step = iterations
                call = snapshot.get("pending_tool_call")
                yield StepEvent(
                    step=step,
                    tool=call["name"] if call else None,
                    args=call["args"] if call else {},
                    result=snapshot.get("last_tool_result"),
                    tests_passing=snapshot.get("tests_passing"),
                    total_tokens=snapshot.get("total_tokens", 0),
                )
