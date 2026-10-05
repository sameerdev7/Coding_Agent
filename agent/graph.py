import json
from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph

from agent import integrity, tools
from agent.context import compact_messages, truncate_output
from agent.llm.base import LLMClient
from agent.prompts import SYSTEM_PROMPT, TOOL_SCHEMAS
from agent.sandbox import Sandbox
from agent.state import WRITE_TOOLS, AgentState

TOOL_DISPATCH: dict[str, Callable[..., str]] = {
    "read_file": tools.read_file,
    "write_file": tools.write_file,
    "edit_file": tools.edit_file,
    "list_dir": tools.list_dir,
    "grep": tools.grep,
    "run_command": tools.run_command,
}

PLAN_OVERVIEW_PROMPT = (
    "Before taking any action, write a short numbered plan (3-5 steps) for how you will "
    "approach this task given the repo you're in. Do not call any tools yet — respond with "
    "the plan as plain text only."
)


def is_success(state: AgentState) -> bool:
    return bool(state["finished"] and state["tests_passing"])


def build_plan_overview_node(llm: LLMClient):
    def plan_overview_node(state: AgentState) -> dict:
        messages = state["messages"] + [{"role": "user", "content": PLAN_OVERVIEW_PROMPT}]
        response = llm.chat(compact_messages(messages), TOOL_SCHEMAS, require_tool=False)
        plan_text = response.content or "(no plan provided)"
        messages = [*messages, {"role": "assistant", "content": plan_text}]
        return {
            "messages": messages,
            "plan": plan_text,
            "total_tokens": state["total_tokens"] + response.total_tokens,
        }

    return plan_overview_node


def build_plan_node(llm: LLMClient):
    def plan_node(state: AgentState) -> dict:
        # State keeps the full history (for transcripts); the model only sees a compacted view.
        response = llm.chat(compact_messages(state["messages"]), TOOL_SCHEMAS)
        total_tokens = state["total_tokens"] + response.total_tokens
        call = response.tool_call

        if call is None:
            assistant_message: dict[str, Any] = {"role": "assistant", "content": response.content or ""}
            return {
                "messages": state["messages"] + [assistant_message],
                "pending_tool_call": None,
                "total_tokens": total_tokens,
            }

        call_id = call.get("id") or f"call_{state['iterations']}"
        call = {**call, "id": call_id}
        assistant_message = {
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": call["name"], "arguments": json.dumps(call["args"])},
                }
            ],
        }
        return {
            "messages": state["messages"] + [assistant_message],
            "pending_tool_call": call,
            "total_tokens": total_tokens,
        }

    return plan_node


def build_execute_tool_node(sandbox: Sandbox | None = None, protect_tests: bool = True):
    def execute_tool_node(state: AgentState) -> dict:
        call = state["pending_tool_call"]
        repo_root = state["repo_root"]

        if call is None:
            return {
                "last_tool_result": "Error: you must call a tool. Use `done` only once tests pass.",
                "finished": False,
            }

        if call["name"] == "done":
            snapshot = state.get("test_snapshot")
            if protect_tests and snapshot is not None:
                changed = integrity.changed_tests(repo_root, snapshot)
                if changed:
                    return {
                        "last_tool_result": (
                            "Error: cannot finish — test files were modified: "
                            f"{', '.join(changed)}. Tests are read-only; restore them and fix the implementation."
                        ),
                        "finished": False,
                    }
            return {
                "last_tool_result": "acknowledged",
                "finished": True,
                "summary": call["args"].get("summary"),
            }

        if protect_tests and call["name"] in WRITE_TOOLS:
            path = call["args"].get("path")
            if isinstance(path, str) and integrity.is_protected_write(repo_root, path):
                return {
                    "last_tool_result": (
                        f"Error: {path} is a test file and test files are read-only. "
                        "Fix the implementation instead of the tests."
                    ),
                    "finished": False,
                }

        fn = TOOL_DISPATCH.get(call["name"])
        if fn is None:
            return {"last_tool_result": f"Error: unknown tool '{call['name']}'", "finished": False}

        args = dict(call["args"])
        if call["name"] == "run_command" and sandbox is not None:
            args["sandbox"] = sandbox

        try:
            result = fn(repo_root, **args)
        except Exception as exc:
            result = f"Error: {exc}"
        return {"last_tool_result": truncate_output(result), "finished": False}

    return execute_tool_node


def observe_node(state: AgentState) -> dict:
    tests_passing = state["tests_passing"]
    call = state["pending_tool_call"]
    result = state["last_tool_result"]

    messages = state["messages"]
    if result is not None:
        if call is None:
            messages = [*messages, {"role": "user", "content": result}]
        else:
            messages = [*messages, {"role": "tool", "tool_call_id": call.get("id", ""), "content": result}]

    if call is not None and call["name"] == "run_command" and "pytest" in call["args"].get("args", []):
        if result is not None and result.startswith("exit=0"):
            tests_passing = True
        elif result is not None:
            tests_passing = False
    elif call is not None and call["name"] in WRITE_TOOLS:
        # A file changed since the last verified pytest run; require re-verification.
        tests_passing = False

    return {
        "messages": messages,
        "iterations": state["iterations"] + 1,
        "tests_passing": tests_passing,
    }


def route_node(state: AgentState) -> str:
    if is_success(state):
        return "success"
    if state["iterations"] >= state["max_iterations"]:
        return "failure"
    return "continue"


def build_graph(llm: LLMClient, sandbox: Sandbox | None = None, protect_tests: bool = True):
    graph = StateGraph(AgentState)
    graph.add_node("plan_overview", build_plan_overview_node(llm))
    graph.add_node("plan", build_plan_node(llm))
    graph.add_node("execute_tool", build_execute_tool_node(sandbox, protect_tests))
    graph.add_node("observe", observe_node)

    graph.set_entry_point("plan_overview")
    graph.add_edge("plan_overview", "plan")
    graph.add_edge("plan", "execute_tool")
    graph.add_edge("execute_tool", "observe")
    graph.add_conditional_edges(
        "observe",
        route_node,
        {"success": END, "failure": END, "continue": "plan"},
    )

    return graph.compile()


def initial_state(task: str, repo_root, max_iterations: int = 15) -> AgentState:
    return AgentState(
        task=task,
        repo_root=repo_root,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ],
        plan=None,
        pending_tool_call=None,
        last_tool_result=None,
        iterations=0,
        max_iterations=max_iterations,
        tests_passing=None,
        finished=False,
        summary=None,
        total_tokens=0,
        test_snapshot=integrity.snapshot_tests(repo_root),
    )


def continue_state(prev_state: AgentState, new_task: str) -> AgentState:
    """Start a new task in the same conversation — used by the interactive `chat` and `tui` modes.

    Keeps the message history (so the model remembers earlier edits in this session) and
    the repo root, but resets everything that tracks progress on the *previous* task. The test
    snapshot is retaken: the repo as it stands now is the baseline for the new task.
    """
    return AgentState(
        task=new_task,
        repo_root=prev_state["repo_root"],
        messages=prev_state["messages"] + [{"role": "user", "content": new_task}],
        plan=None,
        pending_tool_call=None,
        last_tool_result=None,
        iterations=0,
        max_iterations=prev_state["max_iterations"],
        tests_passing=None,
        finished=False,
        summary=None,
        total_tokens=0,
        test_snapshot=integrity.snapshot_tests(prev_state["repo_root"]),
    )
