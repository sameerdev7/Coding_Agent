import json

from langgraph.graph import END, StateGraph

from agent import tools
from agent.llm.base import LLMClient
from agent.prompts import SYSTEM_PROMPT, TOOL_SCHEMAS
from agent.state import AgentState

TOOL_DISPATCH = {
    "read_file": tools.read_file,
    "write_file": tools.write_file,
    "list_dir": tools.list_dir,
    "grep": tools.grep,
    "run_command": tools.run_command,
}


def build_plan_node(llm: LLMClient):
    def plan_node(state: AgentState) -> dict:
        response = llm.chat(state["messages"], TOOL_SCHEMAS)
        call = response.tool_call

        if call is None:
            assistant_message = {"role": "assistant", "content": response.content or ""}
            return {
                "messages": state["messages"] + [assistant_message],
                "pending_tool_call": None,
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
        return {"messages": state["messages"] + [assistant_message], "pending_tool_call": call}

    return plan_node


def execute_tool_node(state: AgentState) -> dict:
    call = state["pending_tool_call"]
    if call is None:
        return {
            "last_tool_result": "Error: you must call a tool. Use `done` only once tests pass.",
            "finished": False,
        }

    if call["name"] == "done":
        return {"last_tool_result": "acknowledged", "finished": True}

    fn = TOOL_DISPATCH.get(call["name"])
    if fn is None:
        return {"last_tool_result": f"Error: unknown tool '{call['name']}'", "finished": False}

    try:
        result = fn(state["repo_root"], **call["args"])
    except Exception as exc:  # noqa: BLE001 - tool errors become observations, not crashes
        result = f"Error: {exc}"
    return {"last_tool_result": result, "finished": False}


def observe_node(state: AgentState) -> dict:
    tests_passing = state["tests_passing"]
    call = state["pending_tool_call"]
    result = state["last_tool_result"]

    messages = state["messages"]
    if result is not None:
        if call is None:
            messages = messages + [{"role": "user", "content": result}]
        else:
            messages = messages + [
                {"role": "tool", "tool_call_id": call.get("id", ""), "content": result}
            ]

    if call is not None and call["name"] == "run_command" and "pytest" in call["args"].get("args", []):
        if result is not None and result.startswith("exit=0"):
            tests_passing = True
        elif result is not None:
            tests_passing = False
    elif call is not None and call["name"] == "write_file":
        # A file changed since the last verified pytest run; require re-verification.
        tests_passing = False

    return {
        "messages": messages,
        "iterations": state["iterations"] + 1,
        "tests_passing": tests_passing,
    }


def route_node(state: AgentState) -> str:
    if state["finished"] and state["tests_passing"]:
        return "success"
    if state["iterations"] >= state["max_iterations"]:
        return "failure"
    return "continue"


def build_graph(llm: LLMClient):
    graph = StateGraph(AgentState)
    graph.add_node("plan", build_plan_node(llm))
    graph.add_node("execute_tool", execute_tool_node)
    graph.add_node("observe", observe_node)

    graph.set_entry_point("plan")
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
        pending_tool_call=None,
        last_tool_result=None,
        iterations=0,
        max_iterations=max_iterations,
        tests_passing=None,
        finished=False,
    )
