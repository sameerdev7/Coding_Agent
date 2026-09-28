from pathlib import Path
from typing import Literal, NotRequired, TypedDict

ToolName = Literal["read_file", "write_file", "list_dir", "grep", "run_command", "done"]


class ToolCall(TypedDict):
    name: ToolName
    args: dict
    id: NotRequired[str]


class AgentState(TypedDict):
    task: str
    repo_root: Path
    messages: list[dict]
    pending_tool_call: ToolCall | None
    last_tool_result: str | None
    iterations: int
    max_iterations: int
    tests_passing: bool | None
    finished: bool
