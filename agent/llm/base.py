from typing import NamedTuple, Protocol

from agent.state import ToolCall


class ChatResponse(NamedTuple):
    tool_call: ToolCall | None
    content: str | None = None
    total_tokens: int = 0


class LLMClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict], require_tool: bool = True) -> ChatResponse: ...
