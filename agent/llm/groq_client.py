import json
import os

from groq import BadRequestError, Groq

from agent.llm.base import ChatResponse
from agent.state import ToolCall

DEFAULT_MODEL = "openai/gpt-oss-120b"


class GroqClient:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self._client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])
        self._model = model

    def chat(self, messages: list[dict], tools: list[dict]) -> ChatResponse:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools,
                tool_choice="required",
            )
        except BadRequestError as exc:
            # e.g. the model hallucinated a tool name outside our schema; let the
            # graph observe this as a correctable mistake rather than crashing.
            detail = exc.body.get("error", {}).get("message", str(exc)) if isinstance(exc.body, dict) else str(exc)
            return ChatResponse(tool_call=None, content=f"Error: {detail}")

        message = response.choices[0].message
        if not message.tool_calls:
            return ChatResponse(tool_call=None, content=message.content)

        call = message.tool_calls[0]
        tool_call: ToolCall = {
            "id": call.id,
            "name": call.function.name,
            "args": json.loads(call.function.arguments or "{}"),
        }
        return ChatResponse(tool_call=tool_call, content=message.content)
