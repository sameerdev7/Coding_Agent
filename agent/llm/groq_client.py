import json
import logging
import os
import re
import time
from typing import Any, cast

from groq import BadRequestError, Groq, RateLimitError

from agent.llm.base import ChatResponse
from agent.state import ToolCall, ToolName

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 5


def _retry_delay_seconds(exc: RateLimitError) -> float | None:
    match = re.search(r"try again in ([\d.]+)s", str(exc))
    return float(match.group(1)) if match else None


def _to_chat_response(response: Any) -> ChatResponse:
    message = response.choices[0].message
    usage = getattr(response, "usage", None)
    total_tokens = (usage.total_tokens if usage else 0) or 0
    if not message.tool_calls:
        return ChatResponse(tool_call=None, content=message.content, total_tokens=total_tokens)

    call = message.tool_calls[0]
    tool_call: ToolCall = {
        "id": call.id,
        "name": cast(ToolName, call.function.name),
        "args": json.loads(call.function.arguments or "{}"),
    }
    return ChatResponse(tool_call=tool_call, content=message.content, total_tokens=total_tokens)


class GroqClient:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self._client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])
        self._model = model

    def chat(self, messages: list[dict], tools: list[dict], require_tool: bool = True) -> ChatResponse:
        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "required" if require_tool else "auto",
        }
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(**request)
            except BadRequestError as exc:
                # e.g. the model hallucinated a tool name outside our schema; let the
                # graph observe this as a correctable mistake rather than crashing.
                detail = exc.body.get("error", {}).get("message", str(exc)) if isinstance(exc.body, dict) else str(exc)
                return ChatResponse(tool_call=None, content=f"Error: {detail}")
            except RateLimitError as exc:
                if attempt == MAX_RETRIES:
                    raise
                wait = _retry_delay_seconds(exc) or DEFAULT_BACKOFF_SECONDS
                logger.warning("Groq rate limit hit, retrying in %.1fs...", wait)
                time.sleep(wait)
                continue
            return _to_chat_response(response)
        raise RuntimeError("unreachable: retry loop ended without a response")
