import logging
import os
import re
import time
from typing import Any, cast

from google import genai
from google.genai import errors, types

from agent.llm.base import ChatResponse
from agent.state import ToolCall, ToolName

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.8-flash"
MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 20
RATE_LIMIT_CODE = 429


def _to_gemini_tools(tools: list[dict]) -> list[dict]:
    return [{"function_declarations": [t["function"] for t in tools]}]


def _to_gemini_contents(messages: list[dict]) -> tuple[str | None, list[dict]]:
    system_instruction = None
    contents = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            system_instruction = msg["content"]
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": msg.get("content") or ""}]})
        else:  # user, tool
            prefix = "[TOOL RESULT] " if role == "tool" else ""
            contents.append({"role": "user", "parts": [{"text": prefix + str(msg.get("content", ""))}]})
    return system_instruction, contents


def _retry_delay_seconds(exc: errors.ClientError) -> float | None:
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s", str(exc))
    return float(match.group(1)) if match else None


def _is_daily_quota(exc: errors.ClientError) -> bool:
    return "PerDay" in str(exc)


class GeminiClient:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self._client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self._model_name = model

    def chat(self, messages: list[dict], tools: list[dict], require_tool: bool = True) -> ChatResponse:
        system_instruction, contents = _to_gemini_contents(messages)
        mode = types.FunctionCallingConfigMode.ANY if require_tool else types.FunctionCallingConfigMode.AUTO
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=cast(Any, _to_gemini_tools(tools)),
            tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode=mode)),
        )

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=cast(Any, contents),
                    config=config,
                )
            except errors.ClientError as exc:
                if exc.code != RATE_LIMIT_CODE or _is_daily_quota(exc) or attempt == MAX_RETRIES:
                    raise
                wait = _retry_delay_seconds(exc) or DEFAULT_BACKOFF_SECONDS
                logger.warning("Gemini free-tier rate limit hit, retrying in %.0fs...", wait)
                time.sleep(wait)
                continue
            return self._to_chat_response(response)
        raise RuntimeError("unreachable: retry loop ended without a response")

    @staticmethod
    def _to_chat_response(response: Any) -> ChatResponse:
        usage = getattr(response, "usage_metadata", None)
        total_tokens = (getattr(usage, "total_token_count", 0) if usage else 0) or 0
        function_calls = response.function_calls
        if function_calls:
            fc = function_calls[0]
            tool_call: ToolCall = {"name": cast(ToolName, fc.name or ""), "args": dict(fc.args or {})}
            return ChatResponse(tool_call=tool_call, content=None, total_tokens=total_tokens)
        return ChatResponse(tool_call=None, content=response.text, total_tokens=total_tokens)
