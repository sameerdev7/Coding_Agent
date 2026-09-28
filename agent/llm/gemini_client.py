import os
import re
import time

from google import genai
from google.genai import errors, types

from agent.llm.base import ChatResponse
from agent.state import ToolCall

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

    def chat(self, messages: list[dict], tools: list[dict]) -> ChatResponse:
        system_instruction, contents = _to_gemini_contents(messages)
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=_to_gemini_tools(tools),
        )

        response = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=contents,
                    config=config,
                )
                break
            except errors.ClientError as exc:
                if exc.code != RATE_LIMIT_CODE or _is_daily_quota(exc) or attempt == MAX_RETRIES:
                    raise
                wait = _retry_delay_seconds(exc) or DEFAULT_BACKOFF_SECONDS
                print(f"Gemini free-tier rate limit hit, retrying in {wait:.0f}s...")
                time.sleep(wait)

        function_calls = response.function_calls
        if function_calls:
            fc = function_calls[0]
            tool_call: ToolCall = {"name": fc.name, "args": dict(fc.args or {})}
            return ChatResponse(tool_call=tool_call, content=None)
        return ChatResponse(tool_call=None, content=response.text)
