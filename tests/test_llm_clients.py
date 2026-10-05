"""Provider adapters, tested against stubbed SDK clients — retry, recovery and parsing, no network."""

from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors
from groq import BadRequestError, RateLimitError

from agent.llm import gemini_client, groq_client
from agent.llm.gemini_client import GeminiClient
from agent.llm.groq_client import GroqClient

TOOLS = [{"type": "function", "function": {"name": "list_dir", "parameters": {}}}]
MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(groq_client.time, "sleep", sleeps.append)
    monkeypatch.setattr(gemini_client.time, "sleep", sleeps.append)
    return sleeps


# --- Groq -------------------------------------------------------------------------------------------


def _http_error(cls, status: int, message: str, body=None):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return cls(message, response=httpx.Response(status, request=request), body=body)


def _groq_response(tool_name=None, arguments="{}", content=None, total_tokens=42):
    tool_calls = None
    if tool_name:
        tool_calls = [SimpleNamespace(id="call_1", function=SimpleNamespace(name=tool_name, arguments=arguments))]
    message = SimpleNamespace(tool_calls=tool_calls, content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=SimpleNamespace(total_tokens=total_tokens))


class StubGroq:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def groq_with(*outcomes) -> tuple[GroqClient, StubGroq]:
    client = GroqClient(api_key="test")
    stub = StubGroq(*outcomes)
    client._client = stub  # type: ignore[assignment]
    return client, stub


def test_groq_parses_a_tool_call_and_token_usage():
    client, _ = groq_with(_groq_response("read_file", '{"path": "a.py"}', total_tokens=77))
    response = client.chat(MESSAGES, TOOLS)
    assert response.tool_call == {"id": "call_1", "name": "read_file", "args": {"path": "a.py"}}
    assert response.total_tokens == 77


def test_groq_tolerates_empty_arguments():
    client, _ = groq_with(_groq_response("done", arguments=""))
    assert client.chat(MESSAGES, TOOLS).tool_call["args"] == {}


def test_groq_text_only_response():
    client, _ = groq_with(_groq_response(content="here is my plan"))
    response = client.chat(MESSAGES, TOOLS, require_tool=False)
    assert response.tool_call is None and response.content == "here is my plan"


def test_groq_tool_choice_follows_require_tool():
    client, stub = groq_with(_groq_response("done"), _groq_response(content="plan"))
    client.chat(MESSAGES, TOOLS, require_tool=True)
    client.chat(MESSAGES, TOOLS, require_tool=False)
    assert [r["tool_choice"] for r in stub.requests] == ["required", "auto"]


def test_groq_bad_request_becomes_a_correctable_observation():
    body = {"error": {"message": "attempted to call tool 'repo_browser.print_tree' which was not in request.tools"}}
    client, _ = groq_with(_http_error(BadRequestError, 400, "bad", body))
    response = client.chat(MESSAGES, TOOLS)
    assert response.tool_call is None
    assert "repo_browser.print_tree" in response.content


def test_groq_retries_rate_limits_using_the_server_suggested_delay(no_sleep):
    limited = _http_error(RateLimitError, 429, "Rate limit reached. Please try again in 2.5s.")
    client, stub = groq_with(limited, limited, _groq_response("done"))
    response = client.chat(MESSAGES, TOOLS)
    assert response.tool_call["name"] == "done"
    assert no_sleep == [2.5, 2.5]
    assert len(stub.requests) == 3


def test_groq_falls_back_to_default_backoff_when_no_delay_is_given(no_sleep):
    client, _ = groq_with(_http_error(RateLimitError, 429, "slow down"), _groq_response("done"))
    client.chat(MESSAGES, TOOLS)
    assert no_sleep == [groq_client.DEFAULT_BACKOFF_SECONDS]


def test_groq_gives_up_after_max_retries():
    limited = [_http_error(RateLimitError, 429, "slow down") for _ in range(groq_client.MAX_RETRIES + 1)]
    client, stub = groq_with(*limited)
    with pytest.raises(RateLimitError):
        client.chat(MESSAGES, TOOLS)
    assert len(stub.requests) == groq_client.MAX_RETRIES + 1


# --- Gemini -----------------------------------------------------------------------------------------


def _gemini_response(function_call=None, text=None, total_tokens=33):
    return SimpleNamespace(
        function_calls=[function_call] if function_call else None,
        text=text,
        usage_metadata=SimpleNamespace(total_token_count=total_tokens),
    )


class StubGemini:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests: list[dict] = []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def gemini_with(*outcomes) -> tuple[GeminiClient, StubGemini]:
    client = GeminiClient(api_key="test")
    stub = StubGemini(*outcomes)
    client._client = stub  # type: ignore[assignment]
    return client, stub


def _client_error(code: int, message: str) -> genai_errors.ClientError:
    return genai_errors.ClientError(code, {"error": {"code": code, "message": message}})


def test_gemini_parses_a_function_call_and_token_usage():
    call = SimpleNamespace(name="read_file", args={"path": "a.py"})
    client, _ = gemini_with(_gemini_response(function_call=call, total_tokens=55))
    response = client.chat(MESSAGES, TOOLS)
    assert response.tool_call == {"name": "read_file", "args": {"path": "a.py"}}
    assert response.total_tokens == 55


def test_gemini_text_only_response():
    client, _ = gemini_with(_gemini_response(text="a plan"))
    response = client.chat(MESSAGES, TOOLS, require_tool=False)
    assert response.tool_call is None and response.content == "a plan"


def test_gemini_function_calling_mode_follows_require_tool():
    client, stub = gemini_with(_gemini_response(text="x"), _gemini_response(text="y"))
    client.chat(MESSAGES, TOOLS, require_tool=True)
    client.chat(MESSAGES, TOOLS, require_tool=False)
    modes = [r["config"].tool_config.function_calling_config.mode for r in stub.requests]
    assert [m.name for m in modes] == ["ANY", "AUTO"]


def test_gemini_converts_roles_and_system_prompt():
    client, stub = gemini_with(_gemini_response(text="x"))
    history = [*MESSAGES, {"role": "assistant", "content": "ok"}, {"role": "tool", "content": "result"}]
    client.chat(history, TOOLS)
    request = stub.requests[0]
    assert request["config"].system_instruction == "sys"
    roles = [c["role"] for c in request["contents"]]
    assert roles == ["user", "model", "user"]
    assert request["contents"][-1]["parts"][0]["text"] == "[TOOL RESULT] result"


def test_gemini_retries_per_minute_limits(no_sleep):
    limited = _client_error(429, "Quota exceeded for metric ...PerMinute... retryDelay: '7s'")
    client, stub = gemini_with(limited, _gemini_response(text="ok"))
    assert client.chat(MESSAGES, TOOLS).content == "ok"
    assert len(stub.requests) == 2
    assert no_sleep == [7.0]  # parsed from the server's retryDelay, not the default


def test_gemini_fails_fast_on_daily_quota(no_sleep):
    daily = _client_error(429, "Quota exceeded: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    client, stub = gemini_with(daily)
    with pytest.raises(genai_errors.ClientError):
        client.chat(MESSAGES, TOOLS)
    assert len(stub.requests) == 1 and no_sleep == []


def test_gemini_does_not_retry_other_client_errors(no_sleep):
    client, stub = gemini_with(_client_error(400, "bad request"))
    with pytest.raises(genai_errors.ClientError):
        client.chat(MESSAGES, TOOLS)
    assert len(stub.requests) == 1 and no_sleep == []
