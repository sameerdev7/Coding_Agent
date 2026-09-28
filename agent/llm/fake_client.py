from agent.llm.base import ChatResponse


class FakeLLMClient:
    """Returns pre-scripted responses in order, ignoring input. For tests only."""

    def __init__(self, responses: list[ChatResponse]):
        self._responses = list(responses)
        self._index = 0

    def chat(self, messages: list[dict], tools: list[dict]) -> ChatResponse:
        if self._index >= len(self._responses):
            raise IndexError("FakeLLMClient ran out of scripted responses")
        response = self._responses[self._index]
        self._index += 1
        return response
