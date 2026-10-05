import json

DEFAULT_MAX_OUTPUT_CHARS = 6000
DEFAULT_KEEP_RECENT = 6
DEFAULT_ELIDE_CHARS = 300


def truncate_output(text: str, max_chars: int = DEFAULT_MAX_OUTPUT_CHARS) -> str:
    """Cap a tool result, keeping head and tail (exit codes live at the head, pytest summaries at the tail)."""
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    omitted = len(text) - max_chars
    return f"{text[:head]}\n[... {omitted} characters truncated ...]\n{text[-tail:]}"


def _elide(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [elided {len(text) - limit} chars]"


def _elide_arguments(arguments: str, limit: int) -> str:
    """Shrink long string values inside a tool call's JSON arguments, keeping the JSON valid."""
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        return arguments
    if not isinstance(parsed, dict):
        return arguments
    shrunk = {k: _elide(v, limit) if isinstance(v, str) else v for k, v in parsed.items()}
    return json.dumps(shrunk)


def compact_messages(
    messages: list[dict],
    keep_recent: int = DEFAULT_KEEP_RECENT,
    elide_chars: int = DEFAULT_ELIDE_CHARS,
) -> list[dict]:
    """Return a copy of `messages` with older tool exchanges shrunk, for sending to the LLM.

    The most recent `keep_recent` tool exchanges stay verbatim. Older tool results and the large
    arguments of the matching assistant tool calls (e.g. whole-file writes) are elided. Every
    assistant `tool_calls` entry keeps its matching `tool` message, so the wire format stays valid.
    """
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    old_tool_indices = set(tool_indices[:-keep_recent]) if keep_recent > 0 else set(tool_indices)
    old_call_ids = {messages[i].get("tool_call_id") for i in old_tool_indices}

    compacted: list[dict] = []
    for i, message in enumerate(messages):
        if i in old_tool_indices:
            compacted.append({**message, "content": _elide(str(message.get("content", "")), elide_chars)})
        elif message.get("role") == "assistant" and message.get("tool_calls"):
            calls = message["tool_calls"]
            if any(call.get("id") in old_call_ids for call in calls):
                calls = [
                    {
                        **call,
                        "function": {
                            **call["function"],
                            "arguments": _elide_arguments(call["function"].get("arguments", ""), elide_chars),
                        },
                    }
                    for call in calls
                ]
                compacted.append({**message, "tool_calls": calls})
            else:
                compacted.append(message)
        else:
            compacted.append(message)
    return compacted
