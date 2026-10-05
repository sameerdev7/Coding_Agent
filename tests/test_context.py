import json

from agent.context import compact_messages, truncate_output


def test_truncate_output_leaves_short_text_alone():
    assert truncate_output("short", max_chars=100) == "short"


def test_truncate_output_keeps_head_and_tail():
    text = "HEAD" + "m" * 10_000 + "TAIL"
    out = truncate_output(text, max_chars=200)
    assert out.startswith("HEAD")
    assert out.endswith("TAIL")
    assert "characters truncated" in out
    assert len(out) < 300


def _exchange(i: int, big: str = "z" * 1000) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"c{i}",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": json.dumps({"path": "a.py", "content": big})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": f"c{i}", "content": big},
    ]


def _history(n: int) -> list[dict]:
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    for i in range(n):
        messages += _exchange(i)
    return messages


def test_compact_keeps_recent_exchanges_verbatim():
    messages = _history(10)
    compacted = compact_messages(messages, keep_recent=3, elide_chars=50)
    assert compacted[-6:] == messages[-6:]


def test_compact_elides_old_tool_results_and_call_arguments():
    compacted = compact_messages(_history(10), keep_recent=3, elide_chars=50)
    old_tool = compacted[3]
    old_call = compacted[2]
    assert "elided" in old_tool["content"] and len(old_tool["content"]) < 120
    arguments = json.loads(old_call["tool_calls"][0]["function"]["arguments"])  # still valid JSON
    assert arguments["path"] == "a.py"
    assert "elided" in arguments["content"]


def test_compact_preserves_tool_call_pairing():
    compacted = compact_messages(_history(10), keep_recent=3, elide_chars=50)
    call_ids = [c["id"] for m in compacted for c in m.get("tool_calls", [])]
    tool_ids = [m["tool_call_id"] for m in compacted if m["role"] == "tool"]
    assert call_ids == tool_ids


def test_compact_does_not_mutate_input():
    messages = _history(10)
    snapshot = json.dumps(messages)
    compact_messages(messages, keep_recent=2, elide_chars=10)
    assert json.dumps(messages) == snapshot


def test_compact_leaves_system_and_user_messages_alone():
    compacted = compact_messages(_history(10), keep_recent=1, elide_chars=10)
    assert compacted[0] == {"role": "system", "content": "sys"}
    assert compacted[1] == {"role": "user", "content": "task"}
