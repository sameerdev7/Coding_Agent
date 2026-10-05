import json
from pathlib import Path

from agent.runlog import MAX_LOGGED_RESULT_CHARS, RunLogger


def test_writes_one_json_object_per_line(tmp_path: Path):
    path = tmp_path / "logs" / "run.jsonl"
    with RunLogger(path) as logger:
        logger.log("run_start", task="t")
        logger.log("step", step=1, tool="list_dir", args={"path": "."})

    lines = path.read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["event"] for r in records] == ["run_start", "step"]
    assert records[1]["args"] == {"path": "."}
    assert all("ts" in r for r in records)


def test_appends_across_sessions(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    for _ in range(2):
        with RunLogger(path) as logger:
            logger.log("run_start")
    assert len(path.read_text().splitlines()) == 2


def test_long_results_are_truncated(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    with RunLogger(path) as logger:
        logger.log("step", result="x" * 100_000)
    assert len(json.loads(path.read_text())["result"]) == MAX_LOGGED_RESULT_CHARS


def test_disabled_logger_is_a_noop():
    with RunLogger(None) as logger:
        logger.log("anything", a=1)
