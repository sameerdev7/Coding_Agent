"""Append-only JSONL run log: one line per event, so a run can be audited or replayed after the fact."""

import json
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

MAX_LOGGED_RESULT_CHARS = 2000


class RunLogger:
    def __init__(self, path: Path | None):
        self._file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("a", encoding="utf-8")

    def log(self, event: str, **fields: Any) -> None:
        if self._file is None:
            return
        if isinstance(fields.get("result"), str):
            fields["result"] = fields["result"][:MAX_LOGGED_RESULT_CHARS]
        record = {"ts": datetime.now(UTC).isoformat(), "event": event, **fields}
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
