"""Submission File Writer & Formatter for ZAC 2026 LifeMate (Task A17).

Ensures 100% compliant JSON Lines output conforming to competition standards:
{"id": "<sample_id>", "tool_calls": [...], "state_hash": "<sha256_hash>"}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SubmissionRecord:
    id: str
    tool_calls: list[dict[str, Any]]
    state_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool_calls": self.tool_calls,
            "state_hash": self.state_hash,
        }

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class SubmissionWriter:
    """Manages line-by-line buffered writing of submission.jsonl."""

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self.count = 0

    def __enter__(self) -> SubmissionWriter:
        self._file = open(self.output_path, "w", encoding="utf-8")
        self.count = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._file:
            self._file.close()

    def write_record(self, record: SubmissionRecord | dict[str, Any]) -> None:
        """Writes one sample record as a single JSON line."""
        if not self._file:
            raise RuntimeError("SubmissionWriter must be used within a context manager 'with'.")

        if isinstance(record, dict):
            line = json.dumps(record, ensure_ascii=False)
        else:
            line = record.to_json_line()

        self._file.write(line + "\n")
        self._file.flush()
        self.count += 1

