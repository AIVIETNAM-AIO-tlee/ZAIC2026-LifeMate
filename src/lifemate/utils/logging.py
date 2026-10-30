"""Structured Logging & Execution Tracer for ZAC 2026 LifeMate Agent.

Logs step-by-step agent interactions (prompt, raw LLM response, parsed decision,
validation status, tool results) in JSON Lines for debugging and error tracing.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any


def setup_logger(name: str = "lifemate", level: int = logging.INFO) -> logging.Logger:
    """Configures standard console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class StructuredTracer:
    """Appends structured JSON Lines trace events for agent debugging."""

    def __init__(self, trace_file: str | Path | None = None):
        self.trace_file = Path(trace_file) if trace_file else None
        if self.trace_file:
            self.trace_file.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Writes an event dict to trace log file."""
        if not self.trace_file:
            return
        payload = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            **data,
        }
        with open(self.trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def trace_step(
        self,
        sample_id: str,
        step: int,
        action: str,
        tool_name: str | None,
        arguments: dict[str, Any] | None,
        validation_ok: bool,
        validation_error: str | None = None,
        tool_result: dict[str, Any] | None = None,
    ) -> None:
        """Helper to trace one step in the agent interactive loop."""
        self.log_event(
            event_type="agent_step",
            data={
                "sample_id": sample_id,
                "step": step,
                "action": action,
                "tool_name": tool_name,
                "arguments": arguments,
                "validation_ok": validation_ok,
                "validation_error": validation_error,
                "tool_result": tool_result,
            },
        )

