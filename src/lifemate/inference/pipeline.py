"""End-to-End Inference Pipeline for ZAC 2026 LifeMate (Task A17).

Coordinates reading sample data, executing the interactive agent loop,
recovering from unexpected errors, and streaming records to submission.jsonl.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from lifemate.agent.planner import AgentPlanner
from lifemate.inference.submission import SubmissionRecord, SubmissionWriter
from lifemate.tools.catalog import ToolCatalog
from lifemate.tools.executor import InteractiveExecutor, get_tool_env
from lifemate.utils.logging import setup_logger
from lifemate.utils.timing import Timer


class InferencePipeline:
    """End-to-end streaming inference pipeline."""

    def __init__(
        self,
        catalog: ToolCatalog,
        planner: AgentPlanner,
        max_steps: int = 8,
        logger: logging.Logger | None = None,
    ):
        self.catalog = catalog
        self.planner = planner
        self.executor = InteractiveExecutor(planner=planner, catalog=catalog, max_steps=max_steps)
        self.logger = logger or setup_logger("inference_pipeline")

    def run(
        self,
        data_path: str | Path,
        output_path: str | Path = "result/submission.jsonl",
        split: str = "public_test",
        root: str | Path = "data",
        limit: int = 0,
    ) -> int:
        """Executes inference on all items in data_path and writes output_path."""
        data_file = Path(data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"Input data not found: {data_file}")

        self.logger.info(f"Starting inference pipeline on {data_file} (split={split})...")

        with SubmissionWriter(output_path) as writer:
            with open(data_file, "r", encoding="utf-8") as f:
                with Timer("total_pipeline") as timer:
                    for line_idx, line in enumerate(f, 1):
                        if not line.strip():
                            continue

                        sample = json.loads(line)
                        sample_id = sample.get("id", f"sample_{line_idx}")

                        try:
                            # Run interactive execution on sample
                            res = self.executor.execute_sample(sample, split=split, root=root)
                            record = SubmissionRecord(
                                id=sample_id,
                                tool_calls=res.get("tool_calls", []),
                                state_hash=res.get("state_hash", ""),
                            )
                        except Exception as e:
                            # Fault tolerance: catch any unexpected exception and fall back to safe empty calls
                            self.logger.error(f"Error on sample {sample_id}: {e}. Falling back to empty calls.")
                            try:
                                fallback_env = get_tool_env(split, sample_id, root=root)
                                fallback_hash = fallback_env.get_state_hash()
                            except Exception:
                                fallback_hash = "0" * 64
                            record = SubmissionRecord(id=sample_id, tool_calls=[], state_hash=fallback_hash)

                        writer.write_record(record)

                        if line_idx % 50 == 0:
                            self.logger.info(f"Processed {line_idx} samples...")

                        if limit and line_idx >= limit:
                            break

        total_time = timer.elapsed
        throughput = writer.count / total_time if total_time > 0 else 0
        self.logger.info(
            f"Inference completed: {writer.count} records written to {output_path} "
            f"in {total_time:.2f}s ({throughput:.2f} samples/sec)."
        )
        return writer.count

