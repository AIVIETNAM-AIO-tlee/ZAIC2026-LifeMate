"""Unit & Integration Tests for Submission Pipeline (Task A17)."""

import json
from pathlib import Path
import pytest

from lifemate.agent.planner import HeuristicPlanner
from lifemate.inference.pipeline import InferencePipeline
from lifemate.inference.submission import SubmissionRecord, SubmissionWriter
from lifemate.tools.catalog import ToolCatalog

CATALOG_PATH = Path("data/train/train_tools_catalog.json")
TRAIN_DATA_PATH = Path("data/train/train.jsonl")


def test_submission_record():
    rec = SubmissionRecord(
        id="sample_01",
        tool_calls=[{"name": "weather.get_forecast", "arguments": {"date": "2026-11-29"}}],
        state_hash="a" * 64,
    )
    line = rec.to_json_line()
    parsed = json.loads(line)
    assert parsed["id"] == "sample_01"
    assert len(parsed["tool_calls"]) == 1
    assert parsed["state_hash"] == "a" * 64


def test_submission_writer(tmp_path):
    out_file = tmp_path / "test_submission.jsonl"
    with SubmissionWriter(out_file) as writer:
        writer.write_record(SubmissionRecord("s1", [], "hash1"))
        writer.write_record({"id": "s2", "tool_calls": [], "state_hash": "hash2"})

    assert writer.count == 2
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 2
    assert lines[0]["id"] == "s1"
    assert lines[1]["id"] == "s2"


def test_inference_pipeline_e2e(tmp_path):
    catalog = ToolCatalog.from_file(CATALOG_PATH)
    planner = HeuristicPlanner(catalog=catalog)
    pipeline = InferencePipeline(catalog=catalog, planner=planner)

    out_file = tmp_path / "submission_pipeline_test.jsonl"
    count = pipeline.run(
        data_path=TRAIN_DATA_PATH,
        output_path=out_file,
        split="train",
        root="data",
        limit=3,
    )

    assert count == 3
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 3
    for r in records:
        assert "id" in r
        assert "tool_calls" in r
        assert "state_hash" in r
        assert len(r["state_hash"]) == 64

