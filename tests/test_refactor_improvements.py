"""Tests for refactor improvements: dynamic configs, audio fallback in executor, and unified planner interface."""

from pathlib import Path
import pytest

from lifemate.agent.planner import AgentPlanner, HeuristicPlanner, MockLLMClient
from lifemate.tools.catalog import ToolCatalog
from lifemate.tools.executor import InteractiveExecutor
from lifemate.tools.resolver import EntityResolver
from lifemate.utils.config import get_agent_config, get_llm_config, get_runtime_config

CATALOG_PATH = Path("data/train/train_tools_catalog.json")


def test_dynamic_configs():
    llm_cfg = get_llm_config()
    assert "provider" in llm_cfg
    assert "model" in llm_cfg
    assert "temperature" in llm_cfg

    agent_cfg = get_agent_config()
    assert agent_cfg["max_steps"] == 8
    assert agent_cfg["strict_validation"] is True

    rt_cfg = get_runtime_config()
    assert rt_cfg["max_tool_calls"] == 12


def test_planner_unified_plan_interface():
    catalog = ToolCatalog.from_file(CATALOG_PATH)
    planner = HeuristicPlanner(catalog=catalog)

    # Call with signature (transcript, context, history, tool_results)
    plan1 = planner.plan(
        transcript="Kiểm tra thời tiết ngày mai",
        context={"current_time": "2026-11-28T10:00:00+07:00", "installed_apps": {"weather": {"tools": ["weather.get_forecast"]}}},
        history=[],
        tool_results=[],
    )
    assert isinstance(plan1, dict)
    assert "tool_calls" in plan1
    assert len(plan1["tool_calls"]) == 1
    assert plan1["tool_calls"][0]["name"] == "weather.get_forecast"

    # Call with sample dict
    sample = {
        "id": "sample_test_01",
        "audio_text": "Kiểm tra thời tiết ngày mai",
        "context": {"current_time": "2026-11-28T10:00:00+07:00", "installed_apps": {"weather": {"tools": ["weather.get_forecast"]}}},
    }
    plan2 = planner.plan(sample, tool_results=[])
    assert plan2["tool_calls"][0]["name"] == "weather.get_forecast"


def test_entity_resolver_generic_id_extraction():
    # Note ID extraction
    history = [
        {"name": "notes.list", "result": {"ok": True, "notes": [{"id": "n_99", "title": "Ghi chú"}]}}
    ]
    extracted = EntityResolver.extract_id_from_tool_results("note_id", "notes.notes", history)
    assert extracted == "n_99"

    # Alarm ID extraction
    history_alarm = [
        {"name": "alarm.list", "result": {"ok": True, "alarms": [{"id": "alm_12", "time": "06:00"}]}}
    ]
    extracted_alarm = EntityResolver.extract_id_from_tool_results("alarm_id", "alarm.alarms", history_alarm)
    assert extracted_alarm == "alm_12"


def test_executor_audio_transcription_fallback():
    class FakeSpeech:
        def transcribe(self, audio_path, context=None, history=None):
            return "Kiểm tra thời tiết ngày mai"

    catalog = ToolCatalog.from_file(CATALOG_PATH)
    planner = HeuristicPlanner(catalog=catalog)
    executor = InteractiveExecutor(planner=planner, catalog=catalog, speech=FakeSpeech())

    sample = {
        "id": "test_audio_sample",
        "audio": "dummy.wav",
        # audio_text is explicitly omitted to test audio fallback
        "context": {
            "current_time": "2026-11-28T10:00:00+07:00",
            "installed_apps": {"weather": {"tools": ["weather.get_forecast"]}},
        },
    }

    class MockEnv:
        def __init__(self):
            self.calls = []

        def call(self, name, args):
            self.calls.append({"name": name, "arguments": args})
            return {"ok": True, "forecast": [{"date": "2026-11-29", "rain": False}]}

        def get_tool_calls(self):
            return self.calls

        def get_state_hash(self):
            return "dummy_hash"

    res = executor.execute_sample(sample, env_instance=MockEnv())
    assert res["transcript"] == "Kiểm tra thời tiết ngày mai"
    assert len(res["tool_calls"]) == 1
    assert res["tool_calls"][0]["name"] == "weather.get_forecast"

