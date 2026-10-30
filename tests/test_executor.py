"""Tests for Interactive Tool Executor (Tasks A08, A09)."""

from pathlib import Path
import pytest

from lifemate.agent.planner import AgentPlanner, MockLLMClient
from lifemate.tools.catalog import ToolCatalog
from lifemate.tools.executor import InteractiveExecutor, get_tool_env

CATALOG_PATH = Path("data/train/train_tools_catalog.json")


@pytest.fixture
def catalog():
    return ToolCatalog.from_file(CATALOG_PATH)


def test_interactive_executor_single_step(catalog):
    # Mock LLM calls weather.get_forecast then finishes
    mock_llm = MockLLMClient()
    mock_llm.add_response({
        "thought": "Xem thời tiết ngày mai",
        "action": "call",
        "tool_name": "weather.get_forecast",
        "arguments": {"date": "2026-11-29"},
    })
    mock_llm.add_response({"action": "finish"})

    planner = AgentPlanner(catalog=catalog, llm_client=mock_llm)
    executor = InteractiveExecutor(planner=planner, catalog=catalog, max_steps=4)

    sample = {
        "id": "train_0001",
        "audio_text": "Xem thời tiết ngày mai",
        "context": {
            "current_time": "2026-11-28T07:10:00+07:00",
            "timezone": "Asia/Ho_Chi_Minh",
            "installed_apps": {"weather": {"tools": ["weather.get_forecast"]}},
        },
    }

    env = get_tool_env("train", "train_0001", root="data")
    result = executor.execute_sample(sample, env_instance=env)

    assert result["id"] == "train_0001"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["name"] == "weather.get_forecast"
    assert result["tool_calls"][0]["arguments"] == {"date": "2026-11-29"}
    assert isinstance(result["state_hash"], str)


def test_interactive_executor_self_correction_on_bad_schema(catalog):
    """Verifies that invalid schema is intercepted before calling env.call, preventing penalty!"""
    mock_llm = MockLLMClient()
    # Step 1: LLM makes an invalid call missing 'date'
    mock_llm.add_response({
        "thought": "Gọi thời tiết nhưng quên date",
        "action": "call",
        "tool_name": "weather.get_forecast",
        "arguments": {},
    })
    # Step 2: In response to validation error feedback, LLM self-corrects with valid 'date'
    mock_llm.add_response({
        "thought": "Sửa lại tham số date",
        "action": "call",
        "tool_name": "weather.get_forecast",
        "arguments": {"date": "2026-11-29"},
    })
    mock_llm.add_response({"action": "finish"})

    planner = AgentPlanner(catalog=catalog, llm_client=mock_llm)
    executor = InteractiveExecutor(
        planner=planner,
        catalog=catalog,
        max_steps=5,
        max_schema_retries=2,
    )

    sample = {
        "id": "train_0001",
        "audio_text": "Xem thời tiết ngày mai",
        "context": {
            "current_time": "2026-11-28T07:10:00+07:00",
            "timezone": "Asia/Ho_Chi_Minh",
            "installed_apps": {"weather": {"tools": ["weather.get_forecast"]}},
        },
    }

    env = get_tool_env("train", "train_0001", root="data")
    result = executor.execute_sample(sample, env_instance=env)

    # env.get_tool_calls() MUST only have 1 call (the valid one) — ZERO rejected calls!
    calls = env.get_tool_calls()
    assert len(calls) == 1
    assert calls[0]["arguments"] == {"date": "2026-11-29"}


def test_interactive_executor_max_steps_limit(catalog):
    mock_llm = MockLLMClient()
    # Infinite calls
    for _ in range(10):
        mock_llm.add_response({
            "action": "call",
            "tool_name": "smart_home.get_status",
            "arguments": {},
        })

    planner = AgentPlanner(catalog=catalog, llm_client=mock_llm)
    executor = InteractiveExecutor(planner=planner, catalog=catalog, max_steps=3)

    sample = {
        "id": "train_0001",
        "audio_text": "Kiểm tra nhà",
        "context": {
            "current_time": "2026-11-28T07:10:00+07:00",
            "installed_apps": {"smart_home": {"tools": ["smart_home.get_status"]}},
        },
    }

    env = get_tool_env("train", "train_0001", root="data")
    result = executor.execute_sample(sample, env_instance=env)
    assert len(result["tool_calls"]) == 3

