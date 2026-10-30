"""End-to-End Multi-Step Integration Tests (Tasks A06, A07, A09, A10)."""

import json
from pathlib import Path
import pytest

from lifemate.agent.planner import AgentPlanner, MockLLMClient
from lifemate.tools.catalog import ToolCatalog
from lifemate.tools.executor import InteractiveExecutor, get_tool_env

CATALOG_PATH = Path("data/train/train_tools_catalog.json")
TRAIN_DATA_PATH = Path("data/train/train.jsonl")


@pytest.fixture
def catalog():
    return ToolCatalog.from_file(CATALOG_PATH)


@pytest.fixture
def sample_train_0001():
    with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["id"] == "train_0001":
                return item
    raise ValueError("Sample train_0001 not found")


def test_multistep_workflow_e2e(catalog, sample_train_0001):
    """Replicates the full multi-step conditional scenario of train_0001:
    1. weather.get_forecast
    2. calendar.list_events
    3. calendar.update_event
    4. contacts.search
    5. sms.send
    6. finish
    """
    mock_llm = MockLLMClient()

    # Step 1: Weather
    mock_llm.add_response({
        "thought": "Kiểm tra dự báo thời tiết ngày mai 2026-11-29",
        "action": "call",
        "tool_name": "weather.get_forecast",
        "arguments": {"date": "2026-11-29"},
    })

    # Step 2: Calendar List
    mock_llm.add_response({
        "thought": "Có mưa -> Tra cứu lịch sự kiện ngày mai",
        "action": "call",
        "tool_name": "calendar.list_events",
        "arguments": {},
    })

    # Step 3: Calendar Update
    mock_llm.add_response({
        "thought": "Dời chạy bộ evt_01 sang 09:00",
        "action": "call",
        "tool_name": "calendar.update_event",
        "arguments": {
            "event_id": "evt_01",
            "start": "2026-11-29T09:00:00+07:00",
            "end": "2026-11-29T10:00:00+07:00",
        },
    })

    # Step 4: Contacts Search
    mock_llm.add_response({
        "thought": "Tìm thông tin liên hệ của Sếp Dũng",
        "action": "call",
        "tool_name": "contacts.search",
        "arguments": {"query": "Sếp Dũng"},
    })

    # Step 5: SMS Send (Argument grounding can supply contact_id 'c_08')
    mock_llm.add_response({
        "thought": "Gửi SMS báo giờ mới cho Sếp Dũng",
        "action": "call",
        "tool_name": "sms.send",
        "arguments": {
            "contact_id": "c_08",
            "content": "mai chạy bộ dời sang 09:00 nhé",
        },
    })

    # Step 6: Finish
    mock_llm.add_response({"thought": "Hoàn thành toàn bộ kế hoạch", "action": "finish"})

    planner = AgentPlanner(catalog=catalog, llm_client=mock_llm)
    executor = InteractiveExecutor(planner=planner, catalog=catalog, max_steps=8)

    env = get_tool_env("train", "train_0001", root="data")
    result = executor.execute_sample(sample_train_0001, env_instance=env)

    tool_calls = result["tool_calls"]
    assert len(tool_calls) == 5
    assert [c["name"] for c in tool_calls] == [
        "weather.get_forecast",
        "calendar.list_events",
        "calendar.update_event",
        "contacts.search",
        "sms.send",
    ]

    # Verify that the final state matches label.final_state
    current_state = env.get_state()
    gold_final = sample_train_0001["label"]["final_state"]
    for app_name, app_state in gold_final.items():
        assert current_state.get(app_name) == app_state, f"State mismatch in app {app_name}"


def test_multistep_missing_info_e2e(catalog):
    """Tests an end-to-end missing info request where sys.ask_missing_info is called."""
    mock_llm = MockLLMClient()
    mock_llm.add_response({
        "thought": "Yêu cầu gửi SMS nhưng thiếu nội dung",
        "action": "call",
        "tool_name": "sys.ask_missing_info",
        "arguments": {
            "tool_name": "sms.send",
            "params": ["content"],
        },
    })
    mock_llm.add_response({"action": "finish"})

    planner = AgentPlanner(catalog=catalog, llm_client=mock_llm)
    executor = InteractiveExecutor(planner=planner, catalog=catalog)

    sample = {
        "id": "train_0003",
        "audio_text": "Gửi tin nhắn cho Lan",
        "context": {
            "current_time": "2026-11-20T10:00:00+07:00",
            "installed_apps": {
                "sms": {"tools": ["sms.send"]},
                "contacts": {"tools": ["contacts.search"]},
            },
        },
    }

    env = get_tool_env("train", "train_0003", root="data")
    result = executor.execute_sample(sample, env_instance=env)

    calls = result["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["name"] == "sys.ask_missing_info"
    assert calls[0]["arguments"]["tool_name"] == "sms.send"
    assert "content" in calls[0]["arguments"]["params"]

