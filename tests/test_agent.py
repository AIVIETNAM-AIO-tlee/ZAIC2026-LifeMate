"""Tests for Agent Tool Selector and Policy (Task A03)."""

from pathlib import Path
import pytest

from lifemate.tools.catalog import ToolCatalog
from lifemate.agent.tool_selector import ToolSelector
from lifemate.agent.policy import AgentPolicy

CATALOG_PATH = Path("data/train/train_tools_catalog.json")


@pytest.fixture
def catalog():
    return ToolCatalog.from_file(CATALOG_PATH)


@pytest.fixture
def installed_apps():
    return {
        "sms": {"tools": ["sms.send", "sms.read_recent"]},
        "contacts": {"tools": ["contacts.search"]},
    }


def test_tool_selector_filter(catalog, installed_apps):
    selector = ToolSelector(catalog)
    allowed_names = selector.get_allowed_tool_names(installed_apps)

    assert "sms.send" in allowed_names
    assert "sms.read_recent" in allowed_names
    assert "contacts.search" in allowed_names
    # sys.ask_missing_info must always be included
    assert "sys.ask_missing_info" in allowed_names
    # Uninstalled app tool must not be included
    assert "weather.get_forecast" not in allowed_names

    allowed_specs = selector.get_allowed_tools(installed_apps)
    assert len(allowed_specs) == 4
    assert any(s.name == "sys.ask_missing_info" for s in allowed_specs)


def test_tool_selector_is_tool_allowed(catalog, installed_apps):
    selector = ToolSelector(catalog)
    assert selector.is_tool_allowed("sms.send", installed_apps) is True
    assert selector.is_tool_allowed("sys.ask_missing_info", installed_apps) is True
    assert selector.is_tool_allowed("calendar.create_event", installed_apps) is False


def test_policy_validate_permission(catalog, installed_apps):
    policy = AgentPolicy(catalog)
    ok, err = policy.validate_tool_permission("sms.send", installed_apps)
    assert ok is True
    assert err is None

    # sys.ask_missing_info is always allowed
    ok_sys, err_sys = policy.validate_tool_permission("sys.ask_missing_info", installed_apps)
    assert ok_sys is True
    assert err_sys is None

    # Disallowed app
    ok_bad, err_bad = policy.validate_tool_permission("smart_home.light_set", installed_apps)
    assert ok_bad is False
    assert "không thuộc app đã cài" in err_bad


def test_policy_find_missing_params(catalog):
    policy = AgentPolicy(catalog)
    # sms.send requires content and one of contact_id / phone_number
    missing = policy.find_missing_required_params("sms.send", {"contact_id": "c_01"})
    assert "content" in missing

    # Valid arguments
    missing_none = policy.find_missing_required_params("sms.send", {"contact_id": "c_01", "content": "hello"})
    assert len(missing_none) == 0


def test_policy_create_ask_missing_info(catalog):
    policy = AgentPolicy(catalog)
    call = policy.create_ask_missing_info_call("sms.send", ["content"])
    assert call["name"] == "sys.ask_missing_info"
    assert call["arguments"] == {"tool_name": "sms.send", "params": ["content"]}


def test_temporal_reasoner():
    from lifemate.agent.context import TemporalReasoner

    tr = TemporalReasoner("2026-11-28T07:10:00+07:00", "Asia/Ho_Chi_Minh")
    assert tr.today_str == "2026-11-28"
    assert tr.tomorrow_str == "2026-11-29"

    assert tr.resolve_date("Nếu ngày mai trời mưa...") == "2026-11-29"
    assert tr.resolve_date("Hôm nay làm gì") == "2026-11-28"
    assert tr.resolve_date("ngày mốt đi chơi") == "2026-11-30"

    shifted = tr.shift_iso_datetime("2026-11-29T08:00:00+07:00", hours=1)
    assert shifted == "2026-11-29T09:00:00+07:00"


def test_agent_context_memory():
    from lifemate.agent.context import AgentContext

    ctx = AgentContext(
        sample_id="test_01",
        current_time="2026-11-28T07:10:00+07:00",
        user_profile={"name": "Em Linh", "city": "Đà Nẵng"},
    )
    assert ctx.profile.name == "Em Linh"

    ctx.record_call_result(
        "weather.get_forecast",
        {"date": "2026-11-29"},
        {"ok": True, "forecast": [{"rain": True}]},
    )
    assert len(ctx.executed_calls) == 1
    assert len(ctx.tool_results) == 1
    res = ctx.get_latest_result_for("weather")
    assert res is not None
    assert res["forecast"][0]["rain"] is True


def test_out_of_scope_detection(catalog, installed_apps):
    selector = ToolSelector(catalog)

    # Chitchat
    is_oos, reason = selector.detect_out_of_scope("Xin chào bạn", installed_apps)
    assert is_oos is True
    assert "chitchat" in reason

    # Cancellation
    is_oos_c, reason_c = selector.detect_out_of_scope("À mà thôi, không cần gửi tin nữa", installed_apps)
    assert is_oos_c is True
    assert "cancelled" in reason_c

    # Uninstalled app (smart_home not in installed_apps)
    is_oos_app, reason_app = selector.detect_out_of_scope("Bật đèn phòng khách giúp tôi", installed_apps)
    assert is_oos_app is True
    assert "uninstalled_app" in reason_app


def test_planner_out_of_scope_decision(catalog, installed_apps):
    from lifemate.agent.planner import HeuristicPlanner
    from lifemate.agent.context import AgentContext

    planner = HeuristicPlanner(catalog=catalog)
    ctx = AgentContext(
        sample_id="test_oos",
        current_time="2026-11-28T07:10:00+07:00",
        installed_apps=installed_apps,
    )

    decision = planner.plan_next_step("Xin chào trợ lý", ctx)
    assert decision.action == "finish"
    assert "out-of-scope" in decision.thought.lower()


def test_state_aware_context(catalog):
    from lifemate.agent.context import AgentContext

    ctx = AgentContext(
        sample_id="test_sa",
        current_time="2026-11-28T07:10:00+07:00",
    )
    ctx.record_call_result(
        "smart_home.get_status",
        {},
        {"ok": True, "devices": [{"id": "light_kitchen", "power": "on"}]},
    )
    assert ctx.is_device_in_state("light_kitchen", "on") is True
    assert ctx.is_device_in_state("light_kitchen", "off") is False


