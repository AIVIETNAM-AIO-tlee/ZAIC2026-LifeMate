"""Tests for Tool Catalog Parser and Entity Resolver (Task A02)."""

import pytest
from pathlib import Path

from lifemate.tools.catalog import ToolCatalog, ToolSpec
from lifemate.tools.resolver import (
    EntityResolver,
    fuzzy_similarity,
    normalize_text,
    remove_accents,
)

CATALOG_PATH = Path("data/train/train_tools_catalog.json")


def test_catalog_load_and_indexing():
    catalog = ToolCatalog.from_file(CATALOG_PATH)
    assert len(catalog.tools) > 40
    assert catalog.has_tool("sms.send")
    assert catalog.has_tool("weather.get_forecast")
    assert catalog.has_tool("sys.ask_missing_info")

    sms_send = catalog.get_tool("sms.send")
    assert isinstance(sms_send, ToolSpec)
    assert sms_send.is_write
    assert "content" in sms_send.required
    assert sms_send.x_one_of == [["contact_id"], ["phone_number"]]


def test_tool_param_specs_and_x_ref():
    catalog = ToolCatalog.from_file(CATALOG_PATH)
    light_set = catalog.get_tool("smart_home.light_set")
    assert light_set is not None
    assert "device_id" in light_set.required
    assert light_set.param_specs["device_id"].x_ref == "smart_home.devices"


def test_catalog_filter_and_openai_export():
    catalog = ToolCatalog.from_file(CATALOG_PATH)
    filtered = catalog.filter_by_names(["sms.send", "weather.get_forecast"])
    assert len(filtered) == 2

    openai_tools = catalog.to_openai_tools(["sms.send"])
    assert len(openai_tools) == 1
    assert openai_tools[0]["type"] == "function"
    assert openai_tools[0]["function"]["name"] == "sms.send"
    assert "content" in openai_tools[0]["function"]["parameters"]["properties"]


def test_remove_accents_and_normalize():
    assert remove_accents("Nguyễn Văn Linh") == "Nguyen Van Linh"
    assert remove_accents("Đà Nẵng") == "Da Nang"
    assert normalize_text("  Sếp Dũng!! ") == "sếp dũng"


def test_fuzzy_similarity():
    assert fuzzy_similarity("Sếp Dũng", "Sếp Dũng") == 1.0
    assert fuzzy_similarity("Sep Dung", "Sếp Dũng") >= 0.95
    assert fuzzy_similarity("Dũng", "Sếp Dũng") >= 0.85
    assert fuzzy_similarity("Hoàn toàn khác", "Sếp Dũng") < 0.4


def test_entity_resolver_contact():
    contacts = [
        {"id": "c_01", "name": "Nguyễn Văn A", "phone": "0912345678", "relation": "bạn"},
        {"id": "c_08", "name": "Sếp Dũng", "phone": "09925652941", "relation": "sếp"},
    ]
    res = EntityResolver.find_best_contact("Sếp Dũng", contacts)
    assert res is not None
    assert res["id"] == "c_08"

    res_rel = EntityResolver.find_best_contact("sếp", contacts)
    assert res_rel is not None
    assert res_rel["id"] == "c_08"

    res_phone = EntityResolver.find_best_contact("0912345678", contacts)
    assert res_phone is not None
    assert res_phone["id"] == "c_01"


def test_entity_resolver_event():
    events = [
        {"id": "evt_01", "title": "chạy bộ", "start": "2026-11-29T08:00:00+07:00"},
        {"id": "evt_02", "title": "đá bóng", "start": "2026-11-29T14:00:00+07:00"},
    ]
    res = EntityResolver.find_best_event("chạy bộ", events, "2026-11-29")
    assert res is not None
    assert res["id"] == "evt_01"


def test_extract_id_from_tool_results():
    history = [
        {
            "name": "contacts.search",
            "arguments": {"query": "Sếp Dũng"},
            "result": {
                "ok": True,
                "contacts": [{"id": "c_08", "name": "Sếp Dũng", "phone": "09925652941"}],
            },
        }
    ]
    cid = EntityResolver.extract_id_from_tool_results("contact_id", "contacts.contacts", history)
    assert cid == "c_08"


def test_tool_validator():
    from lifemate.tools.validator import ToolValidator

    catalog = ToolCatalog.from_file(CATALOG_PATH)
    validator = ToolValidator(catalog)

    # Valid call
    v_ok = validator.validate_call("weather.get_forecast", {"date": "2026-11-29"})
    assert v_ok.ok is True
    assert v_ok.error is None

    # Missing required parameter
    v_missing = validator.validate_call("weather.get_forecast", {})
    assert v_missing.ok is False
    assert "thiếu tham số bắt buộc" in v_missing.error

    # Wrong date format
    v_bad_date = validator.validate_call("fitness.get_activity", {"date": "29-11-2026"})
    assert v_bad_date.ok is False
    assert "phải có dạng date" in v_bad_date.error

    # Missing x-one-of (sms.send needs either contact_id or phone_number)
    v_one_of = validator.validate_call("sms.send", {"content": "hello"})
    assert v_one_of.ok is False
    assert "cần một trong" in v_one_of.error

    # Auto repair string to int
    repaired = validator.auto_repair_arguments(
        "smart_home.get_status",
        {"floor": "2", "room": "living"},
    )
    assert repaired["floor"] == 2

