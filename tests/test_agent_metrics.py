"""Unit Tests for Agent Metrics & Error Analysis (Tasks A14, A15)."""

import pytest
from lifemate.evaluation.agent_metrics import (
    CallMatchResult,
    SampleEvaluationResult,
    aggregate_dataset_metrics,
    compare_arguments,
    compute_sequence_exact_match,
    compute_tool_calls_metrics,
)
from lifemate.evaluation.error_analysis import ErrorAnalyzer, ErrorType


def test_compare_arguments():
    # Exact match
    assert compare_arguments({"a": 1, "b": "hello"}, {"a": 1, "b": "hello"}) is True

    # Missing / extra key
    assert compare_arguments({"a": 1}, {"a": 1, "b": 2}) is False

    # Fuzzy match on Vietnamese text
    assert compare_arguments(
        {"content": "mai chạy bộ dời sang 09:00 nhé"},
        {"content": "mai chay bo doi sang 09:00 nhe"},
        fuzzy_threshold=0.8,
    ) is True


def test_compute_tool_calls_metrics_both_empty():
    res = compute_tool_calls_metrics([], [])
    assert res.f1 == 1.0
    assert res.precision == 1.0
    assert res.recall == 1.0


def test_compute_tool_calls_metrics_one_empty():
    res_pred_empty = compute_tool_calls_metrics([], [{"name": "weather.get_forecast"}])
    assert res_pred_empty.f1 == 0.0
    assert res_pred_empty.fn == 1

    res_gold_empty = compute_tool_calls_metrics([{"name": "weather.get_forecast"}], [])
    assert res_gold_empty.f1 == 0.0
    assert res_gold_empty.fp == 1


def test_compute_tool_calls_metrics_match():
    pred = [
        {"name": "contacts.search", "arguments": {"query": "Sếp Dũng"}},
        {"name": "sms.send", "arguments": {"contact_id": "c_08", "content": "alo"}},
    ]
    gold = [
        {"name": "contacts.search", "arguments": {"query": "Sếp Dũng"}},
        {"name": "sms.send", "arguments": {"contact_id": "c_08", "content": "alo"}},
    ]
    res = compute_tool_calls_metrics(pred, gold)
    assert res.f1 == 1.0
    assert res.tp == 2

    exact = compute_sequence_exact_match(pred, gold)
    assert exact is True


def test_error_analyzer_classification():
    # 1. Wrong tool name
    res1 = SampleEvaluationResult(
        sample_id="s1",
        precision=0.0,
        recall=0.0,
        f1=0.0,
        sequence_exact_match=False,
        state_hash_match=False,
        task_success=False,
        rejected_calls_count=0,
        pred_calls=[{"name": "calendar.list_events", "arguments": {}}],
        gold_calls=[{"name": "weather.get_forecast", "arguments": {"date": "2026-11-29"}}],
        pred_hash="h1",
        gold_hash="h2",
    )
    errs1 = ErrorAnalyzer.classify_sample_errors(res1)
    assert ErrorType.WRONG_TOOL_NAME in errs1
    assert ErrorType.STATE_HASH_MISMATCH in errs1

    # 2. Missing info failure
    res2 = SampleEvaluationResult(
        sample_id="s2",
        precision=0.0,
        recall=0.0,
        f1=0.0,
        sequence_exact_match=False,
        state_hash_match=True,
        task_success=False,
        rejected_calls_count=0,
        pred_calls=[],
        gold_calls=[{"name": "sys.ask_missing_info", "arguments": {"tool_name": "sms.send", "params": ["content"]}}],
        pred_hash="h1",
        gold_hash="h1",
    )
    errs2 = ErrorAnalyzer.classify_sample_errors(res2)
    assert ErrorType.MISSING_INFO_FAIL in errs2


def test_aggregate_dataset_metrics():
    r1 = SampleEvaluationResult(
        sample_id="s1",
        precision=1.0,
        recall=1.0,
        f1=1.0,
        sequence_exact_match=True,
        state_hash_match=True,
        task_success=True,
        rejected_calls_count=0,
        pred_calls=[],
        gold_calls=[],
        pred_hash="h",
        gold_hash="h",
    )
    summary = aggregate_dataset_metrics([r1])
    assert summary.total_samples == 1
    assert summary.task_success_rate == 1.0
    assert summary.mean_f1 == 1.0

