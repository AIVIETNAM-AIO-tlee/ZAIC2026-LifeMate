"""Small reusable metrics for end-to-end tool-call evaluation."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _call_key(call: Mapping[str, Any]) -> tuple[str, str]:
    import json

    return (
        str(call.get("name", "")),
        json.dumps(call.get("arguments", {}), sort_keys=True, ensure_ascii=False, separators=(",", ":")),
    )


def tool_call_metrics(
    predicted: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
) -> dict[str, float | int | bool]:
    predicted_keys = [_call_key(call) for call in predicted]
    expected_keys = [_call_key(call) for call in expected]
    remaining = list(expected_keys)
    matched = 0
    for key in predicted_keys:
        if key in remaining:
            remaining.remove(key)
            matched += 1
    precision = matched / len(predicted_keys) if predicted_keys else float(not expected_keys)
    recall = matched / len(expected_keys) if expected_keys else float(not predicted_keys)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "predicted_calls": len(predicted_keys),
        "expected_calls": len(expected_keys),
        "matched_calls": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": predicted_keys == expected_keys,
    }