"""Per-example and group-level ASR error analysis."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
import re
from typing import Any

from lifemate.evaluation.asr_metrics import compute_asr_metrics
from lifemate.speech.normalizer import normalize_text


def _word_errors(reference: str, hypothesis: str) -> list[dict[str, str]]:
    ref, hyp = reference.split(), hypothesis.split()
    rows = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        rows[i][0] = i
    for j in range(len(hyp) + 1):
        rows[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            rows[i][j] = min(
                rows[i - 1][j] + 1,
                rows[i][j - 1] + 1,
                rows[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
            )
    i, j = len(ref), len(hyp)
    errors = []
    while i or j:
        if i and j and ref[i - 1] == hyp[j - 1] and rows[i][j] == rows[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i and j and rows[i][j] == rows[i - 1][j - 1] + 1:
            errors.append({"type": "substitution", "reference": ref[i - 1], "hypothesis": hyp[j - 1]})
            i, j = i - 1, j - 1
        elif i and rows[i][j] == rows[i - 1][j] + 1:
            errors.append({"type": "deletion", "reference": ref[i - 1], "hypothesis": ""})
            i -= 1
        else:
            errors.append({"type": "insertion", "reference": "", "hypothesis": hyp[j - 1]})
            j -= 1
    errors.reverse()
    return errors


def analyze_errors(
    examples: Iterable[Mapping[str, Any]],
    group_key: str = "group",
) -> dict[str, Any]:
    rows = list(examples)
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    error_types: Counter[str] = Counter()
    details = []
    for row in rows:
        reference = str(row.get("reference", ""))
        hypothesis = str(row.get("hypothesis", ""))
        group = str(row.get(group_key) or "unknown")
        by_group[group].append(row)
        normalized_reference = re.sub(r"[^\w\s]", " ", normalize_text(reference).casefold())
        normalized_hypothesis = re.sub(r"[^\w\s]", " ", normalize_text(hypothesis).casefold())
        errors = _word_errors(normalized_reference, normalized_hypothesis)
        error_types.update(item["type"] for item in errors)
        details.append(
            {
                "id": row.get("id"),
                "group": group,
                "reference": reference,
                "hypothesis": hypothesis,
                "word_errors": errors,
            }
        )
    grouped = {
        group: compute_asr_metrics(
            [str(row.get("reference", "")) for row in members],
            [str(row.get("hypothesis", "")) for row in members],
        )
        for group, members in sorted(by_group.items())
    }
    return {"groups": grouped, "error_types": dict(error_types), "examples": details}