"""Comprehensive Evaluation Metrics for ZAC 2026 LifeMate Agent (Task A14).

Computes Precision, Recall, F1 on tool calls, Sequence Exact Match,
State Hash Accuracy, App-level State Mutations, and Task Success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from lifemate.tools.resolver import fuzzy_similarity


@dataclass
class CallMatchResult:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def compare_arguments(pred_args: dict[str, Any], gold_args: dict[str, Any], fuzzy_threshold: float = 0.85) -> bool:
    """Compares two argument dicts, with fuzzy string matching for free-text parameters."""
    if pred_args == gold_args:
        return True

    if set(pred_args.keys()) != set(gold_args.keys()):
        return False

    for k, gold_val in gold_args.items():
        pred_val = pred_args.get(k)
        if pred_val == gold_val:
            continue

        # If both are strings, allow fuzzy matching for free-text fields
        if isinstance(pred_val, str) and isinstance(gold_val, str):
            sim = fuzzy_similarity(pred_val, gold_val)
            if sim >= fuzzy_threshold:
                continue
            return False

        return False

    return True


def compute_tool_calls_metrics(
    pred_calls: list[dict[str, Any]],
    gold_calls: list[dict[str, Any]],
    fuzzy_args: bool = True,
) -> CallMatchResult:
    """Computes Precision, Recall, and F1 over tool calls (name and arguments)."""
    if not pred_calls and not gold_calls:
        return CallMatchResult(precision=1.0, recall=1.0, f1=1.0, tp=0, fp=0, fn=0)
    if not pred_calls and gold_calls:
        return CallMatchResult(precision=0.0, recall=0.0, f1=0.0, tp=0, fp=0, fn=len(gold_calls))
    if pred_calls and not gold_calls:
        return CallMatchResult(precision=0.0, recall=0.0, f1=0.0, tp=0, fp=len(pred_calls), fn=0)

    gold_matched = [False] * len(gold_calls)
    tp = 0

    for p in pred_calls:
        p_name = p.get("name")
        p_args = p.get("arguments") or {}
        matched = False
        for idx, g in enumerate(gold_calls):
            if not gold_matched[idx]:
                g_name = g.get("name")
                g_args = g.get("arguments") or {}
                if p_name == g_name:
                    if fuzzy_args:
                        arg_match = compare_arguments(p_args, g_args)
                    else:
                        arg_match = (p_args == g_args)

                    if arg_match:
                        gold_matched[idx] = True
                        tp += 1
                        matched = True
                        break

    fp = len(pred_calls) - tp
    fn = len(gold_calls) - tp
    precision = tp / len(pred_calls) if pred_calls else 0.0
    recall = tp / len(gold_calls) if gold_calls else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return CallMatchResult(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)


def compute_sequence_exact_match(pred_calls: list[dict[str, Any]], gold_calls: list[dict[str, Any]]) -> bool:
    """Checks if the predicted tool call sequence matches gold calls in exact order and arguments."""
    if len(pred_calls) != len(gold_calls):
        return False
    for p, g in zip(pred_calls, gold_calls):
        if p.get("name") != g.get("name"):
            return False
        if not compare_arguments(p.get("arguments") or {}, g.get("arguments") or {}):
            return False
    return True


@dataclass
class SampleEvaluationResult:
    sample_id: str
    precision: float
    recall: float
    f1: float
    sequence_exact_match: bool
    state_hash_match: bool
    task_success: bool
    rejected_calls_count: int
    pred_calls: list[dict[str, Any]]
    gold_calls: list[dict[str, Any]]
    pred_hash: str
    gold_hash: str


@dataclass
class DatasetMetricsSummary:
    total_samples: int = 0
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_f1: float = 0.0
    sequence_exact_match_rate: float = 0.0
    state_hash_accuracy: float = 0.0
    task_success_rate: float = 0.0
    total_rejected_calls: int = 0
    schema_rejection_rate: float = 0.0
    sample_results: list[SampleEvaluationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "mean_precision": round(self.mean_precision, 4),
            "mean_recall": round(self.mean_recall, 4),
            "mean_f1": round(self.mean_f1, 4),
            "sequence_exact_match_rate": round(self.sequence_exact_match_rate, 4),
            "state_hash_accuracy": round(self.state_hash_accuracy, 4),
            "task_success_rate": round(self.task_success_rate, 4),
            "total_rejected_calls": self.total_rejected_calls,
            "schema_rejection_rate": round(self.schema_rejection_rate, 4),
        }


def aggregate_dataset_metrics(results: list[SampleEvaluationResult]) -> DatasetMetricsSummary:
    """Aggregates a list of SampleEvaluationResult into a full dataset summary."""
    n = len(results)
    if n == 0:
        return DatasetMetricsSummary()

    mean_p = sum(r.precision for r in results) / n
    mean_r = sum(r.recall for r in results) / n
    mean_f1 = sum(r.f1 for r in results) / n
    exact_rate = sum(1 for r in results if r.sequence_exact_match) / n
    hash_rate = sum(1 for r in results if r.state_hash_match) / n
    success_rate = sum(1 for r in results if r.task_success) / n
    total_rejected = sum(r.rejected_calls_count for r in results)
    total_calls = sum(len(r.pred_calls) for r in results)
    rejection_rate = (total_rejected / total_calls) if total_calls > 0 else 0.0

    return DatasetMetricsSummary(
        total_samples=n,
        mean_precision=mean_p,
        mean_recall=mean_r,
        mean_f1=mean_f1,
        sequence_exact_match_rate=exact_rate,
        state_hash_accuracy=hash_rate,
        task_success_rate=success_rate,
        total_rejected_calls=total_rejected,
        schema_rejection_rate=rejection_rate,
        sample_results=results,
    )

