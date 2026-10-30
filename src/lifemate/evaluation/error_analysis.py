"""Error Analysis & Failure Taxonomy for ZAC 2026 LifeMate (Agent & ASR).

Includes:
- Agent failure taxonomy and report generator (Task A15)
- ASR word-level error analysis and grouped evaluation (Task B07/B08)
"""

from __future__ import annotations

import collections
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

from lifemate.evaluation.agent_metrics import SampleEvaluationResult, compare_arguments
from lifemate.evaluation.asr_metrics import compute_asr_metrics
from lifemate.speech.normalizer import normalize_text


# ============================================================================
# Agent Error Analysis (Task A15)
# ============================================================================

class ErrorType:
    SCHEMA_REJECTION = "SCHEMA_REJECTION"
    OUT_OF_SCOPE_FAIL = "OUT_OF_SCOPE_FAIL"
    MISSING_INFO_FAIL = "MISSING_INFO_FAIL"
    WRONG_TOOL_NAME = "WRONG_TOOL_NAME"
    WRONG_ARGUMENTS = "WRONG_ARGUMENTS"
    MISSING_CALL = "MISSING_CALL"
    EXTRA_CALL = "EXTRA_CALL"
    STATE_HASH_MISMATCH = "STATE_HASH_MISMATCH"


@dataclass
class SampleErrorRecord:
    sample_id: str
    error_types: list[str]
    description: str
    pred_calls: list[dict[str, Any]]
    gold_calls: list[dict[str, Any]]
    pred_hash: str
    gold_hash: str


class ErrorAnalyzer:
    """Analyzes evaluation results and produces diagnostic breakdown."""

    @staticmethod
    def classify_sample_errors(result: SampleEvaluationResult) -> list[str]:
        """Identifies all error modes present in a sample."""
        errors: list[str] = []

        # 1. Schema rejection
        if result.rejected_calls_count > 0:
            errors.append(ErrorType.SCHEMA_REJECTION)

        pred_names = [c.get("name") for c in result.pred_calls]
        gold_names = [c.get("name") for c in result.gold_calls]

        # 2. Out of scope failure
        if not result.gold_calls and result.pred_calls:
            errors.append(ErrorType.OUT_OF_SCOPE_FAIL)
        elif result.gold_calls and not result.pred_calls:
            errors.append(ErrorType.OUT_OF_SCOPE_FAIL)

        # 3. Missing info failure
        gold_has_ask = any(n == "sys.ask_missing_info" for n in gold_names)
        pred_has_ask = any(n == "sys.ask_missing_info" for n in pred_names)
        if gold_has_ask != pred_has_ask:
            errors.append(ErrorType.MISSING_INFO_FAIL)

        # 4. Tool calls count / missing or extra
        if len(result.pred_calls) < len(result.gold_calls):
            errors.append(ErrorType.MISSING_CALL)
        elif len(result.pred_calls) > len(result.gold_calls):
            errors.append(ErrorType.EXTRA_CALL)

        # 5. Wrong tool name or arguments
        for idx, (p, g) in enumerate(zip(result.pred_calls, result.gold_calls)):
            if p.get("name") != g.get("name"):
                if ErrorType.WRONG_TOOL_NAME not in errors:
                    errors.append(ErrorType.WRONG_TOOL_NAME)
            else:
                if not compare_arguments(p.get("arguments") or {}, g.get("arguments") or {}):
                    if ErrorType.WRONG_ARGUMENTS not in errors:
                        errors.append(ErrorType.WRONG_ARGUMENTS)

        # 6. State hash mismatch
        if not result.state_hash_match and ErrorType.STATE_HASH_MISMATCH not in errors:
            errors.append(ErrorType.STATE_HASH_MISMATCH)

        return errors

    @classmethod
    def generate_error_report(
        cls,
        sample_results: list[SampleEvaluationResult],
        report_dir: str | Path = "experiments/reports",
    ) -> dict[str, Any]:
        """Generates comprehensive Markdown and JSON error reports."""
        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)

        error_counts: collections.Counter[str] = collections.Counter()
        error_records: list[SampleErrorRecord] = []
        total_failed_samples = 0

        for r in sample_results:
            if r.task_success:
                continue

            total_failed_samples += 1
            sample_errors = cls.classify_sample_errors(r)
            for err in sample_errors:
                error_counts[err] += 1

            desc = f"Failed with: {', '.join(sample_errors)}"
            error_records.append(
                SampleErrorRecord(
                    sample_id=r.sample_id,
                    error_types=sample_errors,
                    description=desc,
                    pred_calls=r.pred_calls,
                    gold_calls=r.gold_calls,
                    pred_hash=r.pred_hash,
                    gold_hash=r.gold_hash,
                )
            )

        summary = {
            "total_evaluated": len(sample_results),
            "total_successful": len(sample_results) - total_failed_samples,
            "total_failed": total_failed_samples,
            "error_distribution": dict(error_counts),
            "top_failed_examples": [
                {
                    "sample_id": e.sample_id,
                    "errors": e.error_types,
                    "pred_calls": [c.get("name") for c in e.pred_calls],
                    "gold_calls": [c.get("name") for c in e.gold_calls],
                }
                for e in error_records[:10]
            ],
        }

        # Save JSON
        json_path = report_path / "error_analysis_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # Save Markdown
        md_path = report_path / "error_analysis_report.md"
        cls._write_markdown_report(summary, error_records, md_path)

        return summary

    @staticmethod
    def _write_markdown_report(summary: dict, records: list[SampleErrorRecord], output_path: Path) -> None:
        lines = [
            "# Báo Cáo Phân Tích Lỗi Agent (Agent Error Analysis Report)",
            f"- **Tổng số mẫu đánh giá:** {summary['total_evaluated']}",
            f"- **Số mẫu thành công:** {summary['total_successful']}",
            f"- **Số mẫu thất bại:** {summary['total_failed']}",
            "",
            "## 1. Phân Bố Các Loại Lỗi",
            "",
            "| Loại Lỗi (Error Type) | Số Lượng Xuất Hiện | Mô Tả |",
            "| :--- | :---: | :--- |",
        ]
        descriptions = {
            ErrorType.SCHEMA_REJECTION: "Call bị từ chối bởi ToolEnv do sai schema",
            ErrorType.OUT_OF_SCOPE_FAIL: "Xử lý sai mẫu ngoài phạm vi (gọi tool khi không cần hoặc ngược lại)",
            ErrorType.MISSING_INFO_FAIL: "Quên gọi hoặc gọi sai `sys.ask_missing_info`",
            ErrorType.WRONG_TOOL_NAME: "Gọi nhầm công cụ so với đáp án gold",
            ErrorType.WRONG_ARGUMENTS: "Đúng tên công cụ nhưng tham số truyền vào sai lệch",
            ErrorType.MISSING_CALL: "Thiếu một hoặc nhiều bước gọi công cụ bắt buộc",
            ErrorType.EXTRA_CALL: "Gọi thừa bước không cần thiết",
            ErrorType.STATE_HASH_MISMATCH: "Lệch hash trạng thái ứng dụng cuối cùng",
        }
        for err_type, count in summary["error_distribution"].items():
            desc = descriptions.get(err_type, "")
            lines.append(f"| `{err_type}` | {count} | {desc} |")

        lines.extend([
            "",
            "## 2. Danh Sách Mẫu Lỗi Điển Hình",
            "",
            "| Sample ID | Nhóm Lỗi | Predicted Calls | Gold Calls |",
            "| :--- | :--- | :--- | :--- |",
        ])
        for rec in records[:15]:
            p_str = ", ".join(c.get("name", "") for c in rec.pred_calls) or "[]"
            g_str = ", ".join(c.get("name", "") for c in rec.gold_calls) or "[]"
            errs_str = ", ".join(rec.error_types)
            lines.append(f"| `{rec.sample_id}` | `{errs_str}` | `{p_str}` | `{g_str}` |")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


# ============================================================================
# ASR Error Analysis (Task B07 / B08)
# ============================================================================

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
