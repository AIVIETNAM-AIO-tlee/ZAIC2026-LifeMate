#!/usr/bin/env python3
"""Evaluation Harness for ZAC 2026 LifeMate Agent (Tasks A04, A14, A15).

Evaluates the agent pipeline against standard gold transcripts (train.jsonl or validation slice),
computing Tool Call Precision, Recall, F1, Sequence Exact Match, State Hash Accuracy,
Task Success Rate, Schema Rejection Rate, and exports error analysis reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src is in sys.path so script can be run directly from any directory
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from lifemate.agent.planner import AgentPlanner, GeminiRESTClient, HeuristicPlanner, MockLLMClient
from lifemate.evaluation.agent_metrics import (
    SampleEvaluationResult,
    aggregate_dataset_metrics,
    compute_sequence_exact_match,
    compute_tool_calls_metrics,
)
from lifemate.evaluation.error_analysis import ErrorAnalyzer
from lifemate.tools.catalog import ToolCatalog
from lifemate.tools.executor import InteractiveExecutor, get_tool_env
from lifemate.utils.logging import StructuredTracer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate agent baseline.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/public_test)")
    parser.add_argument("--data-path", type=str, default="data/train/train.jsonl")
    parser.add_argument("--catalog-path", type=str, default="data/train/train_tools_catalog.json")
    parser.add_argument("--limit", type=int, default=20, help="Number of samples to evaluate (0 for all)")
    parser.add_argument("--provider", type=str, default="heuristic", choices=["gemini", "mock", "heuristic"])
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--output-submission", type=str, default="result/submission.jsonl")
    parser.add_argument("--error-report", action="store_true", help="Generate detailed error analysis report")
    parser.add_argument("--trace-file", type=str, default=None, help="Path to write structured JSONL trace")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = ToolCatalog.from_file(args.catalog_path)
    tracer = StructuredTracer(args.trace_file) if args.trace_file else None

    # Initialize Planner
    if args.provider == "heuristic":
        planner = HeuristicPlanner(catalog=catalog)
    elif args.provider == "gemini":
        llm_client = GeminiRESTClient(model=args.model)
        planner = AgentPlanner(catalog=catalog, llm_client=llm_client)
    else:
        llm_client = MockLLMClient()
        planner = AgentPlanner(catalog=catalog, llm_client=llm_client)

    executor = InteractiveExecutor(planner=planner, catalog=catalog)

    out_path = Path(args.output_submission)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sample_results: list[SampleEvaluationResult] = []
    total_processed = 0

    print(f"Starting evaluation on {args.data_path} (limit={args.limit}, provider={args.provider})...")

    with open(args.data_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            sample = json.loads(line)
            total_processed += 1
            sample_id = sample.get("id", f"sample_{total_processed}")

            # Run agent session
            result = executor.execute_sample(sample, split=args.split, root="data")
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

            pred_calls = result.get("tool_calls", [])
            pred_hash = result.get("state_hash", "")

            gold_label = sample.get("label", {})
            gold_calls = gold_label.get("tool_calls", [])

            # Compute gold state hash by replaying gold calls
            gold_env = get_tool_env(args.split, sample_id, root="data")
            for gc in gold_calls:
                gold_env.call(gc.get("name", ""), gc.get("arguments") or {})
            gold_hash = gold_env.get_state_hash()

            # Compute metrics
            call_metrics = compute_tool_calls_metrics(pred_calls, gold_calls)
            exact_match = compute_sequence_exact_match(pred_calls, gold_calls)
            hash_match = (pred_hash == gold_hash)
            task_success = (call_metrics.f1 == 1.0 and hash_match)

            sample_res = SampleEvaluationResult(
                sample_id=sample_id,
                precision=call_metrics.precision,
                recall=call_metrics.recall,
                f1=call_metrics.f1,
                sequence_exact_match=exact_match,
                state_hash_match=hash_match,
                task_success=task_success,
                rejected_calls_count=0,
                pred_calls=pred_calls,
                gold_calls=gold_calls,
                pred_hash=pred_hash,
                gold_hash=gold_hash,
            )
            sample_results.append(sample_res)

            if tracer:
                tracer.log_event("sample_evaluated", {
                    "sample_id": sample_id,
                    "f1": call_metrics.f1,
                    "hash_match": hash_match,
                })

            if total_processed % 5 == 0 or total_processed == args.limit:
                current_f1 = sum(r.f1 for r in sample_results) / len(sample_results)
                print(f"Processed {total_processed}/{args.limit} samples... Current Avg F1: {current_f1:.4f}")

            if args.limit and total_processed >= args.limit:
                break

    summary = aggregate_dataset_metrics(sample_results)

    print("\n================ Evaluation Results ================")
    print(f"Total Samples Evaluated:   {summary.total_samples}")
    print(f"Tool Calls Precision:     {summary.mean_precision * 100:.2f}%")
    print(f"Tool Calls Recall:        {summary.mean_recall * 100:.2f}%")
    print(f"Tool Calls F1 Score:      {summary.mean_f1 * 100:.2f}%")
    print(f"Sequence Exact Match:     {summary.sequence_exact_match_rate * 100:.2f}%")
    print(f"State Hash Accuracy:      {summary.state_hash_accuracy * 100:.2f}%")
    print(f"Task Success Rate:        {summary.task_success_rate * 100:.2f}%")
    print(f"Schema Rejection Rate:    {summary.schema_rejection_rate * 100:.2f}%")
    print(f"Results written to:       {out_path}")
    print("====================================================\n")

    if args.error_report:
        err_summary = ErrorAnalyzer.generate_error_report(sample_results, report_dir="experiments/reports")
        print(f"Error report generated at experiments/reports/error_analysis_report.md")


if __name__ == "__main__":
    main()
