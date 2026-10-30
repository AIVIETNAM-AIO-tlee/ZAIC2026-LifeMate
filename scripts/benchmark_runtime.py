#!/usr/bin/env python3
"""CLI Runtime Benchmark for ZAC 2026 LifeMate Agent.

Benchmarks agent latency, throughput, memory consumption, and verifies compliance with
the 10-second per sample competition limit.
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

from lifemate.agent.planner import HeuristicPlanner
from lifemate.evaluation.benchmark import BenchmarkRunner
from lifemate.tools.catalog import ToolCatalog
from lifemate.tools.executor import InteractiveExecutor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark agent runtime.")
    parser.add_argument("--data-path", type=str, default="data/train/train.jsonl")
    parser.add_argument("--catalog-path", type=str, default="data/train/train_tools_catalog.json")
    parser.add_argument("--limit", type=int, default=30, help="Number of samples to profile")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-sample timeout limit in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = ToolCatalog.from_file(args.catalog_path)
    planner = HeuristicPlanner(catalog=catalog)
    executor = InteractiveExecutor(planner=planner, catalog=catalog)
    runner = BenchmarkRunner(executor=executor, max_timeout_sec=args.timeout)

    samples = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                if args.limit and len(samples) >= args.limit:
                    break

    print(f"Benchmarking runtime on {len(samples)} samples (Timeout threshold: {args.timeout}s)...")
    report = runner.benchmark(samples, split=args.split, root="data")

    print("\n================== Runtime Benchmark Results ==================")
    print(f"Total Samples:        {report.total_samples}")
    print(f"Total Time:           {report.total_time_seconds:.2f} s")
    print(f"Throughput:           {report.throughput_samples_per_sec:.2f} samples/sec")
    print(f"Mean Latency:         {report.mean_latency_ms:.2f} ms ({report.mean_latency_ms/1000:.3f} s)")
    print(f"P50 Latency:          {report.p50_latency_ms:.2f} ms")
    print(f"P90 Latency:          {report.p90_latency_ms:.2f} ms")
    print(f"P99 Latency:          {report.p99_latency_ms:.2f} ms")
    print(f"Max Latency:          {report.max_latency_ms:.2f} ms")
    print(f"Peak Memory:          {report.peak_memory_mb:.2f} MB")
    print(f"Timeout Violations:   {report.timeout_violations}")
    print("===============================================================\n")

    if report.timeout_violations == 0:
        print("[SUCCESS]: 100% of samples completed well within competition latency limits!")
    else:
        print(f"[WARNING]: {report.timeout_violations} samples exceeded the {args.timeout}s threshold.")


if __name__ == "__main__":
    main()
