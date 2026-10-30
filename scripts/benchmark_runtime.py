#!/usr/bin/env python3
"""CLI Runtime Benchmark for ZAC 2026 LifeMate (Agent and ASR).

Benchmarks agent latency, throughput, memory consumption, and ASR throughput/latency.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src is in sys.path so script can be run directly from any directory
ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from lifemate.agent.planner import HeuristicPlanner
from lifemate.evaluation.benchmark import BenchmarkRunner
from lifemate.tools.catalog import ToolCatalog
from lifemate.tools.executor import InteractiveExecutor
from lifemate.utils.timing import timed_call


def benchmark_asr(model, audio_paths: list[Path], batch_size: int) -> dict:
    if not audio_paths:
        raise ValueError("No WAV files found for runtime benchmark")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    elapsed = 0.0
    for start in range(0, len(audio_paths), batch_size):
        batch = audio_paths[start : start + batch_size]
        _, batch_elapsed = timed_call(model.transcribe_batch, batch)
        elapsed += batch_elapsed
    return {
        "samples": len(audio_paths),
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "seconds_per_sample": elapsed / len(audio_paths),
        "samples_per_second": len(audio_paths) / elapsed if elapsed else None,
        "device": getattr(model, "device", "unknown"),
        "precision": str(getattr(model, "dtype", "unknown")),
        "model": getattr(model, "model_name", "asr"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark agent and ASR runtime.")
    parser.add_argument("--mode", type=str, choices=["agent", "asr"], default="agent", help="Benchmark target: agent or asr")
    parser.add_argument("--data-path", type=str, default="data/train/train.jsonl")
    parser.add_argument("--catalog-path", type=str, default="data/train/train_tools_catalog.json")
    parser.add_argument("--limit", type=int, default=30, help="Number of samples to profile")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-sample timeout limit in seconds")
    # ASR specific arguments
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "asr.yaml")
    parser.add_argument("--runtime-config", type=Path, default=ROOT / "configs" / "runtime.yaml")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "asr":
        from lifemate.speech.asr import ASRModel, load_asr_config, load_runtime_config
        config = load_asr_config(args.config)
        config["runtime"] = {**config.get("runtime", {}), **load_runtime_config(args.runtime_config)}
        model = ASRModel(config)
        paths = sorted((args.data_root / "audio").rglob("*.wav"))[: args.limit]
        result = benchmark_asr(model, paths, args.batch_size or int(config.get("runtime", {}).get("batch_size", 1)))
        out_path = args.output or (ROOT / "experiments" / "asr" / "runtime_benchmark.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(result)
        print(f"ASR runtime report -> {out_path}")
        return

    # Default: Agent runtime benchmark
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
