"""Runtime Benchmarking Engine for ZAC 2026 LifeMate Agent.

Benchmarks end-to-end execution latency, memory footprint, throughput,
and detects timeout violations against competition limits (10s/sample).
"""

from __future__ import annotations

import statistics
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any

from lifemate.tools.executor import InteractiveExecutor


@dataclass
class BenchmarkReport:
    total_samples: int
    total_time_seconds: float
    throughput_samples_per_sec: float
    mean_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    peak_memory_mb: float
    timeout_violations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "throughput_samples_per_sec": round(self.throughput_samples_per_sec, 2),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p90_latency_ms": round(self.p90_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "timeout_violations": self.timeout_violations,
        }


class BenchmarkRunner:
    """Orchestrates runtime profiling and resource benchmarking."""

    def __init__(self, executor: InteractiveExecutor, max_timeout_sec: float = 10.0):
        self.executor = executor
        self.max_timeout_sec = max_timeout_sec

    def benchmark(
        self,
        samples: list[dict[str, Any]],
        split: str = "train",
        root: str = "data",
    ) -> BenchmarkReport:
        """Executes a list of samples and measures latency and memory."""
        latencies_ms: list[float] = []
        timeout_count = 0

        tracemalloc.start()
        start_wall_time = time.perf_counter()

        for s in samples:
            t0 = time.perf_counter()
            self.executor.execute_sample(s, split=split, root=root)
            elapsed_sec = time.perf_counter() - t0
            elapsed_ms = elapsed_sec * 1000.0

            latencies_ms.append(elapsed_ms)
            if elapsed_sec > self.max_timeout_sec:
                timeout_count += 1

        total_wall_sec = time.perf_counter() - start_wall_time
        _, peak_mem_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        n = len(latencies_ms)
        sorted_latencies = sorted(latencies_ms)
        p50 = sorted_latencies[int(n * 0.50)] if n else 0.0
        p90 = sorted_latencies[min(int(n * 0.90), n - 1)] if n else 0.0
        p99 = sorted_latencies[min(int(n * 0.99), n - 1)] if n else 0.0
        mean = statistics.mean(latencies_ms) if n else 0.0
        max_lat = max(latencies_ms) if n else 0.0
        throughput = (n / total_wall_sec) if total_wall_sec > 0 else 0.0

        return BenchmarkReport(
            total_samples=n,
            total_time_seconds=total_wall_sec,
            throughput_samples_per_sec=throughput,
            mean_latency_ms=mean,
            p50_latency_ms=p50,
            p90_latency_ms=p90,
            p99_latency_ms=p99,
            max_latency_ms=max_lat,
            peak_memory_mb=peak_mem_bytes / (1024 * 1024),
            timeout_violations=timeout_count,
        )

