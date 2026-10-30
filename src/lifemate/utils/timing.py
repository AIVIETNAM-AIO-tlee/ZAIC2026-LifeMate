"""Timing and Profiling Utilities for ZAC 2026 LifeMate Agent & ASR.

Provides precision timers, context managers, and stage-level latency tracking.
"""

from __future__ import annotations

from contextlib import contextmanager
import functools
import time
from time import perf_counter
from typing import Any, Callable, Iterator


class Timer:
    """Context manager and stopwatch for measuring execution duration."""

    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed * 1000.0


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Yield a result mapping populated with elapsed seconds when the block exits."""
    result: dict[str, float] = {}
    started = perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = perf_counter() - started


def timed(func: Callable) -> Callable:
    """Decorator to measure and log function execution time in milliseconds."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration_ms = (time.perf_counter() - start) * 1000.0
        return result, duration_ms
    return wrapper


def timed_call(function: Callable, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Measure function execution time in seconds."""
    started = perf_counter()
    result = function(*args, **kwargs)
    return result, perf_counter() - started


class LatencyTracker:
    """Collects and aggregates timing across pipeline stages."""

    def __init__(self):
        self.stages: dict[str, list[float]] = {}

    def record(self, stage_name: str, duration_ms: float) -> None:
        self.stages.setdefault(stage_name, []).append(duration_ms)

    def summary(self) -> dict[str, dict[str, float]]:
        stats = {}
        for stage, durations in self.stages.items():
            if not durations:
                continue
            sorted_d = sorted(durations)
            n = len(sorted_d)
            p50 = sorted_d[int(n * 0.50)]
            p90 = sorted_d[min(int(n * 0.90), n - 1)]
            p99 = sorted_d[min(int(n * 0.99), n - 1)]
            mean = sum(durations) / n
            stats[stage] = {
                "count": n,
                "mean_ms": round(mean, 2),
                "p50_ms": round(p50, 2),
                "p90_ms": round(p90, 2),
                "p99_ms": round(p99, 2),
            }
        return stats
