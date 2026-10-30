"""Simple monotonic timers for inference and runtime benchmarks."""
from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Yield a result mapping populated with elapsed seconds when the block exits."""
    result: dict[str, float] = {}
    started = perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = perf_counter() - started


def timed_call(function, *args, **kwargs):
    started = perf_counter()
    result = function(*args, **kwargs)
    return result, perf_counter() - started