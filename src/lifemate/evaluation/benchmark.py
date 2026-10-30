"""Runtime and ASR Benchmarking Engine for ZAC 2026 LifeMate.

Provides:
- BenchmarkRunner / BenchmarkReport: end-to-end runtime profiling and resource metrics.
- validation_split / benchmark_models: deterministic validation split and multi-model ASR evaluation.
"""

from __future__ import annotations

import random
import statistics
import time
import tracemalloc
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from lifemate.agent.context import build_asr_context
from lifemate.evaluation.asr_metrics import compute_asr_metrics
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


class BatchTranscriber(Protocol):
    def transcribe_batch(self, audio_paths, contexts=None) -> list[str]: ...


def validation_split(
    samples: Sequence[Mapping[str, Any]],
    fraction: float = 0.1,
    seed: int = 42,
) -> list[Mapping[str, Any]]:
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be strictly between 0 and 1")
    if len(samples) < 2:
        raise ValueError("At least two samples are required to create a validation split")
    ids = [sample.get("id") for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Sample IDs must be unique for a stable validation split")
    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)
    count = min(len(samples) - 1, max(1, round(len(samples) * fraction)))
    chosen = set(order[:count])
    return [sample for index, sample in enumerate(samples) if index in chosen]


def _group_name(sample: Mapping[str, Any]) -> tuple[str, str]:
    context = sample.get("context", {})
    if not isinstance(context, Mapping):
        return "unknown", "none"
    profile = context.get("user_profile", {})
    profile = profile if isinstance(profile, Mapping) else {}
    for source in (sample, context, profile):
        for field in ("accent", "dialect", "region"):
            value = source.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip(), field
    city = profile.get("city")
    if isinstance(city, str) and city.strip():
        return f"city:{city.strip()}", "city_proxy_not_accent"
    return "unknown", "none"


def benchmark_models(
    models: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    audio_root: str | Path,
    batch_size: int = 1,
) -> dict[str, Any]:
    if not models:
        raise ValueError("At least one ASR model is required")
    if not samples:
        raise ValueError("At least one validation sample is required")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    root = Path(audio_root)
    model_results = {}
    metadata_basis: set[str] = set()
    for model_name, model in models.items():
        examples = []
        elapsed = 0.0
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            audio_paths = [root / str(item["audio"]) for item in batch]
            contexts = [
                build_asr_context(item.get("context"), item.get("history"))
                for item in batch
            ]
            started = perf_counter()
            if hasattr(model, "transcribe_batch"):
                hypotheses = model.transcribe_batch(audio_paths, contexts=contexts)
            else:
                hypotheses = [
                    model.transcribe(path, context=context)
                    for path, context in zip(audio_paths, contexts)
                ]
            elapsed += perf_counter() - started
            if len(hypotheses) != len(batch):
                raise ValueError(f"Model {model_name!r} returned a different number of transcripts than inputs")
            for item, hypothesis in zip(batch, hypotheses):
                reference = item.get("audio_text")
                if not isinstance(reference, str):
                    raise ValueError(f"Sample {item.get('id')!r} is missing audio_text")
                group, basis = _group_name(item)
                metadata_basis.add(basis)
                examples.append(
                    {
                        "id": item["id"],
                        "reference": reference,
                        "hypothesis": hypothesis,
                        "group": group,
                    }
                )
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for example in examples:
            grouped[example["group"]].append(example)
        per_group = {
            group: compute_asr_metrics(
                [item["reference"] for item in members],
                [item["hypothesis"] for item in members],
            )
            for group, members in sorted(grouped.items())
        }
        model_results[model_name] = {
            "metrics": compute_asr_metrics(
                [item["reference"] for item in examples],
                [item["hypothesis"] for item in examples],
            ),
            "per_group": per_group,
            "elapsed_seconds": elapsed,
            "seconds_per_sample": elapsed / len(examples),
            "examples": examples,
        }
    return {
        "validation_ids": [sample["id"] for sample in samples],
        "grouping_basis": sorted(metadata_basis),
        "models": model_results,
    }
