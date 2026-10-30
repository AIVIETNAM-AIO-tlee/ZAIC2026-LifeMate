"""Shared deterministic validation split and multi-model ASR benchmarking."""
from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from lifemate.agent.context import build_asr_context
from lifemate.evaluation.asr_metrics import compute_asr_metrics


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