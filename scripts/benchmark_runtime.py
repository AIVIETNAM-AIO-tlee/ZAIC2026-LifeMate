"""Measure ASR throughput and latency for a configured model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lifemate.speech.asr import ASRModel, load_asr_config, load_runtime_config
from lifemate.utils.timing import timed_call


def benchmark(model: ASRModel, audio_paths: list[Path], batch_size: int) -> dict:
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
        "device": model.device,
        "precision": str(model.dtype),
        "model": model.model_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "asr.yaml")
    parser.add_argument("--runtime-config", type=Path, default=ROOT / "configs" / "runtime.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "asr" / "runtime_benchmark.json")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    config = load_asr_config(args.config)
    config["runtime"] = {**config.get("runtime", {}), **load_runtime_config(args.runtime_config)}
    model = ASRModel(config)
    paths = sorted((args.data_root / "audio").rglob("*.wav"))[: args.limit]
    result = benchmark(
        model,
        paths,
        args.batch_size or int(config.get("runtime", {}).get("batch_size", 1)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result)
    print(f"Runtime report -> {args.output}")


if __name__ == "__main__":
    main()