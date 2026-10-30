"""Evaluate one or more configured ASR models on one shared held-out split."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lifemate.evaluation.benchmark import benchmark_models, validation_split
from lifemate.evaluation.error_analysis import analyze_errors
from lifemate.speech.asr import ASRModel, load_asr_config, load_runtime_config


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def run(
    data_root: Path,
    config_path: Path,
    output_path: Path,
    fraction: float | None = None,
    seed: int | None = None,
    batch_size: int | None = None,
    report_path: Path | None = None,
    runtime_config_path: Path | None = None,
) -> dict:
    config = load_asr_config(config_path)
    if runtime_config_path is not None:
        config["runtime"] = {**config.get("runtime", {}), **load_runtime_config(runtime_config_path)}
    samples = _read_rows(data_root / f"{data_root.name}.jsonl")
    validation = validation_split(
        samples,
        fraction=float(fraction if fraction is not None else config.get("validation", {}).get("fraction", 0.1)),
        seed=int(seed if seed is not None else config.get("validation", {}).get("seed", 42)),
    )
    definitions = config.get("models") or [{"name": "default", **config.get("model", {})}]
    if not isinstance(definitions, list) or not definitions:
        raise ValueError("configs/asr.yaml must define at least one model")
    model_results = {}
    grouping_basis = set()
    for definition in definitions:
        if not isinstance(definition, dict):
            raise ValueError("Each entry in the ASR models list must be a mapping")
        name = str(definition.get("name") or definition.get("model_name_or_path") or "asr")
        if name in model_results:
            raise ValueError(f"Duplicate ASR model name: {name}")
        model_config = dict(config)
        model_config["model"] = {key: value for key, value in definition.items() if key != "name"}
        model = ASRModel(model_config)
        evaluated = benchmark_models(
            {name: model},
            validation,
            data_root,
            batch_size=int(batch_size or config.get("runtime", {}).get("batch_size", 1)),
        )
        grouping_basis.update(evaluated["grouping_basis"])
        model_results[name] = evaluated["models"][name]
        del model
        gc.collect()
    result = {
        "validation_ids": [sample["id"] for sample in validation],
        "grouping_basis": sorted(grouping_basis),
        "models": model_results,
    }
    result["split"] = data_root.name
    result["validation_fraction"] = fraction or config.get("validation", {}).get("fraction", 0.1)
    result["seed"] = seed if seed is not None else config.get("validation", {}).get("seed", 42)
    for model_result in result["models"].values():
        analysis = analyze_errors(model_result["examples"])
        model_result["error_analysis"] = {
            "groups": analysis["groups"],
            "error_types": analysis["error_types"],
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = report_path or ROOT / "experiments" / "reports" / "asr_benchmark.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# ASR benchmark — {data_root.name}",
        "",
        f"- Validation samples: {len(validation)}",
        f"- Validation fraction: {result['validation_fraction']}",
        f"- Seed: {result['seed']}",
        f"- Group fields: {', '.join(result['grouping_basis']) or 'none'}",
        "",
    ]
    for name, model_result in result["models"].items():
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Metrics: `{model_result['metrics']}`",
                f"- Elapsed seconds: {model_result['elapsed_seconds']:.3f}",
                f"- Per-group: `{model_result['per_group']}`",
                f"- Error types: `{model_result['error_analysis']['error_types']}`",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "asr.yaml")
    parser.add_argument("--runtime-config", type=Path, default=ROOT / "configs" / "runtime.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "asr" / "benchmark.json")
    parser.add_argument("--report", type=Path, default=ROOT / "experiments" / "reports" / "asr_benchmark.md")
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    report = run(
        args.data_root,
        args.config,
        args.output,
        args.validation_fraction,
        args.seed,
        args.batch_size,
        args.report,
        args.runtime_config,
    )
    for name, metrics in report["models"].items():
        print(f"{name}: {metrics['metrics']}")
    print(f"Benchmark report -> {args.output}")
    print(f"Summary report -> {args.report}")


if __name__ == "__main__":
    main()