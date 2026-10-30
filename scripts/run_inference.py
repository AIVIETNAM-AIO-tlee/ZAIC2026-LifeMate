"""Generate a submission using an injected agent factory and the local tool package."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lifemate.inference.pipeline import InferencePipeline
from lifemate.speech.asr import ASRModel, load_asr_config, load_runtime_config
from lifemate.speech.pipeline import SpeechPipeline


def _load_factory(specification: str):
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("--agent must be formatted as python.module:factory")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"Agent factory is not callable: {specification}")
    return factory


def _read_rows(path: Path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def run(
    data_root: Path,
    split: str,
    config_path: Path,
    output_path: Path,
    agent_factory: Any,
    runtime_config_path: Path | None = None,
) -> int:
    split_dir = data_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
    sys.path.insert(0, str(split_dir))
    from zaic_tools import ToolEnv

    config = load_asr_config(config_path)
    if runtime_config_path is not None:
        config["runtime"] = {**config.get("runtime", {}), **load_runtime_config(runtime_config_path)}
    model = ASRModel(config)
    speech = SpeechPipeline(model, int(config.get("preprocessing", {}).get("sample_rate", 16_000)))
    agent = agent_factory()
    runtime = config.get("runtime", {})
    pipeline = InferencePipeline(speech, max_tool_calls=int(runtime.get("max_tool_calls", 12)))
    dataset = split_dir / f"{split}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for item in _read_rows(dataset):
            environment = ToolEnv(split, item["id"], root=data_root)
            result = pipeline.run(
                audio=split_dir / item["audio"],
                context=item.get("context", {}),
                history=item.get("history", []),
                agent=agent,
                tool_env=environment,
                sample_id=item["id"],
            )
            submission = {
                "id": item["id"],
                "tool_calls": result["tool_calls"],
                "state_hash": result["state_hash"],
            }
            output.write(json.dumps(submission, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="public_test")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "asr.yaml")
    parser.add_argument("--runtime-config", type=Path, default=ROOT / "configs" / "runtime.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "submission.jsonl")
    parser.add_argument("--agent", required=True, help="Factory as python.module:factory")
    args = parser.parse_args()
    count = run(
        args.data_root,
        args.split,
        args.config,
        args.output,
        _load_factory(args.agent),
        args.runtime_config,
    )
    print(f"{count} submission rows -> {args.output}")


if __name__ == "__main__":
    main()