#!/usr/bin/env python3
"""Run End-to-End Inference and Generate submission.jsonl (Tasks A17 & B10).

Processes evaluation datasets, executes the agent/speech pipeline,
generates the submission file, and optionally runs automated schema validation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from lifemate.agent.planner import AgentPlanner, GeminiRESTClient, HeuristicPlanner, MockLLMClient, OpenAIRESTClient
from lifemate.inference.pipeline import InferencePipeline
from lifemate.inference.submission import SubmissionRecord, SubmissionWriter
from lifemate.speech.asr import ASRModel, load_asr_config, load_runtime_config
from lifemate.speech.pipeline import SpeechPipeline
from lifemate.tools.catalog import ToolCatalog
from lifemate.tools.executor import get_tool_env
from lifemate.utils.config import get_llm_config, get_runtime_config


def _load_factory(specification: str) -> Any:
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


def parse_args() -> argparse.Namespace:
    rt_cfg = get_runtime_config()
    llm_cfg = get_llm_config()

    parser = argparse.ArgumentParser(description="Run inference and generate submission.jsonl.")
    parser.add_argument("--split", type=str, default="public_test", help="Split name (public_test/train/private_test)")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data", help="Root data directory")
    parser.add_argument("--data-path", type=str, default=None, help="Direct path to .jsonl (overrides data-root/split)")
    parser.add_argument("--catalog-path", type=str, default=None, help="Path to tool catalog json")
    parser.add_argument("--output", type=str, default=f"{rt_cfg.get('output_dir', 'result')}/submission.jsonl", help="Output submission path")
    parser.add_argument("--provider", type=str, default=llm_cfg.get("provider", "heuristic"), choices=["heuristic", "gemini", "openai", "mock"])
    parser.add_argument("--model", type=str, default=llm_cfg.get("model", "gemini-2.5-flash"))
    parser.add_argument("--agent", type=str, default=None, help="Agent factory formatted as python.module:factory")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "asr.yaml", help="ASR config path")
    parser.add_argument("--runtime-config", type=Path, default=ROOT / "configs" / "runtime.yaml", help="Runtime config path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (0 for all)")
    parser.add_argument("--validate", action="store_true", default=True, help="Automatically validate output with zaic_tools")
    parser.add_argument("--no-validate", dest="validate", action="store_false", help="Disable automated validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    split_dir = data_root / args.split
    data_path = Path(args.data_path) if args.data_path else split_dir / f"{args.split}.jsonl"

    if not data_path.exists():
        print(f"Error: Input data not found at {data_path}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.agent:
        # Injected Agent factory flow (Task B10)
        print(f"[INFO]: Using injected agent factory: {args.agent}")
        agent_factory = _load_factory(args.agent)
        agent = agent_factory()

        config = load_asr_config(args.config)
        if args.runtime_config and args.runtime_config.exists():
            config["runtime"] = {**config.get("runtime", {}), **load_runtime_config(args.runtime_config)}
        model = ASRModel(config)
        speech = SpeechPipeline(model, int(config.get("preprocessing", {}).get("sample_rate", 16_000)))
        runtime = config.get("runtime", {})
        pipeline = InferencePipeline(speech=speech, max_tool_calls=int(runtime.get("max_tool_calls", 12)))

        count = 0
        with output_path.open("w", encoding="utf-8") as output:
            for item in _read_rows(data_path):
                sample_id = item["id"]
                environment = get_tool_env(args.split, sample_id, root=data_root)
                audio_path = split_dir / item["audio"] if "audio" in item else None
                result = pipeline.run(
                    audio=audio_path,
                    transcript=item.get("audio_text"),
                    context=item.get("context", {}),
                    history=item.get("history", []),
                    agent=agent,
                    tool_env=environment,
                    sample_id=sample_id,
                )
                submission = {
                    "id": sample_id,
                    "tool_calls": result["tool_calls"],
                    "state_hash": result["state_hash"],
                }
                output.write(json.dumps(submission, ensure_ascii=False) + "\n")
                count += 1
                if args.limit and count >= args.limit:
                    break
        written_count = count
    else:
        # Standard Planner flow (Task A17)
        catalog_path = Path(args.catalog_path) if args.catalog_path else split_dir / f"{args.split}_tools_catalog.json"
        if not catalog_path.exists():
            print(f"Error: Catalog file not found at {catalog_path}")
            sys.exit(1)

        catalog = ToolCatalog.from_file(catalog_path)
        if args.provider == "heuristic":
            planner = HeuristicPlanner(catalog=catalog)
        elif args.provider == "gemini":
            llm = GeminiRESTClient(model=args.model)
            planner = AgentPlanner(catalog=catalog, llm_client=llm)
        elif args.provider == "openai":
            llm = OpenAIRESTClient(model=args.model)
            planner = AgentPlanner(catalog=catalog, llm_client=llm)
        else:
            llm = MockLLMClient()
            planner = AgentPlanner(catalog=catalog, llm_client=llm)

        # Build optional speech pipeline for transcription when audio_text is missing
        speech = None
        try:
            if args.config.exists():
                cfg = load_asr_config(args.config)
                if args.runtime_config and args.runtime_config.exists():
                    cfg["runtime"] = {**cfg.get("runtime", {}), **load_runtime_config(args.runtime_config)}
                model = ASRModel(cfg)
                speech = SpeechPipeline(model, int(cfg.get("preprocessing", {}).get("sample_rate", 16_000)))
        except Exception:
            # If torch/ASR is not available, proceed without ASR
            speech = None

        pipeline = InferencePipeline(catalog=catalog, planner=planner, speech=speech)
        written_count = pipeline.run(
            data_path=data_path,
            output_path=output_path,
            split=args.split,
            root=data_root,
            limit=args.limit,
        )

    print(f"\n[INFO]: Finished generating submission file: {output_path} ({written_count} lines).")

    # Automated validation if requested
    if args.validate and written_count > 0:
        print("[INFO]: Running automated validation via zaic_tools...")
        val_cmd = f'python -m zaic_tools --split {args.split} --root "{data_root}" --validate "{output_path}"'
        os.environ["PYTHONPATH"] = os.pathsep.join([f"data/{args.split}", "src"])
        exit_code = os.system(val_cmd)
        if exit_code == 0:
            print("[SUCCESS]: Submission file is 100% valid with 0 schema errors!")
        else:
            print(f"[ERROR]: Validation returned exit code {exit_code}.")
            sys.exit(exit_code)


if __name__ == "__main__":
    main()
