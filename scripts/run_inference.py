#!/usr/bin/env python3
"""Run End-to-End Inference and Generate submission.jsonl (Task A17).

Processes test datasets (public_test or private_test), executes the agent pipeline,
generates the submission file, and optionally runs automated schema validation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src is in sys.path so script can be run directly from any directory
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from lifemate.agent.planner import AgentPlanner, GeminiRESTClient, HeuristicPlanner, MockLLMClient
from lifemate.inference.pipeline import InferencePipeline
from lifemate.tools.catalog import ToolCatalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and generate submission.jsonl.")
    parser.add_argument("--split", type=str, default="public_test", help="Split name (public_test/train/private_test)")
    parser.add_argument("--data-path", type=str, default="data/public_test/public_test.jsonl")
    parser.add_argument("--catalog-path", type=str, default="data/public_test/public_test_tools_catalog.json")
    parser.add_argument("--output", type=str, default="result/submission.jsonl", help="Output submission path")
    parser.add_argument("--provider", type=str, default="heuristic", choices=["heuristic", "gemini", "mock"])
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (0 for all)")
    parser.add_argument("--validate", action="store_true", default=True, help="Automatically validate output with zaic_tools")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog_path = Path(args.catalog_path)
    if not catalog_path.exists():
        print(f"Error: Catalog file not found at {catalog_path}")
        sys.exit(1)

    catalog = ToolCatalog.from_file(catalog_path)

    # Initialize Planner
    if args.provider == "heuristic":
        planner = HeuristicPlanner(catalog=catalog)
    elif args.provider == "gemini":
        llm = GeminiRESTClient(model=args.model)
        planner = AgentPlanner(catalog=catalog, llm_client=llm)
    else:
        llm = MockLLMClient()
        planner = AgentPlanner(catalog=catalog, llm_client=llm)

    pipeline = InferencePipeline(catalog=catalog, planner=planner)
    written_count = pipeline.run(
        data_path=args.data_path,
        output_path=args.output,
        split=args.split,
        root="data",
        limit=args.limit,
    )

    print(f"\n[INFO]: Finished generating submission file: {args.output} ({written_count} lines).")

    # Automated validation if requested
    if args.validate and written_count > 0:
        print("[INFO]: Running automated validation via zaic_tools...")
        import os
        val_cmd = f'python -m zaic_tools --split {args.split} --root data --validate {args.output}'
        os.environ["PYTHONPATH"] = f"data/{args.split};src"
        exit_code = os.system(val_cmd)
        if exit_code == 0:
            print("[SUCCESS]: Submission file is 100% valid with 0 schema errors!")
        else:
            print(f"[WARNING]: Validation returned exit code {exit_code}.")


if __name__ == "__main__":
    main()

