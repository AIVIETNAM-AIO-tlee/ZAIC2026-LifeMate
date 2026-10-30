#!/usr/bin/env python3
"""Validate Submission File Schema (Task A17).

Checks submission.jsonl against the official zaic_tools validation rules.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate submission.jsonl using official zaic_tools.")
    parser.add_argument("--submission", type=str, default="result/submission.jsonl", help="Path to submission.jsonl")
    parser.add_argument("--split", type=str, default="public_test", help="Split name (public_test or train)")
    parser.add_argument("--root", type=str, default="data", help="Root data folder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sub_path = Path(args.submission)
    if not sub_path.exists():
        print(f"Error: Submission file not found at {sub_path}")
        sys.exit(1)

    # Set up PYTHONPATH for zaic_tools
    split_dir = Path(args.root) / args.split
    if not split_dir.exists():
        print(f"Error: Split directory not found at {split_dir}")
        sys.exit(1)

    os.environ["PYTHONPATH"] = str(split_dir)
    cmd = f'python -m zaic_tools --split {args.split} --root {args.root} --validate {args.submission}'
    print(f"Executing: {cmd}")
    code = os.system(cmd)
    if code == 0:
        print("\n[VALIDATION PASSED]: File conforms to all competition schema requirements with 0 errors.")
    else:
        print(f"\n[VALIDATION FAILED]: Exit code {code}")
        sys.exit(code)


if __name__ == "__main__":
    main()

