#!/usr/bin/env python3
"""SFT Dataset Preparation for ZAC 2026 LifeMate Agent (Task A13).

Converts train.jsonl into multi-turn conversational format (ChatML / OpenAI / HuggingFace messages)
for Supervised Fine-Tuning (SFT) of open-weights LLMs (Qwen2.5, Llama-3, Gemma-2).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Ensure src is in sys.path so script can be run directly from any directory
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from lifemate.agent.context import AgentContext
from lifemate.agent.prompts import SYSTEM_PROMPT, build_user_prompt
from lifemate.agent.tool_selector import ToolSelector
from lifemate.tools.catalog import ToolCatalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SFT dataset for agent fine-tuning.")
    parser.add_argument("--data-path", type=str, default="data/train/train.jsonl")
    parser.add_argument("--catalog-path", type=str, default="data/train/train_tools_catalog.json")
    parser.add_argument("--output-dir", type=str, default="data/sft")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def convert_sample_to_messages(
    sample: dict,
    selector: ToolSelector,
) -> dict:
    """Converts a train sample with label (calls, results) into a multi-turn message sequence."""
    context = AgentContext.from_sample(sample)
    tools_summary = selector.to_prompt_summary(context.installed_apps)
    context_summary = context.format_for_prompt()
    user_request = sample.get("audio_text", "")

    label = sample.get("label", {})
    gold_calls = label.get("tool_calls", [])
    gold_results = label.get("tool_results", [])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(user_request, context_summary, tools_summary)},
    ]

    if not gold_calls:
        # Out-of-scope / No action needed
        messages.append({
            "role": "assistant",
            "content": json.dumps(
                {"thought": "Yêu cầu không cần gọi công cụ hoặc ngoài phạm vi.", "action": "finish"},
                ensure_ascii=False,
            ),
        })
        return {"id": sample.get("id"), "messages": messages}

    # Multi-turn tool execution
    for idx, call in enumerate(gold_calls):
        call_decision = {
            "thought": f"Thực thi bước {idx + 1}: {call.get('name')}",
            "action": "call",
            "tool_name": call.get("name"),
            "arguments": call.get("arguments") or {},
        }
        messages.append({
            "role": "assistant",
            "content": json.dumps(call_decision, ensure_ascii=False),
        })

        # Tool result feedback
        res = gold_results[idx] if idx < len(gold_results) else {"ok": True}
        messages.append({
            "role": "user",
            "content": f"[KẾT QUẢ TOOL {call.get('name')}]: {json.dumps(res, ensure_ascii=False)}",
        })

    # Final turn
    messages.append({
        "role": "assistant",
        "content": json.dumps({"thought": "Đã hoàn thành toàn bộ tác vụ.", "action": "finish"}, ensure_ascii=False),
    })

    return {"id": sample.get("id"), "messages": messages}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    catalog = ToolCatalog.from_file(args.catalog_path)
    selector = ToolSelector(catalog)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples from {args.data_path}")

    converted_records = [convert_sample_to_messages(s, selector) for s in samples]
    random.shuffle(converted_records)

    val_count = int(len(converted_records) * args.val_ratio)
    val_records = converted_records[:val_count]
    train_records = converted_records[val_count:]

    train_out = out_dir / "sft_train.jsonl"
    val_out = out_dir / "sft_val.jsonl"

    with open(train_out, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_out, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"SFT Dataset successfully prepared:")
    print(f"  - Train records ({len(train_records)}): {train_out}")
    print(f"  - Val records   ({len(val_records)}):   {val_out}")


if __name__ == "__main__":
    main()

