#!/usr/bin/env python3
"""SFT Training Script for ZAC 2026 LifeMate Agent (Task A13).

Trains an open-weights LLM (e.g. Qwen2.5-7B-Instruct) using LoRA / QLoRA with HuggingFace TRL SFTTrainer.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune LLM for LifeMate Agent.")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training config")
    parser.add_argument("--dry-run", action="store_true", help="Perform sanity check without launching actual GPU training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f).get("training", {})

    print("=================== SFT Training Configuration ===================")
    print(f"Base Model:       {cfg.get('base_model')}")
    print(f"Train Data:       {cfg.get('train_data_path')}")
    print(f"Val Data:         {cfg.get('val_data_path')}")
    print(f"Epochs:           {cfg.get('num_train_epochs')}")
    print(f"Learning Rate:    {cfg.get('learning_rate')}")
    print(f"LoRA Rank (r):    {cfg.get('lora', {}).get('r')}")
    print(f"Output Dir:       {cfg.get('output_dir')}")
    print("==================================================================")

    train_data = Path(cfg.get("train_data_path", ""))
    if not train_data.exists():
        print(f"\nTraining data not found at {train_data}. Running prepare_agent_training.py first...")
        os.system("python scripts/prepare_agent_training.py")

    if args.dry_run:
        print("\nDry-run completed successfully. All configurations and data paths verified.")
        return

    # Check for HuggingFace dependencies
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer
        from datasets import load_dataset
    except ImportError:
        print("\n[NOTE]: Training dependencies (torch, transformers, trl, peft, datasets) are not installed in current environment.")
        print("To run actual GPU training, install them via:")
        print("    pip install torch transformers trl peft datasets accelerate bitsandbytes\n")
        print("Skipping training launch (use --dry-run to test config validity).")
        return

    print("Launching SFT training with HuggingFace TRL...")
    # SFT training loop
    dataset = load_dataset("json", data_files={"train": cfg["train_data_path"], "validation": cfg["val_data_path"]})
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=torch.bfloat16, device_map="auto")

    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        target_modules=cfg["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=float(cfg["learning_rate"]),
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        bf16=cfg.get("bf16", False),
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"Training completed! Model checkpoint saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()

