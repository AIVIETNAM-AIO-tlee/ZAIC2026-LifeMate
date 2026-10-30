"""Configuration loader and merger for ZAC 2026 LifeMate.

Loads and merges configuration from YAML files (llm.yaml, agent.yaml, runtime.yaml)
with environment variable overrides and sensible defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Loads a YAML file from path or repo root."""
    p = Path(path)
    if not p.is_file():
        alt = REPO_ROOT / path
        if alt.is_file():
            p = alt
        else:
            return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_llm_config(config_path: str | Path = "configs/llm.yaml") -> dict[str, Any]:
    """Returns merged LLM configuration."""
    raw = load_yaml(config_path).get("llm", {})
    return {
        "provider": os.environ.get("LLM_PROVIDER") or raw.get("provider", "gemini"),
        "model": os.environ.get("GEMINI_MODEL") or os.environ.get("OPENAI_MODEL") or raw.get("model", "gemini-2.5-flash"),
        "temperature": float(raw.get("temperature", 0.1)),
        "max_output_tokens": int(raw.get("max_output_tokens", 1024)),
        "timeout_seconds": int(raw.get("timeout_seconds", 30)),
        "max_retries": int(raw.get("max_retries", 3)),
        "api_key_env": raw.get("api_key_env", "GEMINI_API_KEY"),
        "base_url": os.environ.get("OPENAI_BASE_URL") or raw.get("base_url"),
    }


def get_agent_config(config_path: str | Path = "configs/agent.yaml") -> dict[str, Any]:
    """Returns merged Agent configuration."""
    raw = load_yaml(config_path).get("agent", {})
    return {
        "max_steps": int(os.environ.get("MAX_AGENT_STEPS") or raw.get("max_steps", 8)),
        "strict_validation": bool(raw.get("strict_validation", True)),
        "max_schema_retries": int(raw.get("max_schema_retries", 2)),
        "auto_repair_args": bool(raw.get("auto_repair_args", True)),
        "grounding_enabled": bool(raw.get("grounding_enabled", True)),
        "stop_on_rejected": bool(raw.get("stop_on_rejected", True)),
    }


def get_runtime_config(config_path: str | Path = "configs/runtime.yaml") -> dict[str, Any]:
    """Returns merged Runtime configuration."""
    raw = load_yaml(config_path)
    nested = raw.get("runtime", {})
    merged = {**raw, **nested}
    return {
        "device": os.environ.get("DEVICE") or merged.get("device", "auto"),
        "precision": merged.get("precision", "auto"),
        "batch_size": int(merged.get("batch_size", 1)),
        "max_tool_calls": int(os.environ.get("MAX_TOOL_CALLS") or merged.get("max_tool_calls", 12)),
        "num_workers": int(merged.get("num_workers", 2)),
        "max_timeout_per_sample_seconds": float(os.environ.get("TIMEOUT_PER_SAMPLE") or merged.get("max_timeout_per_sample_seconds", 10.0)),
        "log_level": os.environ.get("LOG_LEVEL") or merged.get("log_level", "INFO"),
        "output_dir": os.environ.get("OUTPUT_DIR") or merged.get("output_dir", "result"),
    }

