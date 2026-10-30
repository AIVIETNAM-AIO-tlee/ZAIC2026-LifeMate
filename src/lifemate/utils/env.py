"""Environment Variables Loader for ZAC 2026 LifeMate.

Loads configuration from .env files without requiring third-party dependencies,
with graceful fallback to python-dotenv if available.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(env_path: str | Path | None = None, override: bool = False) -> dict[str, str]:
    """Finds and loads a .env file into os.environ.

    Args:
        env_path: Optional path to .env file. If None, searches current dir and parent dirs.
        override: If True, overwrites existing environment variables. Default False.

    Returns:
        dict of loaded key-value pairs.
    """
    loaded: dict[str, str] = {}

    target_path: Path | None = None
    if env_path:
        p = Path(env_path)
        if p.is_file():
            target_path = p
    else:
        # Check current working directory, then repo root
        candidates = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parents[3] / ".env",
            Path(__file__).resolve().parents[2] / ".env",
        ]
        for cand in candidates:
            if cand.is_file():
                target_path = cand
                break

    if not target_path or not target_path.exists():
        return loaded

    # Try python-dotenv first if installed
    try:
        from dotenv import load_dotenv
        load_dotenv(target_path, override=override)
    except ImportError:
        pass

    # Standard built-in fallback parser
    with open(target_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue

            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()

            # Strip matching quotes if wrapped
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]

            loaded[key] = val
            if override or key not in os.environ:
                os.environ[key] = val

    return loaded

