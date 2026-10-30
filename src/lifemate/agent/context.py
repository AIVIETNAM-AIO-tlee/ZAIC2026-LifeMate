"""Context signals that can safely help Vietnamese speech recognition."""
from __future__ import annotations

from collections.abc import Mapping, Sequence


def build_asr_context(
    context: Mapping[str, object] | None = None,
    history: Sequence[Mapping[str, object]] | None = None,
    max_hotwords: int = 64,
) -> dict[str, object]:
    """Collect names and place names as ASR hints without constructing an agent prompt."""
    if max_hotwords < 0:
        raise ValueError("max_hotwords cannot be negative")
    if max_hotwords == 0:
        return {"hotwords": []}
    context = context or {}
    profile = context.get("user_profile")
    profile = profile if isinstance(profile, Mapping) else {}
    candidates: list[str] = []
    for key in ("name", "home_address", "work_address", "city"):
        value = profile.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    for turn in history or ():
        if turn.get("role") != "user":
            continue
        transcript = turn.get("audio_text") or turn.get("text")
        if isinstance(transcript, str) and transcript.strip():
            candidates.append(transcript.strip())

    seen: set[str] = set()
    hotwords: list[str] = []
    for value in candidates:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            hotwords.append(value)
        if len(hotwords) >= max_hotwords:
            break
    return {"hotwords": hotwords}