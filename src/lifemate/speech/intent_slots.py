"""Optional adapter for a separately configured intent-and-slot extractor."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


IntentSlotExtractor = Callable[[str, Mapping[str, Any] | None], Mapping[str, Any]]


def extract_intent_slots(
    transcript: str,
    context: Mapping[str, Any] | None = None,
    extractor: IntentSlotExtractor | None = None,
) -> dict[str, Any] | None:
    """Run an injected extractor; return None when slot extraction is not configured."""
    if not transcript.strip():
        raise ValueError("Transcript cannot be empty")
    if extractor is None:
        return None
    result = extractor(transcript, context)
    if not isinstance(result, Mapping):
        raise TypeError("Intent-slot extractor must return a mapping")
    return dict(result)
