"""Speech-to-text pipeline."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from lifemate.agent.context import build_asr_context
from lifemate.speech.intent_slots import IntentSlotExtractor, extract_intent_slots
from lifemate.speech.normalizer import normalize_text


class Transcriber(Protocol):
    def transcribe(self, audio: str | Path, context: Mapping[str, Any] | None = None) -> str: ...


class SpeechPipeline:
    def __init__(
        self,
        transcriber: Transcriber,
        target_sample_rate: int = 16_000,
        normalize=normalize_text,
    ) -> None:
        if target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive")
        self.transcriber = transcriber
        self.target_sample_rate = target_sample_rate
        self.normalize = normalize

    def transcribe(
        self,
        audio: str | Path,
        context: Mapping[str, Any] | None = None,
        history: list[Mapping[str, Any]] | None = None,
    ) -> str:
        hints = build_asr_context(context, history)
        transcript = self.transcriber.transcribe(audio, context=hints)
        if not isinstance(transcript, str):
            raise TypeError("ASR model must return a string transcript")
        return self.normalize(transcript)

    def transcribe_with_slots(
        self,
        audio: str | Path,
        context: Mapping[str, Any] | None = None,
        history: list[Mapping[str, Any]] | None = None,
        slot_extractor: IntentSlotExtractor | None = None,
    ) -> dict[str, Any]:
        transcript = self.transcribe(audio, context=context, history=history)
        return {
            "transcript": transcript,
            "intent_slots": extract_intent_slots(transcript, context, slot_extractor),
        }