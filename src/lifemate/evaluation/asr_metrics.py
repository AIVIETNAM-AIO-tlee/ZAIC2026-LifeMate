"""ASR text and entity error metrics."""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from lifemate.speech.normalizer import normalize_text


_ENTITY = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b"
    r"|\b\d{1,2}:\d{2}\b"
    r"|\b\d+(?:[.,]\d+)?\s*(?:đồng|vnđ|vnd)\b"
    r"|\b(?:\+?84|0)\d{8,10}\b",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    value = normalize_text(text)
    value = unicodedata.normalize("NFC", value).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE).split())


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref in enumerate(reference, start=1):
        current = [row]
        for column, hyp in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref != hyp),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = _clean(reference).split()
    hypothesis_words = _clean(hypothesis).split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return _edit_distance(reference_words, hypothesis_words) / len(reference_words)


def character_error_rate(reference: str, hypothesis: str) -> float:
    reference_chars = list("".join(_clean(reference).split()))
    hypothesis_chars = list("".join(_clean(hypothesis).split()))
    if not reference_chars:
        return 0.0 if not hypothesis_chars else 1.0
    return _edit_distance(reference_chars, hypothesis_chars) / len(reference_chars)


def extract_entities(text: str, patterns: Iterable[re.Pattern[str]] | None = None) -> list[str]:
    compiled_patterns = tuple(patterns) if patterns is not None else (_ENTITY,)
    matches = (match for pattern in compiled_patterns for match in pattern.finditer(normalize_text(text)))
    return [match.group(0).casefold() for match in matches]


def wer(reference: str, hypothesis: str) -> float:
    return word_error_rate(reference, hypothesis)


def cer(reference: str, hypothesis: str) -> float:
    return character_error_rate(reference, hypothesis)


def entity_error_rate(reference: str, hypothesis: str) -> float:
    reference_entities = extract_entities(reference)
    hypothesis_entities = extract_entities(hypothesis)
    if not reference_entities:
        return 0.0 if not hypothesis_entities else 1.0
    return _edit_distance(reference_entities, hypothesis_entities) / len(reference_entities)


def compute_asr_metrics(references: Iterable[str], hypotheses: Iterable[str]) -> dict[str, float | int]:
    refs, hyps = list(references), list(hypotheses)
    if len(refs) != len(hyps):
        raise ValueError(f"Reference/hypothesis count mismatch: {len(refs)} != {len(hyps)}")
    if not refs:
        raise ValueError("At least one reference/hypothesis pair is required")
    word_errors = sum(_edit_distance(_clean(ref).split(), _clean(hyp).split()) for ref, hyp in zip(refs, hyps))
    char_errors = sum(
        _edit_distance(list("".join(_clean(ref).split())), list("".join(_clean(hyp).split())))
        for ref, hyp in zip(refs, hyps)
    )
    entity_counts = [len(extract_entities(ref)) for ref in refs]
    entity_errors = sum(
        _edit_distance(extract_entities(ref), extract_entities(hyp)) for ref, hyp in zip(refs, hyps)
    )
    word_total = sum(len(_clean(ref).split()) for ref in refs)
    char_total = sum(len("".join(_clean(ref).split())) for ref in refs)
    entity_total = sum(entity_counts)
    return {
        "samples": len(refs),
        "wer": word_errors / word_total if word_total else 0.0,
        "cer": char_errors / char_total if char_total else 0.0,
        "entity_error_rate": entity_errors / entity_total if entity_total else 0.0,
        "entities": entity_total,
    }