"""Conservative Vietnamese transcript normalization."""
from __future__ import annotations

import re
import unicodedata


_DATE = re.compile(
    r"\b(?:ngày\s+)?(?P<day>\d{1,2})\s+tháng\s+(?P<month>\d{1,2})"
    r"(?:\s+năm\s+(?P<year>\d{4}))?",
    re.IGNORECASE,
)
_TIME = re.compile(
    r"\b(?P<hour>\d{1,2})\s*(?:giờ|h)\s*"
    r"(?:(?P<minute>\d{1,2})\s*(?:phút)?)?\s*"
    r"(?P<half>rưỡi)?\s*(?P<period>sáng|trưa|chiều|tối)?",
    re.IGNORECASE,
)
_MONEY = re.compile(
    r"(?<![\w/])(?P<amount>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>tỷ|tỉ|triệu|tr|nghìn|ngàn|k|đồng|đ|vnđ|vnd)\b"
    r"(?:\s*(?:đồng|đ|vnđ|vnd)\b)?",
    re.IGNORECASE,
)
_PUNCTUATION_SPACING = re.compile(r"\s+([,.;:!?])")
_MULTIPLE_SPACES = re.compile(r"\s+")


def _normalize_date(match: re.Match[str]) -> str:
    day = int(match.group("day"))
    month = int(match.group("month"))
    year = match.group("year")
    if not 1 <= day <= 31 or not 1 <= month <= 12:
        return match.group(0)
    # Output ISO-8601 YYYY-MM-DD when year is available to match tool schema
    if year:
        return f"{int(year):04d}-{month:02d}-{day:02d}"
    return f"{day:02d}/{month:02d}"


def _normalize_time(match: re.Match[str]) -> str:
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or (30 if match.group("half") else 0))
    period = (match.group("period") or "").lower()
    if hour > 23 or minute > 59:
        return match.group(0)
    if period in {"chiều", "tối"} and 1 <= hour <= 11:
        hour += 12
    elif period == "trưa" and 1 <= hour <= 10:
        hour += 12
    return f"{hour:02d}:{minute:02d}"


def _normalize_money(match: re.Match[str]) -> str:
    amount = float(match.group("amount").replace(",", "."))
    unit = match.group("unit").lower()
    multiplier = {
        "tỷ": 1_000_000_000,
        "tỉ": 1_000_000_000,
        "triệu": 1_000_000,
        "tr": 1_000_000,
        "nghìn": 1_000,
        "ngàn": 1_000,
        "k": 1_000,
        "đồng": 1,
        "đ": 1,
        "vnđ": 1,
        "vnd": 1,
    }[unit]
    value = amount * multiplier
    rendered = str(int(value)) if value.is_integer() else f"{value:g}"
    return f"{rendered} đồng"


def normalize_text(text: str, lowercase: bool = False) -> str:
    """Canonicalize common date, time, and currency expressions without rewriting entities."""
    if not isinstance(text, str):
        raise TypeError("Transcript must be a string")
    value = unicodedata.normalize("NFC", text).strip()
    value = _DATE.sub(_normalize_date, value)
    value = _TIME.sub(_normalize_time, value)
    value = _MONEY.sub(_normalize_money, value)
    value = _PUNCTUATION_SPACING.sub(r"\1", value)
    value = _MULTIPLE_SPACES.sub(" ", value)
    return value.strip().lower() if lowercase else value.strip()
