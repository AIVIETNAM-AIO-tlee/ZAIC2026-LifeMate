"""Entity & Reference Resolver for ZAC 2026 LifeMate.

Resolves entity IDs (contact_id, event_id, device_id, note_id, alarm_id, etc.)
from tool read results and handles fuzzy string matching for Vietnamese names and aliases.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any


def remove_accents(text: str) -> str:
    """Removes Vietnamese tone marks and accents for robust matching."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    return text


def normalize_text(text: str) -> str:
    """Lowercases, removes extra punctuation and normalizes whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fuzzy_similarity(a: str, b: str) -> float:
    """Calculates similarity score between two strings, checking exact, accentless, and sequence similarity."""
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0

    # Substring containment
    if a_norm in b_norm or b_norm in a_norm:
        return 0.9

    a_noacc = remove_accents(a_norm)
    b_noacc = remove_accents(b_norm)
    if a_noacc == b_noacc:
        return 0.95
    if a_noacc in b_noacc or b_noacc in a_noacc:
        return 0.85

    return difflib.SequenceMatcher(None, a_noacc, b_noacc).ratio()


class EntityResolver:
    """Resolves IDs and entities from prior tool results or app states."""

    @staticmethod
    def find_best_contact(query: str, contacts: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Finds the contact matching the query by name, relation, or phone."""
        if not query or not contacts:
            return None

        best_score = 0.0
        best_contact = None

        query_digits = re.sub(r"\D", "", query)

        for c in contacts:
            name = str(c.get("name", ""))
            relation = str(c.get("relation", ""))
            phone = str(c.get("phone", ""))

            # Exact phone match
            if query_digits and query_digits in phone:
                return c

            score_name = fuzzy_similarity(query, name)
            score_rel = fuzzy_similarity(query, relation) if relation else 0.0
            max_s = max(score_name, score_rel)

            if max_s > best_score:
                best_score = max_s
                best_contact = c

        return best_contact if best_score >= 0.55 else None

    @staticmethod
    def find_best_event(
        title_query: str,
        events: list[dict[str, Any]],
        date_filter: str | None = None,
    ) -> dict[str, Any] | None:
        """Finds the calendar event matching the title and optional date filter (YYYY-MM-DD)."""
        if not events:
            return None

        filtered = events
        if date_filter:
            filtered = [
                e for e in events
                if str(e.get("start", "")).startswith(date_filter)
            ] or events

        if not title_query:
            return filtered[0] if len(filtered) == 1 else None

        best_score = 0.0
        best_event = None
        for e in filtered:
            title = str(e.get("title", ""))
            score = fuzzy_similarity(title_query, title)
            if score > best_score:
                best_score = score
                best_event = e

        return best_event if best_score >= 0.5 else None

    @staticmethod
    def find_best_device(
        name_or_room: str,
        devices: list[dict[str, Any]],
        device_type: str | None = None,
    ) -> dict[str, Any] | None:
        """Finds smart home device by device name or room name and optional type."""
        if not devices:
            return None

        candidates = devices
        if device_type:
            candidates = [d for d in devices if d.get("type") == device_type] or devices

        if not name_or_room:
            return candidates[0] if len(candidates) == 1 else None

        best_score = 0.0
        best_dev = None
        for d in candidates:
            dev_id = str(d.get("id", ""))
            room = str(d.get("room", ""))
            dtype = str(d.get("type", ""))

            s_id = fuzzy_similarity(name_or_room, dev_id)
            s_room = fuzzy_similarity(name_or_room, room)
            s_combo = fuzzy_similarity(name_or_room, f"{dtype} {room}")
            score = max(s_id, s_room, s_combo)

            if score > best_score:
                best_score = score
                best_dev = d

        return best_dev if best_score >= 0.5 else None

    @staticmethod
    def extract_id_from_tool_results(
        param_name: str,
        x_ref: str | None,
        tool_results_history: list[dict[str, Any]],
    ) -> str | None:
        """Attempts to auto-ground an ID parameter from the most recent tool results across any app."""
        if not tool_results_history:
            return None

        for result_wrapper in reversed(tool_results_history):
            res = result_wrapper.get("result", {})
            if not isinstance(res, dict) or not res.get("ok", True):
                continue

            # 1. Match by collection name from x_ref (e.g. contacts.contacts, calendar.events, notes.notes)
            if x_ref and "." in x_ref:
                collection_key = x_ref.split(".", 1)[1]
                items = res.get(collection_key)
                if isinstance(items, list) and len(items) == 1 and isinstance(items[0], dict):
                    item_id = items[0].get("id")
                    if item_id:
                        return str(item_id)

            # 2. Match by param_name prefix/plural (e.g. contact_id -> contacts, device_id -> devices)
            clean_name = param_name.replace("_id", "")
            for plural in [f"{clean_name}s", f"{clean_name}es", clean_name]:
                items = res.get(plural)
                if isinstance(items, list) and len(items) == 1 and isinstance(items[0], dict):
                    item_id = items[0].get("id")
                    if item_id:
                        return str(item_id)

            # 3. Direct key match in result dict
            if param_name in res and res[param_name]:
                val = res[param_name]
                if isinstance(val, (str, int)):
                    return str(val)

        return None
