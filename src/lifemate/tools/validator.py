"""Schema Validation (Pre-execution Defense) for ZAC 2026 LifeMate.

Validates tool calls before execution against catalog schema and sample installed_apps
to ensure zero rejected calls in the final submission.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Any
from lifemate.tools.catalog import ToolCatalog

EMPTY_VALUES = (None, "", [], {})
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_DATETIME_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$")
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")


@dataclass
class ValidationResult:
    ok: bool
    error: str | None = None


def check_is_type(value: Any, expected_type: str | None) -> bool:
    if expected_type is None:
        return True
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    return True


def check_format(value: Any, fmt: str) -> bool:
    if not isinstance(value, str):
        return False
    if fmt == "date":
        m = _DATE_RE.match(value)
        if not m:
            return False
        try:
            date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return True
        except ValueError:
            return False
    elif fmt == "date-time":
        m = _DATETIME_RE.match(value)
        if not m:
            return False
        try:
            date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            hh, mm = value[11:13], value[14:16]
            return int(hh) < 24 and int(mm) < 60
        except Exception:
            return False
    elif fmt == "time":
        if not _TIME_RE.match(value):
            return False
        parts = value.split(":")
        return int(parts[0]) < 24 and int(parts[1]) < 60
    return True


def check_value_error(key: str, val: Any, prop_spec: dict[str, Any]) -> str | None:
    expected_type = prop_spec.get("type")
    if not check_is_type(val, expected_type):
        return f"{key} phải là {expected_type}"

    if prop_spec.get("enum") and not any(
        x == val and isinstance(x, bool) == isinstance(val, bool) for x in prop_spec["enum"]
    ):
        return f"{key} phải là một trong {prop_spec['enum']}"

    if expected_type in ("integer", "number"):
        if isinstance(val, float) and not math.isfinite(val):
            return f"{key} phải là số hữu hạn"
        if "minimum" in prop_spec and val < prop_spec["minimum"]:
            return f"{key} phải >= {prop_spec['minimum']}"
        if "exclusiveMinimum" in prop_spec and val <= prop_spec["exclusiveMinimum"]:
            return f"{key} phải > {prop_spec['exclusiveMinimum']}"
        if "maximum" in prop_spec and val > prop_spec["maximum"]:
            return f"{key} phải <= {prop_spec['maximum']}"
        if "exclusiveMaximum" in prop_spec and val >= prop_spec["exclusiveMaximum"]:
            return f"{key} phải < {prop_spec['exclusiveMaximum']}"

    if prop_spec.get("format") and not check_format(val, prop_spec["format"]):
        hint = {
            "date": " (YYYY-MM-DD)",
            "date-time": " (ISO-8601, ví dụ 2026-11-16T17:00:00+07:00)",
            "time": " (HH:MM)",
        }.get(prop_spec["format"], "")
        return f"{key} phải có dạng {prop_spec['format']}{hint}"

    if expected_type == "array" and isinstance(prop_spec.get("items"), dict):
        item_spec = prop_spec["items"]
        for it in val:
            err = check_value_error(f"{key}[]", it, item_spec)
            if err:
                return err

    return None


class ToolValidator:
    """Validates tool calls before execution against catalog schema and sample installed_apps."""

    def __init__(self, catalog: ToolCatalog):
        self.catalog = catalog

    def validate_call(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        allowed_tools: set[str] | list[str] | None = None,
    ) -> ValidationResult:
        """Validates tool call name and arguments against schema and permissions."""
        if not isinstance(name, str) or not self.catalog.has_tool(name):
            return ValidationResult(ok=False, error=f"tool '{name}' không có trong catalog")

        if allowed_tools is not None:
            allowed_set = set(allowed_tools) | {"sys.ask_missing_info"}
            if name not in allowed_set:
                return ValidationResult(ok=False, error=f"tool '{name}' không thuộc app đã cài")

        tool_spec = self.catalog.get_tool(name)
        if not tool_spec:
            return ValidationResult(ok=False, error=f"tool '{name}' không có trong catalog")

        args = arguments if isinstance(arguments, dict) else {}
        params_spec = tool_spec.parameters
        props = params_spec.get("properties") or {}

        # 1. Required parameters
        missing = [k for k in params_spec.get("required", []) if args.get(k) in EMPTY_VALUES]
        if missing:
            return ValidationResult(ok=False, error="thiếu tham số bắt buộc: " + ", ".join(missing))

        # 2. x-one-of groups
        groups = params_spec.get("x-one-of") or []
        if groups and not any(all(args.get(k) not in EMPTY_VALUES for k in g) for g in groups):
            return ValidationResult(
                ok=False,
                error="cần một trong: " + " / ".join(" + ".join(g) for g in groups),
            )

        # 3. Parameter values & formats
        bad_params = [
            err
            for k, p in props.items()
            if args.get(k) not in EMPTY_VALUES
            for err in [check_value_error(k, args[k], p)]
            if err
        ]
        if bad_params:
            return ValidationResult(ok=False, error="sai tham số: " + "; ".join(bad_params))

        return ValidationResult(ok=True, error=None)

    def auto_repair_arguments(
        self,
        name: str,
        arguments: dict[str, Any],
        default_timezone_offset: str = "+07:00",
    ) -> dict[str, Any]:
        """Performs non-destructive auto-repair for common LLM type formatting issues."""
        tool_spec = self.catalog.get_tool(name)
        if not tool_spec or not isinstance(arguments, dict):
            return arguments

        repaired = dict(arguments)
        props = tool_spec.parameters.get("properties", {})

        for k, p in props.items():
            if k not in repaired:
                continue
            val = repaired[k]
            ptype = p.get("type")
            pfmt = p.get("format")

            # String to integer conversion
            if ptype == "integer" and isinstance(val, str) and val.strip().isdigit():
                repaired[k] = int(val.strip())

            # String to float conversion
            elif ptype == "number" and isinstance(val, str):
                try:
                    repaired[k] = float(val.strip())
                except ValueError:
                    pass

            # Fix missing timezone on date-time
            elif pfmt == "date-time" and isinstance(val, str):
                if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$", val):
                    repaired[k] = val + default_timezone_offset

        return repaired

