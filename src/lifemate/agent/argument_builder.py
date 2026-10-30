"""Tool Argument Grounding for ZAC 2026 LifeMate.

Grounds tool call arguments using prior read results, context, entity resolution,
and temporal reasoning to eliminate hallucinated IDs and invalid parameter formats.
"""

from __future__ import annotations

import re
from typing import Any
from lifemate.agent.context import AgentContext
from lifemate.tools.catalog import ToolCatalog, ToolSpec
from lifemate.tools.resolver import EntityResolver


class ArgumentBuilder:
    """Grounds, validates and builds argument dictionaries for planned tool calls."""

    def __init__(self, catalog: ToolCatalog):
        self.catalog = catalog

    def ground_arguments(
        self,
        tool_name: str,
        raw_arguments: dict[str, Any],
        context: AgentContext,
    ) -> dict[str, Any]:
        """Grounds and refines raw arguments with real entity IDs from context memory."""
        tool_spec = self.catalog.get_tool(tool_name)
        if not tool_spec:
            return raw_arguments

        grounded = dict(raw_arguments)
        x_refs = tool_spec.get_x_refs()

        # Dynamic ID grounding across all catalog parameters (e.g. contact_id, event_id, device_id, note_id, etc.)
        for p_name in tool_spec.param_specs:
            if p_name.endswith("_id") or p_name in x_refs:
                if not grounded.get(p_name):
                    if p_name == "contact_id" and grounded.get("phone_number"):
                        continue
                    xref_target = x_refs.get(p_name)
                    extracted_id = EntityResolver.extract_id_from_tool_results(
                        p_name,
                        xref_target,
                        context.tool_results,
                    )
                    if extracted_id:
                        grounded[p_name] = extracted_id

        # Temporal / Date Grounding
        for param_name, param_spec in tool_spec.param_specs.items():
            val = grounded.get(param_name)
            if isinstance(val, str):
                if param_spec.format == "date" or param_name == "date":
                    # If value contains relative words like 'mai', resolve it
                    if any(w in val.lower() for w in ["mai", "hôm nay", "mốt", "kia", "hôm qua"]):
                        grounded[param_name] = context.temporal.resolve_date(val)
                elif param_spec.format == "date-time" or param_name in ("start", "end", "time"):
                    # Ensure ISO-8601 timezone offset
                    if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$", val):
                        offset = "+07:00"
                        if context.current_time_str and ("+" in context.current_time_str[-6:] or "-" in context.current_time_str[-6:]):
                            offset = context.current_time_str[-6:]
                        elif context.temporal.tz:
                            now_str = context.temporal.now.strftime("%z")
                            if len(now_str) == 5:
                                offset = f"{now_str[:3]}:{now_str[3:]}"
                        grounded[param_name] = val + offset

        return grounded

    def check_missing_parameters(
        self,
        tool_name: str,
        grounded_arguments: dict[str, Any],
    ) -> list[str]:
        """Identifies mandatory parameters still missing after grounding."""
        tool_spec = self.catalog.get_tool(tool_name)
        if not tool_spec:
            return []

        empty_vals = (None, "", [], {})
        missing: list[str] = []

        for req in tool_spec.required:
            if grounded_arguments.get(req) in empty_vals:
                missing.append(req)

        # Check x-one-of
        if tool_spec.x_one_of:
            has_one = any(
                all(grounded_arguments.get(p) not in empty_vals for p in group)
                for group in tool_spec.x_one_of
            )
            if not has_one:
                # Return the missing parameters of the primary one-of group
                for p in tool_spec.x_one_of[0]:
                    if grounded_arguments.get(p) in empty_vals and p not in missing:
                        missing.append(p)

        return missing
