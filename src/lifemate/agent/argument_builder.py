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

        # 1. Contact ID grounding (x-ref: contacts.contacts)
        if "contact_id" in tool_spec.param_specs or "contacts.contacts" in x_refs.values():
            if not grounded.get("contact_id") and not grounded.get("phone_number"):
                # Try to extract contact_id from prior contacts.search results
                cid = EntityResolver.extract_id_from_tool_results(
                    "contact_id",
                    "contacts.contacts",
                    context.tool_results,
                )
                if cid:
                    grounded["contact_id"] = cid

        # 2. Event ID grounding (x-ref: calendar.events)
        if "event_id" in tool_spec.param_specs or "calendar.events" in x_refs.values():
            if not grounded.get("event_id"):
                eid = EntityResolver.extract_id_from_tool_results(
                    "event_id",
                    "calendar.events",
                    context.tool_results,
                )
                if eid:
                    grounded["event_id"] = eid

        # 3. Device ID grounding (x-ref: smart_home.devices)
        if "device_id" in tool_spec.param_specs or "smart_home.devices" in x_refs.values():
            if not grounded.get("device_id"):
                did = EntityResolver.extract_id_from_tool_results(
                    "device_id",
                    "smart_home.devices",
                    context.tool_results,
                )
                if did:
                    grounded["device_id"] = did

        # 4. Temporal / Date Grounding
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
                        # Extract timezone offset from current_time_str
                        offset = "+07:00"
                        if context.current_time_str and ("+" in context.current_time_str[-6:] or "-" in context.current_time_str[-6:]):
                            offset = context.current_time_str[-6:]
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

