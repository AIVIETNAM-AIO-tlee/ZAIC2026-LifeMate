"""Agent Policy & Constraints for ZAC 2026 LifeMate.

Defines rules on tool availability, exceptions for sys.ask_missing_info,
and policies for detecting out-of-scope or missing information scenarios.
"""

from __future__ import annotations

from typing import Any
from lifemate.tools.catalog import ToolCatalog, ToolSpec
from lifemate.agent.tool_selector import ToolSelector

EMPTY_VALUES = (None, "", [], {})


class AgentPolicy:
    """Enforces execution and tool selection policies."""

    def __init__(self, catalog: ToolCatalog):
        self.catalog = catalog
        self.selector = ToolSelector(catalog)

    def is_out_of_scope_or_cancelled(
        self,
        user_request: str,
        installed_apps: dict[str, Any] | list[str] | None,
    ) -> tuple[bool, str]:
        """Checks if request should result in [] (out of scope, chitchat, or cancelled)."""
        return self.selector.detect_out_of_scope(user_request, installed_apps)

    def validate_tool_permission(
        self,
        tool_name: str,
        installed_apps: dict[str, Any] | list[str] | None,
    ) -> tuple[bool, str | None]:
        """Verifies if calling tool_name is permitted under current installed_apps."""
        if tool_name == "sys.ask_missing_info":
            return True, None

        if not self.catalog.has_tool(tool_name):
            return False, f"tool '{tool_name}' không có trong catalog"

        allowed: set[str] = set()
        if isinstance(installed_apps, dict):
            for a_info in installed_apps.values():
                if isinstance(a_info, dict):
                    allowed.update(a_info.get("tools", []))
                elif isinstance(a_info, list):
                    allowed.update(a_info)
        elif isinstance(installed_apps, list):
            allowed.update(installed_apps)

        if tool_name not in allowed:
            return False, f"tool '{tool_name}' không thuộc app đã cài"

        return True, None

    def find_missing_required_params(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> list[str]:
        """Finds any required parameters or x-one-of groups missing from arguments."""
        tool_spec = self.catalog.get_tool(tool_name)
        if not tool_spec:
            return []

        missing: list[str] = []
        for req in tool_spec.required:
            if arguments.get(req) in EMPTY_VALUES:
                missing.append(req)

        # Check x-one-of: at least one complete group must be present
        if tool_spec.x_one_of:
            has_valid_group = any(
                all(arguments.get(p) not in EMPTY_VALUES for p in group)
                for group in tool_spec.x_one_of
            )
            if not has_valid_group:
                # Add first missing parameter from the primary one-of group
                primary_group = tool_spec.x_one_of[0]
                for p in primary_group:
                    if p not in missing and arguments.get(p) in EMPTY_VALUES:
                        missing.append(p)

        return missing

    def should_ask_missing_info(
        self,
        target_tool: str,
        missing_params: list[str],
        has_read_tools_available: bool = False,
    ) -> bool:
        """Determines if the agent should call sys.ask_missing_info."""
        if not missing_params:
            return False

        # If missing param is an ID (e.g. contact_id, event_id) and we have a read tool to query it,
        # we should search first rather than asking the user immediately.
        searchable_params = {"contact_id", "event_id", "device_id", "scene_id", "track_id", "playlist_id"}
        if has_read_tools_available and all(p in searchable_params for p in missing_params):
            return False

        return True

    def create_ask_missing_info_call(self, target_tool: str, params: list[str]) -> dict[str, Any]:
        """Constructs a valid sys.ask_missing_info call dict."""
        return {
            "name": "sys.ask_missing_info",
            "arguments": {
                "tool_name": target_tool,
                "params": list(dict.fromkeys(params)),  # preserve order, unique
            },
        }

