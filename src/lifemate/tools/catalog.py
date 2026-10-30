"""Tool Catalog Parser for ZAC 2026 LifeMate.

Parses tool catalog JSON files (e.g. train_tools_catalog.json, public_test_tools_catalog.json)
and provides indexing, parameter constraint extraction, and LLM schema formatting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolParamSpec:
    """Metadata and constraints for a single tool parameter."""
    name: str
    type: str = "string"
    description: str = ""
    enum: list[Any] | None = None
    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: float | None = None
    exclusive_maximum: float | None = None
    format: str | None = None
    items: dict[str, Any] | None = None
    x_ref: str | None = None
    x_match: str | None = None

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> ToolParamSpec:
        return cls(
            name=name,
            type=data.get("type", "string"),
            description=data.get("description", ""),
            enum=data.get("enum"),
            minimum=data.get("minimum"),
            maximum=data.get("maximum"),
            exclusive_minimum=data.get("exclusiveMinimum"),
            exclusive_maximum=data.get("exclusiveMaximum"),
            format=data.get("format"),
            items=data.get("items"),
            x_ref=data.get("x-ref"),
            x_match=data.get("x-match"),
        )


@dataclass
class ToolSpec:
    """Schema and constraints for an individual tool."""
    name: str
    app: str
    kind: str  # 'read', 'write', 'meta'
    description: str
    parameters: dict[str, Any]
    required: list[str] = field(default_factory=list)
    x_one_of: list[list[str]] = field(default_factory=list)
    returns: str = ""
    param_specs: dict[str, ToolParamSpec] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolSpec:
        name = data.get("name", "")
        app = data.get("app") or (name.split(".", 1)[0] if "." in name else "")
        kind = data.get("kind", "write")
        description = data.get("description", "")
        params_raw = data.get("parameters") or {}
        required = list(params_raw.get("required") or [])
        x_one_of = list(params_raw.get("x-one-of") or [])
        returns = data.get("returns", "")

        param_specs = {}
        for prop_name, prop_data in params_raw.get("properties", {}).items():
            param_specs[prop_name] = ToolParamSpec.from_dict(prop_name, prop_data)

        return cls(
            name=name,
            app=app,
            kind=kind,
            description=description,
            parameters=params_raw,
            required=required,
            x_one_of=x_one_of,
            returns=returns,
            param_specs=param_specs,
        )

    @property
    def is_read(self) -> bool:
        return self.kind == "read"

    @property
    def is_write(self) -> bool:
        return self.kind == "write"

    @property
    def is_meta(self) -> bool:
        return self.kind == "meta"

    def get_x_refs(self) -> dict[str, str]:
        """Returns {param_name: referenced_resource_key}."""
        return {k: p.x_ref for k, p in self.param_specs.items() if p.x_ref}

    def to_openai_tool(self) -> dict[str, Any]:
        """Converts tool spec to OpenAI function-calling format."""
        clean_properties = {}
        for k, p in self.param_specs.items():
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            if p.format:
                prop["format"] = p.format
            if p.items:
                prop["items"] = p.items
            clean_properties[k] = prop

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": clean_properties,
                    "required": self.required,
                },
            },
        }


class ToolCatalog:
    """Manages all tool schemas across apps in a dataset split."""

    def __init__(
        self,
        schema_version: str = "",
        apps: dict[str, dict[str, Any]] | None = None,
        tools: dict[str, ToolSpec] | None = None,
    ):
        self.schema_version = schema_version
        self.apps = apps or {}
        self.tools = tools or {}

    @classmethod
    def from_file(cls, path: str | Path) -> ToolCatalog:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCatalog:
        schema_version = data.get("schema_version", "")
        apps: dict[str, dict[str, Any]] = {}
        tools: dict[str, ToolSpec] = {}

        # Handle 'apps' array
        for app in data.get("apps", []):
            app_id = app.get("id", "")
            apps[app_id] = {
                "id": app_id,
                "name_vi": app.get("name_vi", ""),
                "group": app.get("group", ""),
                "description": app.get("description", ""),
            }
            for tool_data in app.get("tools", []):
                spec = ToolSpec.from_dict(tool_data)
                tools[spec.name] = spec

        # Handle 'tools' list at root if any
        for tool_data in data.get("tools", []):
            spec = ToolSpec.from_dict(tool_data)
            tools[spec.name] = spec

        # Always ensure sys.ask_missing_info exists in catalog
        if "sys.ask_missing_info" not in tools:
            sys_spec = ToolSpec(
                name="sys.ask_missing_info",
                app="sys",
                kind="meta",
                description="Báo các tham số bắt buộc còn thiếu của một tool",
                parameters={
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Tên tool bị thiếu tham số (ví dụ: sms.send)",
                        },
                        "params": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Danh sách tham số còn thiếu (ví dụ: ['contact_id'])",
                        },
                    },
                    "required": ["tool_name", "params"],
                },
                required=["tool_name", "params"],
                returns="",
                param_specs={
                    "tool_name": ToolParamSpec(name="tool_name", type="string", description="Tên tool bị thiếu tham số"),
                    "params": ToolParamSpec(
                        name="params",
                        type="array",
                        items={"type": "string"},
                        description="Danh sách tham số còn thiếu",
                    ),
                },
            )
            tools["sys.ask_missing_info"] = sys_spec

        return cls(schema_version=schema_version, apps=apps, tools=tools)

    def get_tool(self, name: str) -> ToolSpec | None:
        return self.tools.get(name)

    def has_tool(self, name: str) -> bool:
        return name in self.tools

    def get_app_tools(self, app_id: str) -> list[ToolSpec]:
        return [t for t in self.tools.values() if t.app == app_id]

    def filter_by_names(self, allowed_names: set[str] | list[str]) -> list[ToolSpec]:
        allowed = set(allowed_names)
        return [t for name, t in self.tools.items() if name in allowed]

    def to_openai_tools(self, allowed_names: set[str] | list[str] | None = None) -> list[dict[str, Any]]:
        target_tools = (
            self.filter_by_names(allowed_names)
            if allowed_names is not None
            else list(self.tools.values())
        )
        return [t.to_openai_tool() for t in target_tools]

    def to_compact_prompt_summary(self, allowed_names: set[str] | list[str] | None = None) -> str:
        """Generates a compact, readable Vietnamese prompt description of allowed tools."""
        target_tools = (
            self.filter_by_names(allowed_names)
            if allowed_names is not None
            else list(self.tools.values())
        )
        lines = []
        for t in sorted(target_tools, key=lambda x: (x.app, x.name)):
            req_str = f", bắt buộc: {', '.join(t.required)}" if t.required else ""
            one_of_str = (
                f", cần một trong: {' / '.join(' + '.join(g) for g in t.x_one_of)}"
                if t.x_one_of
                else ""
            )
            lines.append(f"- `{t.name}` ({t.kind}): {t.description}{req_str}{one_of_str}")
        return "\n".join(lines)

