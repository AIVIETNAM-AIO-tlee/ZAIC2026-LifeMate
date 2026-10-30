"""Interactive Tool Executor for ZAC 2026 LifeMate.

Orchestrates the interactive feedback loop between Planner, Pre-Validator,
and ToolEnv, enforcing step limits, error recovery, and submission formatting.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from lifemate.agent.context import AgentContext
from lifemate.agent.planner import AgentPlanner
from lifemate.tools.catalog import ToolCatalog
from lifemate.tools.validator import ToolValidator


def get_tool_env(split: str, question_id: str, root: str | Path = "data"):
    """Loads ToolEnv from the split's zaic_tools package."""
    root_path = Path(root).resolve()
    split_dir = root_path / split
    if str(split_dir) not in sys.path:
        sys.path.insert(0, str(split_dir))

    from zaic_tools import ToolEnv
    return ToolEnv(split, question_id, root=root_path)


class InteractiveExecutor:
    """Coordinates multi-step tool execution on a ToolEnv session."""

    def __init__(
        self,
        planner: AgentPlanner,
        catalog: ToolCatalog,
        max_steps: int = 8,
        max_schema_retries: int = 2,
        strict_validation: bool = True,
    ):
        self.planner = planner
        self.catalog = catalog
        self.validator = ToolValidator(catalog)
        self.max_steps = max_steps
        self.max_schema_retries = max_schema_retries
        self.strict_validation = strict_validation

    def execute_sample(
        self,
        sample: dict[str, Any],
        split: str = "train",
        root: str | Path = "data",
        env_instance: Any = None,
    ) -> dict[str, Any]:
        """Runs the interactive agent loop on a single sample and returns submission result."""
        sample_id = sample.get("id", "")
        audio_text = sample.get("audio_text", "")
        context = AgentContext.from_sample(sample)

        env = env_instance or get_tool_env(split, sample_id, root=root)
        allowed_tools = self.planner.selector.get_allowed_tool_names(context.installed_apps)

        last_error: str | None = None
        schema_retries = 0

        for step in range(self.max_steps):
            decision = self.planner.plan_next_step(
                user_request=audio_text,
                context=context,
                feedback_error=last_error,
            )

            if decision.action != "call":
                # Agent decided to finish
                break

            tool_name = decision.tool_name
            tool_args = decision.arguments

            # Pre-execution validation check
            if self.strict_validation:
                val_result = self.validator.validate_call(
                    name=tool_name,
                    arguments=tool_args,
                    allowed_tools=allowed_tools,
                )
                if not val_result.ok:
                    if schema_retries < self.max_schema_retries:
                        schema_retries += 1
                        last_error = val_result.error
                        # Give LLM a chance to self-correct without executing illegal call
                        continue
                    else:
                        # Reached retry limit; do not call illegal tool to avoid penalty
                        break

            # Valid call: reset error & retry counter, execute on env
            last_error = None
            schema_retries = 0
            tool_result = env.call(tool_name, tool_args)
            context.record_call_result(tool_name, tool_args, tool_result)

        return {
            "id": sample_id,
            "tool_calls": env.get_tool_calls(),
            "state_hash": env.get_state_hash(),
        }

