"""Interactive Tool Executor for ZAC 2026 LifeMate.

Orchestrates the interactive feedback loop between Planner, Pre-Validator,
and ToolEnv, enforcing step limits, error recovery, and submission formatting.
Supports automatic audio transcription via SpeechPipeline when audio_text is missing.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from lifemate.agent.context import AgentContext
from lifemate.agent.planner import AgentPlanner
from lifemate.tools.catalog import ToolCatalog
from lifemate.tools.validator import ToolValidator
from lifemate.utils.config import get_agent_config


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
        max_steps: int | None = None,
        max_schema_retries: int | None = None,
        strict_validation: bool | None = None,
        speech: Any = None,
    ):
        cfg = get_agent_config()
        self.planner = planner
        self.catalog = catalog
        self.validator = ToolValidator(catalog)
        self.max_steps = max_steps if max_steps is not None else cfg.get("max_steps", 8)
        self.max_schema_retries = max_schema_retries if max_schema_retries is not None else cfg.get("max_schema_retries", 2)
        self.strict_validation = strict_validation if strict_validation is not None else cfg.get("strict_validation", True)
        self.speech = speech

    def execute_sample(
        self,
        sample: dict[str, Any],
        split: str = "train",
        root: str | Path = "data",
        env_instance: Any = None,
    ) -> dict[str, Any]:
        """Runs the interactive agent loop on a single sample and returns submission result."""
        sample_id = sample.get("id", "")
        audio_text = sample.get("audio_text")

        # Fallback to speech transcription when audio_text is not provided (e.g. in test sets)
        if not audio_text:
            audio_file = sample.get("audio")
            if audio_file and self.speech is not None:
                audio_path = Path(root) / split / audio_file if not Path(audio_file).is_absolute() else Path(audio_file)
                target = audio_path if audio_path.exists() else audio_file
                try:
                    audio_text = self.speech.transcribe(
                        target,
                        context=sample.get("context"),
                        history=sample.get("history"),
                    )
                except Exception:
                    pass

        audio_text = audio_text or ""
        context = AgentContext.from_sample(sample)

        env = env_instance or get_tool_env(split, sample_id, root=root)
        allowed_tools = self.planner.selector.get_allowed_tool_names(context.installed_apps)

        last_error: str | None = None
        schema_retries = 0
        rejected_calls = 0

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
                        # Reached retry limit; count as rejected call to avoid penalty on env
                        rejected_calls += 1
                        break

            # Valid call: reset error & retry counter, execute on env
            last_error = None
            schema_retries = 0
            tool_result = env.call(tool_name, tool_args)
            if isinstance(tool_result, dict) and not tool_result.get("ok", True):
                rejected_calls += 1
            context.record_call_result(tool_name, tool_args, tool_result)

        return {
            "id": sample_id,
            "transcript": audio_text,
            "tool_calls": env.get_tool_calls(),
            "state_hash": env.get_state_hash(),
            "rejected_calls_count": rejected_calls,
        }
