"""Connect speech recognition, an injected agent planner, and the local tool runtime."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

from lifemate.speech.pipeline import SpeechPipeline


class ToolEnvironment(Protocol):
    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...
    def get_tool_calls(self) -> list[dict[str, Any]]: ...
    def get_state_hash(self) -> str: ...


class AgentPlanner(Protocol):
    def plan(
        self,
        transcript: str,
        context: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        tool_results: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any] | Sequence[Mapping[str, Any]]: ...


class InferencePipeline:
    """Run one sample in one environment; no environment is created or reset here."""

    def __init__(self, speech: SpeechPipeline, max_tool_calls: int = 12) -> None:
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be positive")
        self.speech = speech
        self.max_tool_calls = max_tool_calls

    def run(
        self,
        audio: str | Path,
        context: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        agent: AgentPlanner,
        tool_env: ToolEnvironment,
        sample_id: str | None = None,
    ) -> dict[str, Any]:
        transcript = self.speech.transcribe(audio, context=context, history=list(history))
        results: list[dict[str, Any]] = []
        final_response = None
        while True:
            plan = agent.plan(
                transcript=transcript,
                context=context,
                history=history,
                tool_results=deepcopy(results),
            )
            if isinstance(plan, Mapping):
                calls = plan.get("tool_calls", plan.get("calls", []))
                final_response = plan.get("final_response", plan.get("response", final_response))
            else:
                calls = plan
            if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
                raise TypeError("Agent plan must contain a sequence of tool calls")
            if not calls:
                break
            for call in calls:
                if not isinstance(call, Mapping):
                    raise TypeError("Each agent tool call must be a mapping")
                name = call.get("name")
                arguments = call.get("arguments", {})
                if not isinstance(name, str) or not name:
                    raise ValueError("Each agent tool call must have a non-empty name")
                if not isinstance(arguments, Mapping):
                    raise TypeError(f"Tool arguments for {name!r} must be a mapping")
                if len(results) >= self.max_tool_calls:
                    raise RuntimeError(f"Agent exceeded the limit of {self.max_tool_calls} tool calls")
                result = tool_env.call(name, dict(arguments))
                results.append({"name": name, "arguments": dict(arguments), "result": result})
        return {
            "id": sample_id,
            "transcript": transcript,
            "tool_calls": tool_env.get_tool_calls(),
            "tool_results": results,
            "state_hash": tool_env.get_state_hash(),
            "final_response": final_response,
        }