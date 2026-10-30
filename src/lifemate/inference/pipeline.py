"""End-to-End Inference Pipeline for ZAC 2026 LifeMate (Agent & Speech).

Supports:
1. Audio + Speech + Agent interactive tool loop per sample (Task B10)
2. Dataset batch inference streaming to submission.jsonl (Task A17)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import logging
from pathlib import Path
from typing import Any, Protocol

from lifemate.inference.submission import SubmissionRecord, SubmissionWriter
from lifemate.speech.pipeline import SpeechPipeline
from lifemate.tools.catalog import ToolCatalog
from lifemate.tools.executor import InteractiveExecutor, get_tool_env
from lifemate.utils.logging import setup_logger
from lifemate.utils.timing import Timer


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
    """End-to-end inference pipeline supporting single-sample execution and dataset streaming."""

    def __init__(
        self,
        speech_or_catalog: Any = None,
        planner: Any = None,
        speech: SpeechPipeline | None = None,
        catalog: ToolCatalog | None = None,
        max_tool_calls: int = 12,
        max_steps: int = 8,
        logger: logging.Logger | None = None,
    ) -> None:
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be positive")

        self.speech = speech
        self.catalog = catalog
        self.planner = planner
        self.max_tool_calls = max_tool_calls
        self.max_steps = max_steps
        self.logger = logger or setup_logger("inference_pipeline")

        if speech_or_catalog is not None:
            if hasattr(speech_or_catalog, "transcribe") or isinstance(speech_or_catalog, SpeechPipeline):
                self.speech = speech_or_catalog
            elif isinstance(speech_or_catalog, ToolCatalog) or hasattr(speech_or_catalog, "get_tool"):
                self.catalog = speech_or_catalog

        if self.catalog is not None and self.planner is not None:
            self.executor = InteractiveExecutor(
                planner=self.planner,
                catalog=self.catalog,
                max_steps=self.max_steps,
                speech=self.speech,
            )
        else:
            self.executor = None

    def run_sample(
        self,
        audio: str | Path | None = None,
        context: Mapping[str, Any] | None = None,
        history: Sequence[Mapping[str, Any]] | None = None,
        agent: Any = None,
        tool_env: ToolEnvironment | None = None,
        sample_id: str | None = None,
        transcript: str | None = None,
    ) -> dict[str, Any]:
        """Execute interactive agent loop on a single sample."""
        context = context or {}
        history = list(history or [])

        # Transcribe audio if speech pipeline and audio are available
        if transcript is None:
            if audio is not None and self.speech is not None:
                transcript = self.speech.transcribe(audio, context=context, history=history)
            else:
                transcript = ""

        active_agent = agent or self.planner
        if active_agent is None:
            raise ValueError("No agent planner provided for run_sample")

        if tool_env is None:
            raise ValueError("No tool_env provided for run_sample")

        results: list[dict[str, Any]] = []
        final_response = None

        while True:
            try:
                plan = active_agent.plan(
                    transcript=transcript,
                    context=context,
                    history=history,
                    tool_results=deepcopy(results),
                )
            except TypeError:
                sample_dict = {
                    "id": sample_id,
                    "audio_text": transcript,
                    "context": context,
                    "history": history,
                }
                plan = active_agent.plan(sample_dict, tool_results=deepcopy(results))

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

    def run_dataset(
        self,
        data_path: str | Path,
        output_path: str | Path = "result/submission.jsonl",
        split: str = "public_test",
        root: str | Path = "data",
        limit: int = 0,
    ) -> int:
        """Executes inference on all items in data_path and writes output_path."""
        data_file = Path(data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"Input data not found: {data_file}")

        self.logger.info(f"Starting inference pipeline on {data_file} (split={split})...")

        with SubmissionWriter(output_path) as writer:
            with open(data_file, "r", encoding="utf-8") as f:
                with Timer("total_pipeline") as timer:
                    for line_idx, line in enumerate(f, 1):
                        if not line.strip():
                            continue

                        sample = json.loads(line)
                        sample_id = sample.get("id", f"sample_{line_idx}")
                        env = None

                        try:
                            if self.executor is not None:
                                env = get_tool_env(split, sample_id, root=root)
                                res = self.executor.execute_sample(sample, split=split, root=root, env_instance=env)
                                record = SubmissionRecord(
                                    id=sample_id,
                                    tool_calls=res.get("tool_calls", []),
                                    state_hash=res.get("state_hash", ""),
                                )
                            else:
                                env = get_tool_env(split, sample_id, root=root)
                                audio_path = Path(root) / split / sample.get("audio", "") if sample.get("audio") else None
                                res = self.run_sample(
                                    audio=audio_path,
                                    transcript=sample.get("audio_text"),
                                    context=sample.get("context", {}),
                                    history=sample.get("history", []),
                                    tool_env=env,
                                    sample_id=sample_id,
                                )
                                record = SubmissionRecord(
                                    id=sample_id,
                                    tool_calls=res.get("tool_calls", []),
                                    state_hash=res.get("state_hash", ""),
                                )
                        except Exception as e:
                            self.logger.error(f"Error on sample {sample_id}: {e}")
                            calls = []
                            state_hash = ""
                            if env is not None:
                                try:
                                    calls = env.get_tool_calls()
                                    state_hash = env.get_state_hash()
                                except Exception:
                                    pass
                            record = SubmissionRecord(id=sample_id, tool_calls=calls, state_hash=state_hash)

                        writer.write_record(record)

                        if line_idx % 50 == 0:
                            self.logger.info(f"Processed {line_idx} samples...")

                        if limit and line_idx >= limit:
                            break

        total_time = timer.elapsed
        throughput = writer.count / total_time if total_time > 0 else 0
        self.logger.info(
            f"Inference completed: {writer.count} records written to {output_path} "
            f"in {total_time:.2f}s ({throughput:.2f} samples/sec)."
        )
        return writer.count

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Unified run method dispatching to run_sample or run_dataset."""
        if "audio" in kwargs or "tool_env" in kwargs or "agent" in kwargs:
            return self.run_sample(*args, **kwargs)
        if "data_path" in kwargs or (args and isinstance(args[0], (str, Path)) and not str(args[0]).endswith(".wav")):
            return self.run_dataset(*args, **kwargs)
        if args and isinstance(args[0], (str, Path)) and str(args[0]).endswith(".wav"):
            return self.run_sample(*args, **kwargs)
        return self.run_dataset(*args, **kwargs)
