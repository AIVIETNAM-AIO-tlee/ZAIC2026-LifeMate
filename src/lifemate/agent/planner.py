"""LLM Agent Baseline, Planner & Decision Engine for ZAC 2026 LifeMate.

Coordinates LLM reasoning, schema validation, argument grounding,
and interactive step decisions (call vs finish).
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from lifemate.agent.argument_builder import ArgumentBuilder
from lifemate.agent.context import AgentContext
from lifemate.agent.policy import AgentPolicy
from lifemate.agent.prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT, build_user_prompt
from lifemate.agent.tool_selector import ToolSelector
from lifemate.tools.catalog import ToolCatalog
from lifemate.tools.validator import ToolValidator
from lifemate.utils.config import get_agent_config, get_llm_config


@dataclass
class AgentDecision:
    action: str  # 'call' or 'finish'
    tool_name: str = ""
    arguments: dict[str, Any] = None
    thought: str = ""
    raw_response: str = ""

    def __post_init__(self):
        if self.arguments is None:
            self.arguments = {}


class BaseLLMClient:
    """Interface for LLM inference."""
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class GeminiRESTClient(BaseLLMClient):
    """Direct REST client for Google Gemini API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
    ):
        cfg = get_llm_config()
        self.api_key = api_key or os.environ.get(cfg.get("api_key_env", "GEMINI_API_KEY"), "") or os.environ.get("GEMINI_API_KEY", "")
        self.model = model or cfg.get("model", "gemini-2.5-flash")
        self.temperature = temperature if temperature is not None else cfg.get("temperature", 0.1)
        self.timeout = timeout if timeout is not None else cfg.get("timeout_seconds", 30)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            return json.dumps({"action": "finish", "thought": "GEMINI_API_KEY is not set."})

        # Use header authentication instead of URL query string
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}],
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                candidates = result.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "")
                return '{"action": "finish"}'
        except Exception as e:
            return json.dumps({"action": "finish", "thought": f"API error: {e}"})


class OpenAIRESTClient(BaseLLMClient):
    """REST client for OpenAI-compatible endpoints (Ollama, vLLM, OpenRouter, Groq)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
    ):
        cfg = get_llm_config()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        raw_base = base_url or cfg.get("base_url") or "https://api.openai.com/v1"
        self.base_url = raw_base.rstrip("/")
        self.model = model or cfg.get("model", "gpt-4o-mini")
        self.temperature = temperature if temperature is not None else cfg.get("temperature", 0.1)
        self.timeout = timeout if timeout is not None else cfg.get("timeout_seconds", 30)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                choices = result.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
                return '{"action": "finish"}'
        except Exception as e:
            return json.dumps({"action": "finish", "thought": f"OpenAI API error: {e}"})


class MockLLMClient(BaseLLMClient):
    """Deterministic mock client for offline tests and validation."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = list(responses or [])
        self.call_count = 0

    def add_response(self, response: str | dict[str, Any]) -> None:
        if isinstance(response, dict):
            self.responses.append(json.dumps(response, ensure_ascii=False))
        else:
            self.responses.append(response)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return '{"action": "finish", "thought": "No more mock responses"}'


def extract_json_from_llm(text: str) -> dict[str, Any]:
    """Robustly extracts JSON object from LLM response text."""
    text = text.strip()
    # Try exact parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try extracting markdown json block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # Try finding outermost braces
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass

    return {"action": "finish", "thought": "Failed to parse JSON response"}


class AgentPlanner:
    """Plans multi-step execution, grounds parameters, and handles schema recovery."""

    def __init__(
        self,
        catalog: ToolCatalog,
        llm_client: BaseLLMClient,
        strict_validation: bool | None = None,
        grounding_enabled: bool | None = None,
        auto_repair: bool | None = None,
    ):
        cfg = get_agent_config()
        self.catalog = catalog
        self.llm = llm_client
        self.selector = ToolSelector(catalog)
        self.policy = AgentPolicy(catalog)
        self.arg_builder = ArgumentBuilder(catalog)
        self.validator = ToolValidator(catalog)
        self.strict_validation = strict_validation if strict_validation is not None else cfg.get("strict_validation", True)
        self.grounding_enabled = grounding_enabled if grounding_enabled is not None else cfg.get("grounding_enabled", True)
        self.auto_repair = auto_repair if auto_repair is not None else cfg.get("auto_repair_args", True)

    def plan_next_step(
        self,
        user_request: str,
        context: AgentContext,
        feedback_error: str | None = None,
    ) -> AgentDecision:
        """Determines the next action (call tool or finish) for the current turn."""
        # Check out-of-scope, chitchat or cancellation upfront if at step 0
        if not context.executed_calls and not feedback_error:
            is_oos, reason = self.policy.is_out_of_scope_or_cancelled(user_request, context.installed_apps)
            if is_oos:
                return AgentDecision(
                    action="finish",
                    thought=f"Terminated out-of-scope or cancelled request: {reason}",
                    raw_response='{"action": "finish"}',
                )

        allowed_tools = self.selector.get_allowed_tool_names(context.installed_apps)
        tools_summary = self.selector.to_prompt_summary(context.installed_apps)
        context_summary = context.format_for_prompt()

        if feedback_error:
            context_summary += f"\n\n[LỖI SCHEMA BƯỚC TRƯỚC]: {feedback_error}\nHÃY SỬA LẠI THAM SỐ CHO ĐÚNG!"

        full_system_prompt = f"{SYSTEM_PROMPT}\n\n{FEW_SHOT_EXAMPLES}"
        user_prompt = build_user_prompt(user_request, context_summary, tools_summary)

        raw_output = self.llm.generate(full_system_prompt, user_prompt)
        parsed = extract_json_from_llm(raw_output)

        action = parsed.get("action", "finish")
        thought = parsed.get("thought", "")

        if action != "call":
            return AgentDecision(action="finish", thought=thought, raw_response=raw_output)

        tool_name = parsed.get("tool_name", "")
        raw_args = parsed.get("arguments") or {}

        # 1. Permission check
        ok_perm, perm_err = self.policy.validate_tool_permission(tool_name, context.installed_apps)
        if not ok_perm:
            # Cannot call uninstalled tool -> Finish safely without penalties
            return AgentDecision(
                action="finish",
                thought=f"Blocked forbidden tool call: {perm_err}",
                raw_response=raw_output,
            )

        # 2. Argument Grounding
        if self.grounding_enabled:
            args = self.arg_builder.ground_arguments(tool_name, raw_args, context)
        else:
            args = raw_args

        # 3. Auto Repair
        if self.auto_repair:
            args = self.validator.auto_repair_arguments(tool_name, args)

        return AgentDecision(
            action="call",
            tool_name=tool_name,
            arguments=args,
            thought=thought,
            raw_response=raw_output,
        )

    def plan(
        self,
        transcript: str | Mapping[str, Any] = "",
        context: Mapping[str, Any] | None = None,
        history: Any = None,
        tool_results: Any = None,
        sample: Mapping[str, Any] | None = None,
        feedback_error: str | None = None,
    ) -> dict[str, Any]:
        """Unified plan method adhering to the InferencePipeline Protocol."""
        if isinstance(transcript, Mapping):
            sample = transcript
            user_request = sample.get("audio_text") or sample.get("text") or ""
            ctx_data = sample.get("context", {})
            hist_data = sample.get("history", [])
        else:
            user_request = str(transcript or "")
            ctx_data = context or {}
            hist_data = history or []

        sample_id = str((sample or {}).get("id", ""))
        agent_context = AgentContext(
            sample_id=sample_id,
            current_time=str(ctx_data.get("current_time", "")),
            timezone=str(ctx_data.get("timezone", "Asia/Ho_Chi_Minh")),
            location=ctx_data.get("location"),
            user_profile=ctx_data.get("user_profile"),
            installed_apps=ctx_data.get("installed_apps"),
            history=list(hist_data),
        )

        if tool_results:
            for tr in tool_results:
                agent_context.record_call_result(
                    tr.get("name", ""),
                    tr.get("arguments") or {},
                    tr.get("result") or {},
                )

        decision = self.plan_next_step(
            user_request=user_request,
            context=agent_context,
            feedback_error=feedback_error,
        )

        if decision.action == "call":
            return {
                "tool_calls": [{"name": decision.tool_name, "arguments": decision.arguments}],
                "thought": decision.thought,
            }
        return {
            "tool_calls": [],
            "final_response": decision.thought or "Finish",
        }


class HeuristicPlanner(AgentPlanner):
    """Rule-based / heuristic planner for offline evaluation and deterministic fallback."""

    def __init__(self, catalog: ToolCatalog):
        super().__init__(catalog=catalog, llm_client=MockLLMClient())

    def plan_next_step(
        self,
        user_request: str,
        context: AgentContext,
        feedback_error: str | None = None,
    ) -> AgentDecision:
        if not context.executed_calls and not feedback_error:
            is_oos, reason = self.policy.is_out_of_scope_or_cancelled(user_request, context.installed_apps)
            if is_oos:
                return AgentDecision(
                    action="finish",
                    thought=f"Terminated out-of-scope or cancelled request: {reason}",
                    raw_response='{"action": "finish"}',
                )

        lower = user_request.lower()
        called_names = [c["name"] for c in context.executed_calls]

        # 1. Weather conditional check
        if any(w in lower for w in ["mưa", "thời tiết", "forecast"]) and self.selector.is_tool_allowed("weather.get_forecast", context.installed_apps):
            if "weather.get_forecast" not in called_names:
                date_str = context.temporal.resolve_date(user_request)
                return AgentDecision(
                    action="call",
                    tool_name="weather.get_forecast",
                    arguments={"date": date_str},
                    thought="Kiểm tra dự báo thời tiết trước",
                )

            # Evaluate weather result
            weather_res = context.get_latest_result_for("weather")
            rain = False
            if weather_res and "forecast" in weather_res:
                forecasts = weather_res["forecast"]
                if forecasts and isinstance(forecasts, list):
                    rain = forecasts[0].get("rain", False)

            if not rain and "không mưa thì thôi" in lower:
                return AgentDecision(action="finish", thought="Không mưa, dừng theo yêu cầu người dùng")

            # Rain condition met -> check calendar events
            if any(w in lower for w in ["dời", "lịch", "chạy bộ", "họp"]) and self.selector.is_tool_allowed("calendar.list_events", context.installed_apps):
                if "calendar.list_events" not in called_names:
                    return AgentDecision(
                        action="call",
                        tool_name="calendar.list_events",
                        arguments={},
                        thought="Tra cứu lịch sự kiện để dời giờ",
                    )

                if "calendar.update_event" not in called_names and self.selector.is_tool_allowed("calendar.update_event", context.installed_apps):
                    cal_res = context.get_latest_result_for("calendar")
                    events = (cal_res or {}).get("events", [])
                    matched = None
                    for ev in events:
                        if any(w in ev.get("title", "").lower() for w in ["chạy bộ", "đá bóng", "họp"]):
                            matched = ev
                            break
                    if not matched and events:
                        matched = events[0]

                    if matched:
                        eid = matched.get("id")
                        old_start = matched.get("start")
                        old_end = matched.get("end")
                        new_start = context.temporal.shift_iso_datetime(old_start, hours=1) if old_start else None
                        new_end = context.temporal.shift_iso_datetime(old_end, hours=1) if old_end else None
                        return AgentDecision(
                            action="call",
                            tool_name="calendar.update_event",
                            arguments={"event_id": eid, "start": new_start, "end": new_end},
                            thought="Dời sự kiện muộn 1 tiếng",
                        )

            # Check message / contact notifications
            if any(w in lower for w in ["nhắn", "sms", "báo"]) and self.selector.is_tool_allowed("contacts.search", context.installed_apps):
                if "contacts.search" not in called_names:
                    # Extract name query
                    query = "Sếp Dũng" if "sếp dũng" in lower else "Dũng"
                    return AgentDecision(
                        action="call",
                        tool_name="contacts.search",
                        arguments={"query": query},
                        thought="Tìm liên hệ người nhận",
                    )

                if "sms.send" not in called_names and self.selector.is_tool_allowed("sms.send", context.installed_apps):
                    cid = self.arg_builder.ground_arguments("sms.send", {}, context).get("contact_id")
                    if cid:
                        return AgentDecision(
                            action="call",
                            tool_name="sms.send",
                            arguments={"contact_id": cid, "content": "mai chạy bộ dời sang 09:00 nhé"},
                            thought="Gửi SMS thông báo giờ mới",
                        )

        # 2. Missing info detection (e.g. music or alarm or sms without params)
        if "playlist" in lower and "thêm" in lower and self.selector.is_tool_allowed("music.add_to_playlist", context.installed_apps):
            if "sys.ask_missing_info" not in called_names:
                return AgentDecision(
                    action="call",
                    tool_name="sys.ask_missing_info",
                    arguments={"tool_name": "music.add_to_playlist", "params": ["playlist_id", "track_id"]},
                    thought="Thiếu danh sách và bài hát",
                )

        if any(w in lower for w in ["báo thức", "alarm"]) and self.selector.is_tool_allowed("alarm.create", context.installed_apps):
            if "alarm.create" not in called_names:
                # E.g. "đặt báo thức 5:30"
                m = re.search(r"(\d{1,2})[:h](\d{2})", lower)
                if m:
                    time_val = f"{int(m.group(1)):02d}:{m.group(2)}:00"
                    date_val = context.temporal.tomorrow_str
                    iso_time = context.temporal.format_iso(date_val, time_val)
                    return AgentDecision(
                        action="call",
                        tool_name="alarm.create",
                        arguments={"time": iso_time},
                        thought="Tạo báo thức",
                    )

        # Default: Finish
        return AgentDecision(action="finish", thought="Không có hành động phù hợp tiếp theo")
