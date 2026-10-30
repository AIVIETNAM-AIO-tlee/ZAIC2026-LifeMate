"""Context & History Reasoning for ZAC 2026 LifeMate.

Manages temporal reasoning (Vietnamese relative date/time expressions),
location, user profile, dialogue history, and intermediate tool execution results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None


@dataclass
class UserProfile:
    name: str = ""
    home_address: str = ""
    work_address: str = ""
    city: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> UserProfile:
        data = data or {}
        return cls(
            name=data.get("name", ""),
            home_address=data.get("home_address", ""),
            work_address=data.get("work_address", ""),
            city=data.get("city", ""),
        )


@dataclass
class UserLocation:
    lat: float | None = None
    lng: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> UserLocation:
        data = data or {}
        return cls(lat=data.get("lat"), lng=data.get("lng"))


class TemporalReasoner:
    """Calculates absolute ISO dates and datetimes from Vietnamese relative expressions."""

    def __init__(self, current_time_str: str, timezone_str: str = "Asia/Ho_Chi_Minh"):
        self.timezone_str = timezone_str or "Asia/Ho_Chi_Minh"
        
        # Resolve tzinfo robustly without requiring external tzdata package on Windows
        self.tz = None
        if current_time_str:
            try:
                parsed = datetime.fromisoformat(current_time_str)
                if parsed.tzinfo is not None:
                    self.tz = parsed.tzinfo
            except Exception:
                pass

        if self.tz is None and ZoneInfo is not None:
            try:
                self.tz = ZoneInfo(self.timezone_str)
            except Exception:
                pass

        if self.tz is None:
            # Standard UTC+7 fallback for Vietnam time
            self.tz = timezone(timedelta(hours=7))

        if current_time_str:
            # Handle ISO format
            self.now = datetime.fromisoformat(current_time_str)
            if self.now.tzinfo is None:
                self.now = self.now.replace(tzinfo=self.tz)
        else:
            self.now = datetime.now(self.tz)

    @property
    def today_str(self) -> str:
        """YYYY-MM-DD for today."""
        return self.now.strftime("%Y-%m-%d")

    @property
    def tomorrow_str(self) -> str:
        """YYYY-MM-DD for tomorrow."""
        return (self.now + timedelta(days=1)).strftime("%Y-%m-%d")

    def resolve_date(self, text: str) -> str:
        """Extracts and resolves date to YYYY-MM-DD from Vietnamese text."""
        lower = text.lower()
        if "hôm nay" in lower or "nay" in lower and "hôm" in lower:
            return self.today_str
        if "ngày mốt" in lower or "ngày kia" in lower:
            return (self.now + timedelta(days=2)).strftime("%Y-%m-%d")
        if "ngày mai" in lower or "mai" in lower:
            return self.tomorrow_str
        if "hôm qua" in lower:
            return (self.now - timedelta(days=1)).strftime("%Y-%m-%d")

        # Explicit YYYY-MM-DD
        m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
        if m:
            return m.group(0)

        # DD/MM or DD-MM
        m2 = re.search(r"\b(\d{1,2})[/-](\d{1,2})\b", text)
        if m2:
            day = int(m2.group(1))
            month = int(m2.group(2))
            year = self.now.year
            try:
                dt = datetime(year, month, day, tzinfo=self.tz)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass

        return self.today_str

    def shift_iso_datetime(self, iso_dt: str, hours: float = 0, minutes: float = 0) -> str:
        """Adds or subtracts hours/minutes to an ISO-8601 datetime string, preserving timezone offset."""
        dt = datetime.fromisoformat(iso_dt)
        shifted = dt + timedelta(hours=hours, minutes=minutes)
        return shifted.isoformat()

    def format_iso(self, date_str: str, time_str: str = "08:00:00") -> str:
        """Combines YYYY-MM-DD and HH:MM[:SS] with current timezone offset."""
        if len(time_str.split(":")) == 2:
            time_str += ":00"
        dt = datetime.fromisoformat(f"{date_str}T{time_str}").replace(tzinfo=self.tz)
        return dt.isoformat()


class AgentContext:
    """Comprehensive context representation for an agent execution session."""

    def __init__(
        self,
        sample_id: str,
        current_time: str,
        timezone: str = "Asia/Ho_Chi_Minh",
        location: dict[str, Any] | None = None,
        user_profile: dict[str, Any] | None = None,
        installed_apps: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ):
        self.sample_id = sample_id
        self.current_time_str = current_time
        self.timezone_str = timezone
        self.temporal = TemporalReasoner(current_time, timezone)
        self.location = UserLocation.from_dict(location)
        self.profile = UserProfile.from_dict(user_profile)
        self.installed_apps = installed_apps or {}
        self.history = history or []

        # Session memory
        self.executed_calls: list[dict[str, Any]] = []
        self.tool_results: list[dict[str, Any]] = []
        self.observed_state: dict[str, Any] = {}

    @classmethod
    def from_sample(cls, sample: dict[str, Any]) -> AgentContext:
        ctx = sample.get("context", {})
        return cls(
            sample_id=sample.get("id", ""),
            current_time=ctx.get("current_time", ""),
            timezone=ctx.get("timezone", "Asia/Ho_Chi_Minh"),
            location=ctx.get("location"),
            user_profile=ctx.get("user_profile"),
            installed_apps=ctx.get("installed_apps"),
            history=sample.get("history", []),
        )

    def record_call_result(self, name: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
        """Stores a call and its result into session memory and updates observed state."""
        self.executed_calls.append({"name": name, "arguments": arguments})
        self.tool_results.append({"name": name, "arguments": arguments, "result": result})

        if isinstance(result, dict) and result.get("ok", True):
            app = name.split(".", 1)[0]
            app_state = self.observed_state.setdefault(app, {})
            for key in ["devices", "scenes", "events", "contacts", "forecast", "bills", "balance", "alarms", "messages", "notes"]:
                if key in result:
                    app_state[key] = result[key]

    def is_device_in_state(self, device_id: str, power: str) -> bool:
        """Checks if a smart home device currently matches desired power state ('on'/'off')."""
        devices = self.observed_state.get("smart_home", {}).get("devices", [])
        for d in devices:
            if d.get("id") == device_id or d.get("room") == device_id:
                return d.get("power") == power
        return False

    def is_forecast_rain(self, date_str: str | None = None) -> bool:
        """Checks if weather forecast predicts rain."""
        forecasts = self.observed_state.get("weather", {}).get("forecast", [])
        if not forecasts:
            return False
        for f in forecasts:
            if not date_str or f.get("date") == date_str:
                return bool(f.get("rain", False))
        return False

    def get_unpaid_bills(self) -> list[dict[str, Any]]:
        """Returns list of unpaid bills from observed state."""
        bills = self.observed_state.get("bill_payment", {}).get("bills", [])
        return [b for b in bills if b.get("status") == "unpaid"]

    def get_latest_result_for(self, tool_prefix: str) -> dict[str, Any] | None:
        """Finds the most recent result for a tool or app prefix."""
        for item in reversed(self.tool_results):
            if item["name"].startswith(tool_prefix):
                return item["result"]
        return None

    def format_for_prompt(self) -> str:
        """Formats the context into a concise summary block for the LLM."""
        lines = [
            f"- Thời điểm hiện tại: {self.current_time_str} (Múi giờ: {self.timezone_str})",
            f"- Ngày hôm nay: {self.temporal.today_str}, Ngày mai: {self.temporal.tomorrow_str}",
        ]
        if self.profile.name:
            lines.append(f"- Người dùng: {self.profile.name}")
        if self.profile.city:
            lines.append(f"- Thành phố: {self.profile.city}")
        if self.profile.home_address:
            lines.append(f"- Địa chỉ nhà: {self.profile.home_address}")
        if self.profile.work_address:
            lines.append(f"- Nơi làm việc: {self.profile.work_address}")

        # Dialogue History
        if self.history:
            lines.append("- Lịch sử hội thoại trước đó:")
            for idx, turn in enumerate(self.history, 1):
                role = turn.get("role", "unknown")
                text = turn.get("audio_text") or turn.get("text", "")
                calls = turn.get("tool_calls", [])
                calls_str = f" [Đã gọi: {', '.join(c.get('name', '') for c in calls)}]" if calls else ""
                lines.append(f"  {idx}. {role}: {text}{calls_str}")

        # Session Tool Results
        if self.tool_results:
            lines.append("- Kết quả các tool đã chạy trong phiên này:")
            for tr in self.tool_results:
                r_preview = str(tr["result"])
                if len(r_preview) > 200:
                    r_preview = r_preview[:200] + "..."
                lines.append(f"  * {tr['name']}({tr['arguments']}) -> {r_preview}")

        return "\n".join(lines)

