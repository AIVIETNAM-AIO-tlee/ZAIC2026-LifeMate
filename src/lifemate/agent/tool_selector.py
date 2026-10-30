"""Tool Availability Filter & Out-of-Scope Detection for ZAC 2026 LifeMate.

Filters available tools per sample strictly based on context.installed_apps,
always including the sys.ask_missing_info meta tool, and detects out-of-scope,
chitchat, cancellation, and missing app requests.
"""

from __future__ import annotations

import re
from typing import Any
from lifemate.tools.catalog import ToolCatalog, ToolSpec

# Chitchat and conversational greetings
CHITCHAT_PATTERNS = [
    r"^(xin\s+)?chào(\s+(bạn|em|ad|trợ\s+lý|lifemate))?[\.!\?]?$",
    r"^hôm\s+nay\s+thế\s+nào[\.!\?]?$",
    r"^(cảm\s+ơn|thanks)(\s+(bạn|em))?[\.!\?]?$",
    r"^(bạn\s+là\s+ai|bạn\s+làm\s+được\s+gì)[\.!\?]?$",
    r"^tạm\s+biệt[\.!\?]?$",
]

# User cancellation or revocation of intention
CANCELLATION_PATTERNS = [
    r"à\s+mà\s+thôi",
    r"thôi\s+(không|khỏi|đừng)\s+(cần|làm|gửi|chuyển|đặt|bật|tắt)",
    r"cứ\s+để\s+(nguyên|yên|vậy|đó)",
    r"không\s+cần\s+(làm\s+gì|sửa\s+gì|nữa)",
]

# Comprehensive domain keyword indicators across all catalog apps
APP_DOMAIN_KEYWORDS = {
    "smart_home": ["đèn", "quạt", "máy lạnh", "rèm", "ổ cắm", "thiết bị", "phòng khách", "phòng ngủ", "nhà thông minh"],
    "music": ["bài hát", "nhạc", "playlist", "nghe nhạc", "phát bài", "danh sách phát", "album"],
    "calendar": ["lịch", "sự kiện", "cuộc hẹn", "họp", "chạy bộ", "đá bóng", "lên lịch", "dời lịch"],
    "sms": ["nhắn tin", "tin nhắn", "sms", "gửi tin"],
    "weather": ["thời tiết", "dự báo", "nhiệt độ", "mưa", "nắng", "bão"],
    "alarm": ["báo thức", "hẹn giờ", "đặt báo thức"],
    "bill_payment": ["hóa đơn", "tiền điện", "tiền nước", "tiền internet", "thanh toán cước"],
    "banking": ["chuyển khoản", "chuyển tiền", "số dư tài khoản", "ngân hàng"],
    "ewallet": ["ví điện tử", "nạp ví", "zalopay", "momo", "quét qr"],
    "flights": ["chuyến bay", "vé máy bay", "bay đi", "hàng không"],
    "food_delivery": ["đặt món", "giao đồ ăn", "đồ ăn", "trà sữa", "cơm", "bún", "phở", "quán ăn", "đặt đồ ăn"],
    "ride_hailing": ["đặt xe", "gọi xe", "xe ôm", "taxi", "grab", "be", "chở đi", "điểm đón", "điểm đến"],
    "contacts": ["danh bạ", "liên hệ", "số điện thoại", "tìm liên hệ"],
    "ecommerce": ["mua hàng", "đơn hàng", "giỏ hàng", "sản phẩm"],
    "email": ["hòm thư", "gửi mail", "soạn mail", "thư điện tử", "hộp thư"],
    "stocks": ["cổ phiếu", "chứng khoán", "thị trường chứng khoán", "giá cổ phiếu"],
    "gold_price": ["giá vàng", "vàng sjc", "giá vàng hôm nay"],
    "fuel_price": ["giá xăng", "giá dầu", "xăng ron 95", "xăng e5"],
    "exchange_rate": ["tỷ giá", "đổi ngoại tệ", "tỷ giá usd", "tỷ giá eur"],
    "maps": ["chỉ đường", "bản đồ", "định vị", "khoảng cách"],
    "medication": ["uống thuốc", "nhắc uống thuốc", "đơn thuốc", "liều thuốc"],
    "reminders": ["nhắc việc", "nhắc tôi", "nhắc nhở"],
    "tasks": ["nhiệm vụ", "công việc cần làm", "todo list", "danh sách việc"],
    "tv": ["bật tivi", "kênh tivi", "tivi", "chuyển kênh"],
    "door_lock": ["khóa cửa", "mở khóa", "chốt cửa"],
    "home_security": ["an ninh", "camera an ninh", "báo động"],
    "package_tracking": ["tra cứu vận đơn", "mã vận đơn", "kiện hàng", "giao hàng"],
    "event_tickets": ["vé xem phim", "vé ca nhạc", "mua vé", "rạp chiếu phim"],
    "train_bus_tickets": ["vé tàu", "vé xe khách", "vé xe bus"],
    "household_chores": ["việc nhà", "lịch dọn nhà"],
    "habit_tracker": ["thói quen", "theo dõi thói quen"],
    "subscriptions": ["gói thuê bao", "gia hạn", "hủy thuê bao"],
    "video_meeting": ["họp online", "cuộc họp trực tuyến", "zoom", "google meet", "tạo cuộc họp"],
    "translator": ["dịch thuật", "dịch câu", "dịch từ"],
    "news": ["tin tức", "đọc báo", "bản tin"],
    "podcast_radio": ["podcast", "nghe đài", "radio", "phát thanh"],
    "library": ["mượn sách", "thư viện", "trả sách"],
    "lottery": ["kết quả xổ số", "xổ số", "vé số"],
    "loyalty": ["tích điểm", "điểm thưởng", "đổi quà"],
    "parking": ["bãi đỗ xe", "tìm chỗ đỗ xe", "gửi xe"],
    "pantry": ["tủ lạnh", "nguyên liệu", "thực phẩm trong nhà"],
    "recipes": ["công thức nấu ăn", "cách nấu", "món ngon"],
    "calculator": ["tính toán", "phép tính", "máy tính"],
    "bike_rental": ["thuê xe đạp"],
    "chat_ott": ["nhắn tin zalo", "chat"],
    "phone": ["gọi điện", "cuộc gọi"],
    "public_transit": ["xe buýt", "tuyến bus", "tàu điện"],
    "shopping_list": ["danh sách mua sắm", "mua đồ"],
    "learning": ["học bài", "khóa học", "buổi học", "bài giảng", "gia sư"],
    "fitness": ["chạy bộ", "tập thể dục", "calo", "bước chân", "bài tập gym"],
    "notes": ["ghi chú", "tạo note", "nhắc nhở"],
    "robot_vacuum": ["hút bụi", "robot hút bụi", "lau nhà"],
}


class ToolSelector:
    """Filters tools available to the agent for a given sample context and detects out-of-scope requests."""

    def __init__(self, catalog: ToolCatalog):
        self.catalog = catalog

    def get_allowed_tool_names(self, installed_apps: dict[str, Any] | list[str] | None) -> set[str]:
        """Extracts allowed tool names from installed_apps, always including sys.ask_missing_info."""
        allowed: set[str] = set()
        if isinstance(installed_apps, dict):
            for app_name, app_info in installed_apps.items():
                if isinstance(app_info, dict):
                    tools = app_info.get("tools", [])
                    allowed.update(tools)
                elif isinstance(app_info, list):
                    allowed.update(app_info)
        elif isinstance(installed_apps, list):
            allowed.update(installed_apps)

        # sys.ask_missing_info is universally permitted across all samples
        allowed.add("sys.ask_missing_info")
        return allowed

    def get_allowed_tools(self, installed_apps: dict[str, Any] | list[str] | None) -> list[ToolSpec]:
        """Returns list of ToolSpec for allowed tools."""
        allowed_names = self.get_allowed_tool_names(installed_apps)
        return self.catalog.filter_by_names(allowed_names)

    def is_tool_allowed(self, tool_name: str, installed_apps: dict[str, Any] | list[str] | None) -> bool:
        """Checks if a tool name is allowed for this sample."""
        if tool_name == "sys.ask_missing_info":
            return True
        allowed_names = self.get_allowed_tool_names(installed_apps)
        return tool_name in allowed_names

    def get_installed_app_ids(self, installed_apps: dict[str, Any] | list[str] | None) -> set[str]:
        """Extracts the set of installed app identifiers."""
        if isinstance(installed_apps, dict):
            return set(installed_apps.keys())
        if isinstance(installed_apps, list):
            return {item.split(".", 1)[0] for item in installed_apps if "." in item}
        return set()

    def detect_out_of_scope(
        self,
        user_request: str,
        installed_apps: dict[str, Any] | list[str] | None,
    ) -> tuple[bool, str]:
        """Detects if request is out of scope, pure chitchat, user cancellation, or uninstalled app.
        Returns: (is_out_of_scope, reason)
        """
        text = user_request.strip().lower()

        # 1. Pure chitchat check
        for pattern in CHITCHAT_PATTERNS:
            if re.search(pattern, text):
                return True, "chitchat_greeting"

        # 2. Cancellation check
        for pattern in CANCELLATION_PATTERNS:
            if re.search(pattern, text):
                # Check if user cancelled the whole request without any surviving sub-intent
                if not any(kw in text for kw in ["xem giúp", "chỉ cần xem", "hãy xem", "kiểm tra giúp"]):
                    return True, "user_cancelled"

        # 3. Required app not installed check
        installed_set = self.get_installed_app_ids(installed_apps)
        detected_apps = set()
        for app, keywords in APP_DOMAIN_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                detected_apps.add(app)

        # If user explicitly asks for actions of an app that is NOT installed
        if detected_apps and not (detected_apps & installed_set):
            # None of the detected apps are installed
            # Check if this isn't handled by sys
            return True, f"uninstalled_app:{list(detected_apps)[0]}"

        return False, "in_scope"

    def to_openai_tools(self, installed_apps: dict[str, Any] | list[str] | None) -> list[dict[str, Any]]:
        """Exports allowed tools in OpenAI function calling format."""
        allowed_names = self.get_allowed_tool_names(installed_apps)
        return self.catalog.to_openai_tools(allowed_names)

    def to_prompt_summary(self, installed_apps: dict[str, Any] | list[str] | None) -> str:
        """Returns compact text summary of allowed tools for system prompt."""
        allowed_names = self.get_allowed_tool_names(installed_apps)
        return self.catalog.to_compact_prompt_summary(allowed_names)
