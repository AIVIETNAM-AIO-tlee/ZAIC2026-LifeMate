#!/usr/bin/env python3
"""Dataset & Label Analysis for ZAC 2026 LifeMate.

Analyzes train.jsonl and train_tools_catalog.json to extract statistics on:
- Tool call counts and distributions (0 calls, single call, multi-step chains)
- Tool frequency by name, kind (read/write/meta)
- sys.ask_missing_info frequency, target tools, and missing parameters
- Empty calls (out-of-scope / conversational)
- Tool results schemas and success rates
- Final state mutations per application
- Task complexity archetypes
- Context statistics (history turns, installed apps count, current time)

Outputs reports in Markdown and JSON formats to experiments/reports/.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze agent training dataset and tool catalog.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/train/train.jsonl",
        help="Path to train.jsonl",
    )
    parser.add_argument(
        "--catalog-path",
        type=str,
        default="data/train/train_tools_catalog.json",
        help="Path to train_tools_catalog.json",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="experiments/reports",
        help="Directory to save analysis reports",
    )
    return parser.parse_args()


def load_catalog(catalog_path: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """Loads catalog and returns tool definitions and tool->kind mapping."""
    if not catalog_path.exists():
        return {}, {}
    with open(catalog_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    tools_map = {}
    kind_map = {}
    for app in raw.get("apps", []):
        for tool in app.get("tools", []):
            name = tool.get("name")
            if name:
                tools_map[name] = tool
                kind_map[name] = tool.get("kind", "write")

    # Add default sys tools if not explicitly present
    if "sys.ask_missing_info" not in kind_map:
        kind_map["sys.ask_missing_info"] = "meta"

    return tools_map, kind_map


def classify_task_archetype(tool_calls: list[dict], kind_map: dict[str, str]) -> str:
    """Classifies a sample into one of several behavioral archetypes."""
    if not tool_calls:
        return "out_of_scope_or_no_tool"

    call_names = [c.get("name", "") for c in tool_calls]
    if any(name == "sys.ask_missing_info" for name in call_names):
        return "clarification_missing_info"

    kinds = [kind_map.get(name, "write") for name in call_names]
    reads = [k for k in kinds if k == "read"]
    writes = [k for k in kinds if k == "write"]

    if len(tool_calls) == 1:
        if reads:
            return "single_read_query"
        return "single_write_action"

    # Multi-call
    if reads and not writes:
        return "multi_read_query"
    if writes and not reads:
        return "multi_write_direct"
    if reads and writes:
        return "read_then_write_dependent"

    return "complex_mixed"


def run_analysis(data_path: Path, catalog_path: Path, report_dir: Path) -> dict:
    tools_map, kind_map = load_catalog(catalog_path)

    total_samples = 0
    call_counts: list[int] = []
    tool_counter: collections.Counter[str] = collections.Counter()
    kind_counter: collections.Counter[str] = collections.Counter()
    archetype_counter: collections.Counter[str] = collections.Counter()
    app_counter: collections.Counter[str] = collections.Counter()
    installed_apps_count: list[int] = []
    history_turns_count: list[int] = []

    missing_info_tools: collections.Counter[str] = collections.Counter()
    missing_info_params: collections.Counter[str] = collections.Counter()

    final_state_app_mutations: collections.Counter[str] = collections.Counter()
    tool_results_ok_count = 0
    tool_results_total_count = 0
    tool_results_keys: collections.Counter[str] = collections.Counter()

    empty_call_samples: list[dict] = []
    missing_info_samples: list[dict] = []
    complex_samples: list[dict] = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_samples += 1
            item = json.loads(line)
            sample_id = item.get("id", f"sample_{total_samples}")
            audio_text = item.get("audio_text", "")
            context = item.get("context", {})
            history = item.get("history", [])
            label = item.get("label", {})

            # Installed apps
            inst_apps = context.get("installed_apps", {})
            installed_apps_count.append(len(inst_apps))
            for app_name in inst_apps:
                app_counter[app_name] += 1

            # History turns
            history_turns_count.append(len(history))

            # Tool calls
            calls = label.get("tool_calls", [])
            call_counts.append(len(calls))

            archetype = classify_task_archetype(calls, kind_map)
            archetype_counter[archetype] += 1

            if not calls:
                if len(empty_call_samples) < 5:
                    empty_call_samples.append({
                        "id": sample_id,
                        "audio_text": audio_text,
                        "installed_apps": list(inst_apps.keys()),
                    })

            for call in calls:
                name = call.get("name", "")
                args = call.get("arguments", {})
                tool_counter[name] += 1
                k = kind_map.get(name, "unknown")
                kind_counter[k] += 1

                if name == "sys.ask_missing_info":
                    target_tool = args.get("tool_name", "unknown")
                    missing_info_tools[target_tool] += 1
                    for p in args.get("params", []):
                        missing_info_params[p] += 1
                    if len(missing_info_samples) < 5:
                        missing_info_samples.append({
                            "id": sample_id,
                            "audio_text": audio_text,
                            "target_tool": target_tool,
                            "params": args.get("params", []),
                        })

            if archetype in ("read_then_write_dependent", "complex_mixed") and len(complex_samples) < 5:
                complex_samples.append({
                    "id": sample_id,
                    "audio_text": audio_text,
                    "calls": [c.get("name") for c in calls],
                })

            # Tool results
            results = label.get("tool_results", [])
            for res in results:
                tool_results_total_count += 1
                if isinstance(res, dict):
                    if res.get("ok", True):
                        tool_results_ok_count += 1
                    for k in res:
                        tool_results_keys[k] += 1

            # Final state mutations
            final_state = label.get("final_state", {})
            for mutated_app in final_state:
                final_state_app_mutations[mutated_app] += 1

    summary = {
        "dataset": {
            "total_samples": total_samples,
            "installed_apps_per_sample": {
                "min": min(installed_apps_count) if installed_apps_count else 0,
                "max": max(installed_apps_count) if installed_apps_count else 0,
                "mean": round(statistics.mean(installed_apps_count), 2) if installed_apps_count else 0,
            },
            "history_turns_per_sample": {
                "min": min(history_turns_count) if history_turns_count else 0,
                "max": max(history_turns_count) if history_turns_count else 0,
                "mean": round(statistics.mean(history_turns_count), 2) if history_turns_count else 0,
                "zero_history_count": sum(1 for h in history_turns_count if h == 0),
                "has_history_count": sum(1 for h in history_turns_count if h > 0),
            },
        },
        "tool_calls_distribution": {
            "total_calls": sum(call_counts),
            "min_calls": min(call_counts) if call_counts else 0,
            "max_calls": max(call_counts) if call_counts else 0,
            "mean_calls": round(statistics.mean(call_counts), 2) if call_counts else 0,
            "median_calls": statistics.median(call_counts) if call_counts else 0,
            "distribution_by_count": dict(collections.Counter(call_counts)),
        },
        "task_archetypes": dict(archetype_counter),
        "tool_kinds": dict(kind_counter),
        "tool_frequency_top20": dict(tool_counter.most_common(20)),
        "all_tools_frequency": dict(tool_counter),
        "missing_info": {
            "total_missing_info_calls": sum(missing_info_tools.values()),
            "by_target_tool": dict(missing_info_tools),
            "by_param": dict(missing_info_params),
            "examples": missing_info_samples,
        },
        "empty_calls": {
            "total_empty_samples": archetype_counter["out_of_scope_or_no_tool"],
            "examples": empty_call_samples,
        },
        "tool_results_stats": {
            "total_results": tool_results_total_count,
            "ok_true_count": tool_results_ok_count,
            "ok_rate": round(tool_results_ok_count / tool_results_total_count, 4) if tool_results_total_count else 0,
            "top_keys": dict(tool_results_keys.most_common(15)),
        },
        "final_state_mutations": dict(final_state_app_mutations),
        "installed_apps_frequency": dict(app_counter),
        "complex_workflow_examples": complex_samples,
    }

    report_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON report
    json_path = report_dir / "agent_dataset_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Generate Markdown report
    md_path = report_dir / "agent_dataset_analysis.md"
    generate_markdown_report(summary, md_path)

    print(f"Analysis completed successfully:")
    print(f"  - JSON report: {json_path}")
    print(f"  - Markdown report: {md_path}")
    return summary


def generate_markdown_report(data: dict, output_path: Path) -> None:
    lines = [
        "# Báo Cáo Phân Tích Dữ Liệu Huấn Luyện (Agent Dataset Analysis)",
        f"**Tổng số mẫu phân tích:** {data['dataset']['total_samples']:,}",
        "",
        "## 1. Phân Phối Số Lượng Tool Calls",
        "",
        f"- **Tổng số lượt gọi tool:** {data['tool_calls_distribution']['total_calls']:,}",
        f"- **Trung bình:** {data['tool_calls_distribution']['mean_calls']} calls/mẫu (Trung vị: {data['tool_calls_distribution']['median_calls']})",
        f"- **Dải số bước:** Min = {data['tool_calls_distribution']['min_calls']}, Max = {data['tool_calls_distribution']['max_calls']}",
        "",
        "| Số Tool Calls | Số Lượng Mẫu | Tỷ Lệ (%) |",
        "| :---: | :---: | :---: |",
    ]

    total = data["dataset"]["total_samples"]
    dist = data["tool_calls_distribution"]["distribution_by_count"]
    for count in sorted(dist.keys()):
        freq = dist[count]
        pct = round(freq / total * 100, 2)
        lines.append(f"| {count} | {freq} | {pct}% |")

    lines.extend([
        "",
        "## 2. Phân Loại Tác Vụ (Task Archetypes)",
        "",
        "| Loại Tác Vụ (Archetype) | Số Mẫu | Tỷ Lệ (%) | Mô Tả |",
        "| :--- | :---: | :---: | :--- |",
    ])

    archetypes_desc = {
        "out_of_scope_or_no_tool": "Không cần gọi tool (ngoài phạm vi / hội thoại thuần túy)",
        "single_read_query": "Chỉ đọc thông tin 1 lần (thời tiết, tìm danh bạ, kiểm tra đèn)",
        "single_write_action": "Thực thi 1 hành động trực tiếp (đã có đủ thông tin)",
        "read_then_write_dependent": "Đọc để lấy ID/thông tin -> Thực hiện hành động ghi (Multi-step)",
        "multi_read_query": "Đọc nhiều thông tin liên tiếp mà không thay đổi trạng thái",
        "multi_write_direct": "Thực hiện nhiều hành động thay đổi liên tiếp",
        "clarification_missing_info": "Thiếu thông tin bắt buộc -> Gọi `sys.ask_missing_info`",
        "complex_mixed": "Hỗn hợp phức tạp đa bước",
    }

    for arch, count in data["task_archetypes"].items():
        pct = round(count / total * 100, 2)
        desc = archetypes_desc.get(arch, "")
        lines.append(f"| `{arch}` | {count} | {pct}% | {desc} |")

    lines.extend([
        "",
        "## 3. Thống Kê Loại Tool (Read / Write / Meta)",
        "",
        "| Loại (Kind) | Số Lượt Gọi | Tỷ Lệ (%) |",
        "| :--- | :---: | :---: |",
    ])
    total_calls = data["tool_calls_distribution"]["total_calls"]
    for kind, count in data["tool_kinds"].items():
        pct = round(count / total_calls * 100, 2) if total_calls else 0
        lines.append(f"| `{kind}` | {count} | {pct}% |")

    lines.extend([
        "",
        "## 4. Top 15 Tool Được Sử Dụng Nhiều Nhất",
        "",
        "| STT | Tên Tool | Số Lần Gọi |",
        "| :---: | :--- | :---: |",
    ])
    for idx, (tname, freq) in enumerate(list(data["tool_frequency_top20"].items())[:15], 1):
        lines.append(f"| {idx} | `{tname}` | {freq} |")

    lines.extend([
        "",
        "## 5. Phân Tích Thiếu Thông Tin (`sys.ask_missing_info`)",
        "",
        f"- **Tổng số lần gọi `sys.ask_missing_info`:** {data['missing_info']['total_missing_info_calls']}",
        "",
        "### Các công cụ bị thiếu tham số nhiều nhất:",
        "| Target Tool | Số Lần Bị Thiếu |",
        "| :--- | :---: |",
    ])
    for tname, freq in data["missing_info"]["by_target_tool"].items():
        lines.append(f"| `{tname}` | {freq} |")

    lines.extend([
        "",
        "### Các tham số thường xuyên bị thiếu:",
        "| Tên Tham Số | Số Lần Bị Thiếu |",
        "| :--- | :---: |",
    ])
    for pname, freq in data["missing_info"]["by_param"].items():
        lines.append(f"| `{pname}` | {freq} |")

    lines.extend([
        "",
        "## 6. Biến Động Trạng Thái Ứng Dụng (`final_state`)",
        "",
        "Những ứng dụng có dữ liệu bị thay đổi sau khi chạy toàn bộ chuỗi tool:",
        "",
        "| Ứng Dụng | Số Lượng Mẫu Có Biến Động State |",
        "| :--- | :---: |",
    ])
    for app, count in data["final_state_mutations"].items():
        lines.append(f"| `{app}` | {count} |")

    lines.extend([
        "",
        "## 7. Đúc Rút Chiến Lược Cho Agent",
        "",
        "1. **Tỷ lệ Multi-step cao:** Rất nhiều mẫu yêu cầu gọi tool đọc (như `contacts.search`, `calendar.list_events`, `weather.get_forecast`) trước khi thực hiện thao tác ghi (`sms.send`, `calendar.update_event`). Agent bắt buộc phải có cơ chế **Argument Grounding** truyền ID từ tool result sang tool call kế tiếp.",
        "2. **Cơ chế Thiếu thông tin (`sys.ask_missing_info`):** Phải phân biệt rõ khi nào tham số có thể tra cứu được bằng tool đọc vs khi nào người dùng thực sự không cung cấp (như thiếu `content` trong SMS, thiếu tên bài hát/playlist).",
        "3. **Xử lý Ngoài phạm vi (`[]`):** Khi yêu cầu không liên quan đến app đã cài đặt hoặc là câu chào hỏi, Agent phải dừng ngay và không gọi bất kỳ tool nào để bảo toàn state_hash ban đầu.",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    run_analysis(
        data_path=Path(args.data_path),
        catalog_path=Path(args.catalog_path),
        report_dir=Path(args.report_dir),
    )


if __name__ == "__main__":
    main()

