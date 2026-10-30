# Hướng Dẫn Đánh Giá Hệ Thống LifeMate (Evaluation Guide)

Tài liệu mô tả phương pháp đo lường, công thức tính điểm và cách sử dụng bộ công cụ đánh giá cho cả hai luồng: **Speech/ASR (Track B)** và **Agent & Tool Execution (Track A)** trong cuộc thi Zalo AI Challenge 2026.

---

## 1. Cơ Chế Chấm Điểm của ZAC 2026

Ban giám khảo chấm điểm bài thi dựa trên 2 tiêu chí chính:
1. **Replay trên môi trường gốc (`zaic_tools`):**
   - Lấy chuỗi `tool_calls` trong file nộp `submission.jsonl`, replay tuần tự trên `state.json` ban đầu của từng mẫu bằng package `zaic_tools`.
   - Tính toán hàm băm trạng thái cuối cùng `state_hash` sau khi chạy xong tất cả tool.
2. **So sánh với Nhãn (`label`):**
   - **State Hash Match:** `pred_hash == gold_hash`. Phản ánh việc Agent có làm thay đổi dữ liệu ứng dụng chính xác như mong đợi hay không.
   - **Tool Call F1:** F1 score giữa tập tool call dự đoán và tập tool call chuẩn của mẫu.
   - **Zero Rejection:** Nếu một tool call bị `ToolEnv` từ chối (do sai schema, app chưa cài, thiếu tham số bắt buộc), call đó vẫn bị ghi lại và làm giảm điểm bài thi.

---

## 2. Đánh Giá Agent & Tool Execution (Track A)

Module `src/lifemate/evaluation/agent_metrics.py` cung cấp:
- **Precision, Recall, F1:** Đo lường độ chính xác của các tool calls (kèm fuzzy matching $\ge 0.85$ cho các trường văn bản tự do).
- **Sequence Exact Match Rate:** Tỷ lệ khớp hoàn toàn 100% thứ tự và tham số.
- **State Hash Accuracy:** Tỷ lệ khớp hash trạng thái ứng dụng cuối cùng.
- **Task Success Rate:** Thỏa mãn đồng thời F1 = 1.0 và State Hash match.
- **Schema Rejection Rate:** Mục tiêu duy trì 0.0% nhờ `ToolValidator`.

### Lệnh chạy đánh giá Agent:
```powershell
# Chạy đánh giá Agent với Heuristic Planner:
python scripts/evaluate_agent.py --split train --limit 50 --provider heuristic --error-report

# Chạy đánh giá Agent với Gemini LLM:
python scripts/evaluate_agent.py --split train --limit 20 --provider gemini --model gemini-2.5-flash
```

Báo cáo lỗi Agent được xuất tại `experiments/reports/error_analysis_report.md`.

---

## 3. Đánh Giá Speech & ASR (Track B)

### 3.1. Audio Dataset EDA
Thống kê sample rate, số kênh, độ dài, từ transcript và metadata vùng miền:
```powershell
python scripts/analyze_audio.py --data-root data/train --output-dir experiments/reports
```

### 3.2. ASR Benchmark
Đánh giá WER, CER, và Entity Error Rate trên cùng một tập validation split có seed cố định:
```powershell
python scripts/evaluate_asr.py --data-root data/train --runtime-config configs/runtime.yaml
```
- **WER (Word Error Rate) & CER (Character Error Rate):** Tính theo Levenshtein distance trên văn bản tiếng Việt đã chuẩn hóa.
- **Entity Error Rate:** Đo lường tỷ lệ lỗi trên các thực thể quan trọng (ngày, giờ, số tiền, SĐT).
- **Per-group Metrics:** Đánh giá độ bền bỉ theo phương ngữ / vùng miền (hoặc city proxy).
Kết quả lưu tại `experiments/asr/benchmark.json` và `experiments/reports/asr_benchmark.md`.

---

## 4. Đo Lường Runtime & Throughput
```powershell
python scripts/benchmark_runtime.py --limit 30 --timeout 10.0
```
Ghi nhận latency trung bình, P50/P90/P99, RAM tiêu thụ và số lượng vi phạm timeout 10s.
