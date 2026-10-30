# Hướng Dẫn Đánh Giá Hệ Thống Agent (LifeMate Evaluation Guide)

Tài liệu mô tả phương pháp đo lường, công thức tính điểm và cách sử dụng bộ công cụ đánh giá (`src/lifemate/evaluation/` & `scripts/evaluate_agent.py`) cho LifeMate trong cuộc thi Zalo AI Challenge 2026.

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

## 2. Các Chỉ Số Đo Lường (Metrics)

Module [`src/lifemate/evaluation/agent_metrics.py`](file:///d:/HCMUT/ZAIC-lifemate/src/lifemate/evaluation/agent_metrics.py) cung cấp:

### 2.1. Tool Calls Precision, Recall & F1
- **Precision:** Tỷ lệ tool call đúng trên tổng số tool call Agent sinh ra:
  $$\text{Precision} = \frac{\text{TP}}{\text{Tổng số predicted calls}}$$
- **Recall:** Tỷ lệ tool call đúng trên tổng số tool call chuẩn (gold):
  $$\text{Recall} = \frac{\text{TP}}{\text{Tổng số gold calls}}$$
- **F1 Score:** Trung bình điều hòa giữa Precision và Recall:
  $$\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- *Lưu ý:* Hai tool call được tính là True Positive (TP) khi và chỉ khi trùng tên tool (`name`) và khớp toàn bộ tham số (`arguments`). Với các trường văn bản tự do (`x-match: fuzzy`), hệ thống hỗ trợ so sánh độ tương đồng xấp xỉ $\ge 0.85$.

### 2.2. Sequence Exact Match Rate
Tỷ lệ các mẫu có chuỗi tool call khớp hoàn toàn 100% với gold cả về thứ tự thực thi và tham số:
$$\text{Exact Match Rate} = \frac{\sum \mathbb{I}(\text{pred\_calls} == \text{gold\_calls})}{N}$$

### 2.3. State Hash Accuracy
Tỷ lệ các mẫu có hash trạng thái ứng dụng cuối cùng trùng khớp tuyệt đối với đáp án sau khi replay:
$$\text{State Hash Accuracy} = \frac{\sum \mathbb{I}(\text{pred\_hash} == \text{gold\_hash})}{N}$$

### 2.4. Task Success Rate
Một mẫu được tính là **Task Success** hoàn chỉnh khi thỏa mãn đồng thời:
1. $\text{Tool Call F1} = 1.0$ (hoặc cả 2 cùng là `[]`).
2. $\text{State Hash}$ khớp 100%.

### 2.5. Schema Rejection Rate (Tỷ lệ Bị Từ Chối)
Tỷ lệ số cuộc gọi bị `ToolEnv` từ chối do vi phạm schema trên tổng số cuộc gọi:
$$\text{Rejection Rate} = \frac{\text{Total Rejected Calls}}{\text{Total Calls}}$$
*Mục tiêu:* Phải luôn duy trì **0.0%** nhờ lớp phòng vệ `ToolValidator`.

---

## 3. Cách Chạy Script Đánh Giá

### Đánh giá với Heuristic Planner (Nhanh & Offline):
```powershell
python scripts/evaluate_agent.py --split train --limit 50 --provider heuristic
```

### Đánh giá với mô hình LLM (Gemini):
```powershell
$env:GEMINI_API_KEY="AIzaSy..."
python scripts/evaluate_agent.py --split train --limit 20 --provider gemini --model gemini-2.5-flash
```

### Tự động xuất báo cáo phân tích lỗi (`error_analysis_report.md`):
```powershell
python scripts/evaluate_agent.py --split train --limit 50 --provider heuristic --error-report
```

Báo cáo phân loại lỗi sẽ được xuất trực tiếp tại `experiments/reports/error_analysis_report.md` và `.json`.

