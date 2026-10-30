# Hướng Dẫn Đóng Gói Docker & Triển Khai (Deployment Guide)

Tài liệu hướng dẫn cài đặt cục bộ, đóng gói hệ thống LifeMate thành Docker container và chạy đánh giá theo tiêu chuẩn của Ban tổ chức Zalo AI Challenge 2026.

---

## 1. Cài đặt Cục bộ (Local Environment Setup)

Cài đặt project và các phụ thuộc ASR:

```powershell
python -m pip install -e ".[asr]"
```

Mô hình mặc định là `openai/whisper-small`. Khi chạy lần đầu, Transformers sẽ tải về thư mục `models/asr`. Khi chạy offline trong phòng thi, đặt `local_files_only: true` và đặt checkpoint tương thích vào thư mục đó. Tùy chỉnh CPU/CUDA và độ chính xác trong `configs/asr.yaml`.

---

## 2. Cấu Trúc Dockerfile & Môi Trường

Hệ thống được đóng gói dựa trên image chuẩn `python:3.11-slim`:
- Hệ điều hành Linux tối ưu, cài sẵn `ffmpeg` cho xử lý âm thanh thoại.
- Biến môi trường:
  - `PYTHONUNBUFFERED=1`: Xuất log trực tiếp, không đệm buffer.
  - `PYTHONPATH=/app/src`: Tự động nạp mã nguồn package `lifemate`.
  - `PYTHONIOENCODING=utf-8`: Đảm bảo xử lý văn bản tiếng Việt Unicode trơn tru.

---

## 3. Hướng Dẫn Build Docker Image

Từ thư mục gốc của dự án (`ZAIC-lifemate/`), chạy lệnh:

```bash
docker build -t zaic-lifemate:latest .
```

Kiểm tra kích thước image sau khi build:
```bash
docker images | grep zaic-lifemate
```

---

## 4. Chạy Thử Nghiệm Container Cục Bộ (Local Testing)

Khi BTC đánh giá trên vòng Private Test (hoặc chạy test trên `public_test`):
- Thư mục dữ liệu `data/` được mount vào container tại `/app/data`.
- Thư mục kết quả `result/` được mount tại `/app/result`.

Lệnh chạy Docker:
```bash
docker run --rm \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/result:/app/result \
  zaic-lifemate:latest \
  --split public_test \
  --data-path data/public_test/public_test.jsonl \
  --catalog-path data/public_test/public_test_tools_catalog.json \
  --output result/submission.jsonl
```

Sau khi chạy xong, file `result/submission.jsonl` sẽ được sinh ra ở máy chủ host.

---

## 5. Kiểm Tra Tương Thích & Giới Hạn Thời Gian

BTC quy định thời gian xử lý tối đa cho mỗi mẫu là **10 giây/mẫu**. Bạn có thể kiểm tra hiệu năng hệ thống bằng script benchmark:

```bash
python scripts/benchmark_runtime.py --limit 30 --timeout 10.0
```

Yêu cầu nghiệm thu:
- **Throughput:** $\ge 5$ mẫu/giây với Heuristic, $\ge 0.5$ mẫu/giây với mô hình LLM.
- **Timeout Violations:** 0 mẫu vi phạm ngưỡng 10s.
- **RAM tiêu thụ:** $< 4$ GB.
