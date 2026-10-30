# Báo Cáo Phân Tích Lỗi Agent (Agent Error Analysis Report)
- **Tổng số mẫu đánh giá:** 50
- **Số mẫu thành công:** 7
- **Số mẫu thất bại:** 43

## 1. Phân Bố Các Loại Lỗi

| Loại Lỗi (Error Type) | Số Lượng Xuất Hiện | Mô Tả |
| :--- | :---: | :--- |
| `EXTRA_CALL` | 2 | Gọi thừa bước không cần thiết |
| `STATE_HASH_MISMATCH` | 39 | Lệch hash trạng thái ứng dụng cuối cùng |
| `OUT_OF_SCOPE_FAIL` | 38 | Xử lý sai mẫu ngoài phạm vi (gọi tool khi không cần hoặc ngược lại) |
| `MISSING_CALL` | 39 | Thiếu một hoặc nhiều bước gọi công cụ bắt buộc |
| `WRONG_ARGUMENTS` | 2 | Đúng tên công cụ nhưng tham số truyền vào sai lệch |
| `MISSING_INFO_FAIL` | 4 | Quên gọi hoặc gọi sai `sys.ask_missing_info` |
| `WRONG_TOOL_NAME` | 2 | Gọi nhầm công cụ so với đáp án gold |

## 2. Danh Sách Mẫu Lỗi Điển Hình

| Sample ID | Nhóm Lỗi | Predicted Calls | Gold Calls |
| :--- | :--- | :--- | :--- |
| `train_0002` | `EXTRA_CALL, STATE_HASH_MISMATCH` | `weather.get_forecast, calendar.list_events, calendar.update_event` | `weather.get_forecast` |
| `train_0003` | `OUT_OF_SCOPE_FAIL, MISSING_CALL, STATE_HASH_MISMATCH` | `[]` | `bill_payment.list_bills, contacts.search, ewallet.topup_mobile, contacts.search, ewallet.transfer` |
| `train_0004` | `OUT_OF_SCOPE_FAIL, MISSING_CALL, STATE_HASH_MISMATCH` | `[]` | `smart_home.get_status, smart_home.light_set` |
| `train_0005` | `OUT_OF_SCOPE_FAIL, MISSING_CALL, STATE_HASH_MISMATCH` | `[]` | `calendar.list_events, ride_hailing.book` |
| `train_0006` | `OUT_OF_SCOPE_FAIL, MISSING_CALL` | `[]` | `gold_price.get_price` |
| `train_0007` | `OUT_OF_SCOPE_FAIL, MISSING_CALL, STATE_HASH_MISMATCH` | `[]` | `alarm.create` |
| `train_0008` | `EXTRA_CALL, STATE_HASH_MISMATCH` | `weather.get_forecast, calendar.list_events, calendar.update_event` | `weather.get_forecast` |
| `train_0009` | `OUT_OF_SCOPE_FAIL, MISSING_CALL, STATE_HASH_MISMATCH` | `[]` | `ewallet.topup_mobile` |
| `train_0011` | `OUT_OF_SCOPE_FAIL, MISSING_CALL, STATE_HASH_MISMATCH` | `[]` | `bill_payment.list_bills, ewallet.get_balance, bill_payment.pay, reminders.create` |
| `train_0012` | `OUT_OF_SCOPE_FAIL, MISSING_CALL, STATE_HASH_MISMATCH` | `[]` | `banking.get_balance, banking.list_beneficiaries, banking.transfer, contacts.search, ewallet.topup_mobile` |
| `train_0014` | `OUT_OF_SCOPE_FAIL, MISSING_CALL, STATE_HASH_MISMATCH` | `[]` | `smart_home.get_status, smart_home.light_set, smart_home.light_set` |
| `train_0015` | `OUT_OF_SCOPE_FAIL, MISSING_CALL, STATE_HASH_MISMATCH` | `[]` | `bill_payment.list_bills, banking.get_balance, banking.list_beneficiaries, banking.transfer` |
| `train_0016` | `WRONG_ARGUMENTS, STATE_HASH_MISMATCH` | `weather.get_forecast, calendar.list_events, calendar.update_event, contacts.search, sms.send` | `weather.get_forecast, calendar.list_events, calendar.update_event, contacts.search, sms.send` |
| `train_0017` | `OUT_OF_SCOPE_FAIL, MISSING_CALL, STATE_HASH_MISMATCH` | `[]` | `tv.guide, tv.tune` |
| `train_0018` | `OUT_OF_SCOPE_FAIL, MISSING_CALL, STATE_HASH_MISMATCH` | `[]` | `contacts.search, phone.call` |
