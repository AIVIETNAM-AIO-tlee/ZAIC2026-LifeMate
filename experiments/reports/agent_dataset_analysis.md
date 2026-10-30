# Báo Cáo Phân Tích Dữ Liệu Huấn Luyện (Agent Dataset Analysis)
**Tổng số mẫu phân tích:** 1,545

## 1. Phân Phối Số Lượng Tool Calls

- **Tổng số lượt gọi tool:** 3,541
- **Trung bình:** 2.29 calls/mẫu (Trung vị: 2)
- **Dải số bước:** Min = 0, Max = 6

| Số Tool Calls | Số Lượng Mẫu | Tỷ Lệ (%) |
| :---: | :---: | :---: |
| 0 | 111 | 7.18% |
| 1 | 369 | 23.88% |
| 2 | 428 | 27.7% |
| 3 | 359 | 23.24% |
| 4 | 161 | 10.42% |
| 5 | 107 | 6.93% |
| 6 | 10 | 0.65% |

## 2. Phân Loại Tác Vụ (Task Archetypes)

| Loại Tác Vụ (Archetype) | Số Mẫu | Tỷ Lệ (%) | Mô Tả |
| :--- | :---: | :---: | :--- |
| `read_then_write_dependent` | 796 | 51.52% | Đọc để lấy ID/thông tin -> Thực hiện hành động ghi (Multi-step) |
| `single_read_query` | 97 | 6.28% | Chỉ đọc thông tin 1 lần (thời tiết, tìm danh bạ, kiểm tra đèn) |
| `single_write_action` | 212 | 13.72% | Thực thi 1 hành động trực tiếp (đã có đủ thông tin) |
| `out_of_scope_or_no_tool` | 111 | 7.18% | Không cần gọi tool (ngoài phạm vi / hội thoại thuần túy) |
| `clarification_missing_info` | 217 | 14.05% | Thiếu thông tin bắt buộc -> Gọi `sys.ask_missing_info` |
| `multi_write_direct` | 85 | 5.5% | Thực hiện nhiều hành động thay đổi liên tiếp |
| `multi_read_query` | 27 | 1.75% | Đọc nhiều thông tin liên tiếp mà không thay đổi trạng thái |

## 3. Thống Kê Loại Tool (Read / Write / Meta)

| Loại (Kind) | Số Lượt Gọi | Tỷ Lệ (%) |
| :--- | :---: | :---: |
| `read` | 1599 | 45.16% |
| `write` | 1725 | 48.72% |
| `meta` | 217 | 6.13% |

## 4. Top 15 Tool Được Sử Dụng Nhiều Nhất

| STT | Tên Tool | Số Lần Gọi |
| :---: | :--- | :---: |
| 1 | `contacts.search` | 234 |
| 2 | `sys.ask_missing_info` | 217 |
| 3 | `smart_home.light_set` | 126 |
| 4 | `reminders.create` | 115 |
| 5 | `bill_payment.list_bills` | 112 |
| 6 | `smart_home.get_status` | 107 |
| 7 | `calendar.list_events` | 105 |
| 8 | `bill_payment.pay` | 104 |
| 9 | `chat_ott.list_chats` | 94 |
| 10 | `banking.transfer` | 82 |
| 11 | `sms.send` | 79 |
| 12 | `alarm.create` | 78 |
| 13 | `chat_ott.send_message` | 69 |
| 14 | `banking.get_balance` | 62 |
| 15 | `weather.get_forecast` | 60 |

## 5. Phân Tích Thiếu Thông Tin (`sys.ask_missing_info`)

- **Tổng số lần gọi `sys.ask_missing_info`:** 217

### Các công cụ bị thiếu tham số nhiều nhất:
| Target Tool | Số Lần Bị Thiếu |
| :--- | :---: |
| `phone.call` | 43 |
| `sms.send` | 27 |
| `ewallet.transfer` | 21 |
| `reminders.create` | 7 |
| `alarm.create` | 1 |
| `email.create_draft` | 4 |
| `video_meeting.create` | 6 |
| `chat_ott.send_message` | 5 |
| `flights.search` | 2 |
| `maps.get_route` | 1 |
| `calendar.create_event` | 1 |
| `learning.schedule_lesson` | 2 |
| `video_meeting.join` | 2 |
| `podcast_radio.play` | 2 |
| `appliances.stop` | 5 |
| `bike_rental.reserve` | 3 |
| `tasks.complete` | 1 |
| `fitness.set_goal` | 5 |
| `bill_payment.schedule_autopay` | 2 |
| `news.save` | 2 |
| `robot_vacuum.start` | 1 |
| `ewallet.pay_qr` | 4 |
| `subscriptions.cancel` | 3 |
| `package_tracking.reschedule_delivery` | 6 |
| `email.archive` | 5 |
| `household_chores.assign` | 2 |
| `loyalty.redeem` | 4 |
| `home_energy.set_alert` | 3 |
| `smart_home.scene_run` | 1 |
| `flights.cancel` | 1 |
| `tv.tune` | 3 |
| `medication.remove` | 1 |
| `library.reserve` | 3 |
| `reminders.complete` | 1 |
| `flights.check_in` | 1 |
| `learning.log_progress` | 2 |
| `podcast_radio.subscribe` | 1 |
| `smart_home.switch_set` | 1 |
| `package_tracking.schedule_pickup` | 1 |
| `fitness.log_workout` | 1 |
| `ride_hailing.book` | 2 |
| `music.add_to_playlist` | 1 |
| `tv.record` | 1 |
| `reminders.delete` | 1 |
| `door_lock.grant_guest_code` | 2 |
| `exchange_rate.get_rate` | 2 |
| `home_security.acknowledge_alert` | 3 |
| `banking.transfer` | 1 |
| `medication.log_taken` | 1 |
| `medication.add` | 1 |
| `banking.open_savings` | 2 |
| `lottery.check_ticket` | 2 |
| `home_security.arm` | 1 |
| `food_delivery.cancel_order` | 2 |
| `tv.control` | 2 |
| `notes.append` | 3 |
| `ewallet.topup_mobile` | 1 |
| `music.play` | 1 |
| `contacts.update` | 1 |

### Các tham số thường xuyên bị thiếu:
| Tên Tham Số | Số Lần Bị Thiếu |
| :--- | :---: |
| `contact_id` | 73 |
| `amount` | 21 |
| `time` | 10 |
| `to` | 4 |
| `start` | 7 |
| `chat_id` | 5 |
| `date` | 3 |
| `origin` | 1 |
| `course_id` | 4 |
| `meeting_id` | 2 |
| `episode_id` | 2 |
| `appliance_id` | 5 |
| `bike_id` | 3 |
| `task_id` | 1 |
| `steps` | 5 |
| `type` | 3 |
| `article_id` | 2 |
| `rooms` | 1 |
| `merchant_id` | 3 |
| `subscription_id` | 3 |
| `email_id` | 4 |
| `chore_id` | 1 |
| `reward_id` | 4 |
| `threshold_kwh` | 3 |
| `scene_id` | 1 |
| `booking_id` | 2 |
| `channel` | 3 |
| `schedule_id` | 2 |
| `book_id` | 3 |
| `reminder_id` | 2 |
| `show_id` | 1 |
| `device_id` | 1 |
| `address` | 1 |
| `dropoff` | 1 |
| `track_id` | 1 |
| `program_id` | 1 |
| `tracking_id` | 4 |
| `lock_id` | 2 |
| `from_currency` | 2 |
| `alert_id` | 3 |
| `vehicle` | 1 |
| `beneficiary_id` | 1 |
| `action` | 1 |
| `times` | 1 |
| `term_months` | 2 |
| `number` | 1 |
| `mode` | 1 |
| `order_id` | 2 |
| `volume` | 2 |
| `note_id` | 3 |
| `playlist_id` | 1 |
| `assignee` | 1 |

## 6. Biến Động Trạng Thái Ứng Dụng (`final_state`)

Những ứng dụng có dữ liệu bị thay đổi sau khi chạy toàn bộ chuỗi tool:

| Ứng Dụng | Số Lượng Mẫu Có Biến Động State |
| :--- | :---: |
| `calendar` | 96 |
| `sms` | 78 |
| `ewallet` | 58 |
| `smart_home` | 107 |
| `ride_hailing` | 51 |
| `alarm` | 86 |
| `bill_payment` | 84 |
| `reminders` | 118 |
| `banking` | 64 |
| `tv` | 15 |
| `phone` | 36 |
| `chat_ott` | 73 |
| `food_delivery` | 32 |
| `medication` | 18 |
| `shopping_list` | 21 |
| `bike_rental` | 28 |
| `learning` | 34 |
| `library` | 19 |
| `subscriptions` | 28 |
| `email` | 35 |
| `door_lock` | 34 |
| `appliances` | 25 |
| `household_chores` | 24 |
| `package_tracking` | 18 |
| `flights` | 28 |
| `tasks` | 50 |
| `notes` | 38 |
| `video_meeting` | 25 |
| `contacts` | 10 |
| `home_automation` | 35 |
| `fitness` | 25 |
| `robot_vacuum` | 32 |
| `music` | 22 |
| `podcast_radio` | 21 |
| `home_security` | 13 |
| `loyalty` | 18 |
| `home_energy` | 13 |
| `news` | 11 |
| `maps` | 12 |

## 7. Đúc Rút Chiến Lược Cho Agent

1. **Tỷ lệ Multi-step cao:** Rất nhiều mẫu yêu cầu gọi tool đọc (như `contacts.search`, `calendar.list_events`, `weather.get_forecast`) trước khi thực hiện thao tác ghi (`sms.send`, `calendar.update_event`). Agent bắt buộc phải có cơ chế **Argument Grounding** truyền ID từ tool result sang tool call kế tiếp.
2. **Cơ chế Thiếu thông tin (`sys.ask_missing_info`):** Phải phân biệt rõ khi nào tham số có thể tra cứu được bằng tool đọc vs khi nào người dùng thực sự không cung cấp (như thiếu `content` trong SMS, thiếu tên bài hát/playlist).
3. **Xử lý Ngoài phạm vi (`[]`):** Khi yêu cầu không liên quan đến app đã cài đặt hoặc là câu chào hỏi, Agent phải dừng ngay và không gọi bất kỳ tool nào để bảo toàn state_hash ban đầu.
