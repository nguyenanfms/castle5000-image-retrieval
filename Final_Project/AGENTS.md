# Làm việc với Castle Local

- Trước mỗi phiên, đọc `notes/PROJECT_CONTEXT.md` và `notes/PROGRESS.md`; cập nhật tiến độ, lệnh kiểm chứng và lỗi còn lại trước khi kết thúc.
- Ưu tiên baseline chạy thật; mock chỉ kiểm tra logic. Đọc `notes/DECISIONS.md` và `notes/EXPERIMENTS.md` trước khi đổi retrieval/model.
- Giữ nguyên ảnh gốc, tên file và ba Batch. Không suy Batch thành ngày hay frame thành giây.
- Không tự thay torch/torchvision/torchaudio hoặc driver. Dùng đúng interpreter đã xác minh và constraints của setup.
- Không tự gọi/nộp DRES thật trong kiểm thử; dùng mock local. Không lưu password, SessionId hoặc token vào source/notes/log.
- Không đưa data, model weights, cache, artifacts hay runtime vào git. Báo đúng giới hạn của nhãn nội bộ và kiểm tra thực tế.
