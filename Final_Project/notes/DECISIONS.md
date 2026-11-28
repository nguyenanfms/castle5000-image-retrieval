# Quyết định kỹ thuật

## 20/11/2026

- Giữ baseline ViT-B-32 / laion2b_s34b_b79k và NumPy exact search: corpus 5000 ảnh đủ nhỏ, ưu tiên ổn định trước 23/11. Chưa có bằng chứng cần đổi model.
- Dùng venv ml-gpu đã có, bảo vệ torch 2.14.1+cu130; xác minh CUDA bằng tensor thực. Không đổi driver/torch để chạy dependency. Kiểm tra pip resolver trước cài torchvision còn thiếu.
- Giữ prefix day1_Kitchen_ theo hướng dẫn người dùng; Batch không đại diện ngày. Chưa xác minh live collection; không tạo metadata giả.
- SHA-256 chỉ nhận đúng hai bảng màu mẫu đã biết; không loại toàn bộ đầu chuỗi.
- Corpus thực có 12 filename khớp hai hash đó. Loại đúng các bản sao này, dự kiến 4988 vector, giữ nguyên file gốc.
- Cho phép Pillow >=10,<14 thay vì <13, pin bản đang có khi setup để không hạ Pillow 13.0.0 dùng chung. Resolver không thay dependency cũ; tương thích sẽ được kiểm tra bằng decode toàn corpus, unit tests và CLIP thật.
- Sau baseline 4988 ảnh, screenshot/ảnh gốc cho thấy 17 bảng màu khác hash. Chỉ loại 17 đường dẫn đã xem trong data/exclude.txt (bản sao ở evaluation/calibration_exclusions.txt), không loại theo khoảng frame. Dự kiến index cuối 4971. Giữ các generation baseline và feature cache để so sánh/khôi phục.
- Không đổi model hoặc thêm caption/reranker trong phiên này: baseline đã chạy được nhưng nhãn nội bộ hiện quá thưa để so chất lượng đáng tin. Ưu tiên sửa dữ liệu gây nhiễu rõ ràng; bước tiếp theo là gán nhãn nhiều frame theo sự kiện.
