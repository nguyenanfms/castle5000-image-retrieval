# Castle Local — DS314

- Thi thử sáng 23/11/2026. Thi chính thức đầu tháng 12; 01/12 chưa xác nhận.
- Mục tiêu trước thi thử: baseline ổn định, đủ ảnh, xem khoảnh khắc nhanh, nộp đúng imageID.
- Root `D:\UIT\DS314\Final_Project`; data/Batch1, Batch2, Batch3 có 5000 JPG tổng cộng. Không đổi tên/di chuyển/sửa ảnh gốc.
- Venv đã xác minh `D:\UIT\.venvs\ml-gpu\Scripts\python.exe`; Python 3.14.3, torch 2.14.1+cu130, RTX 3050 Laptop 6 GB. CUDA tensor matmul thành công 20/11/2026; kết quả CLIP xem PROGRESS/EXPERIMENTS.
- FastAPI, web HTML/CSS/JS local, OpenCLIP ViT-B-32 / laion2b_s34b_b79k, L2 + NumPy exact dot product, embedding cache, index generation công bố atomic. Web có **CLIP** và **CLIP + BM25** dùng metadata của partner. Lựa chọn tìm ảnh cũng áp dụng ở bước lấy ứng viên của luồng tự nộp.
- Cache tại `.cache`, kết quả tại `artifacts`, interpreter/constraints tại `.runtime`, đều trên D.
- Batch là phân vùng file, không phải day. Frame không phải timestamp/giây đã xác minh. Nhiều ảnh gần giống nên cần xem lớn/neighbors.
- Có metadata mô tả do partner tạo với 5.000 tên file, trích từ `b317148` vào `.runtime/partner_metadata/b317148` và ghép đủ 4.971 ảnh trong chỉ mục. Dùng bộ này cho tìm kiếm. Quy tắc mã nộp theo PDF `DRES_HuongDan_Castle5000_QA.pdf` là `HH_frame.jpg` → `day1_Kitchen_HH_frame`. Ưu tiên imageID DRES chính thức khi có. Không đổi prefix theo Batch, mã trùng phải chặn nộp.
- DRES API `https://dres.2hbread.com/api/v2`. Cuộc thi có KIS (một `mediaItemName` và collection `castle5000`) và QA (một đáp án `text`, trọng tài chấm, thường trả `202 INDETERMINATE`). Cả hai dùng `POST /submit/{evaluation_id}?session=...` cho evaluation ACTIVE và task đang chạy. Viewer là nguồn câu hỏi và đồng hồ. `200` cần đọc `submission`. Khi timeout, kiểm tra trạng thái bài trước khi gửi lại.
- Castle Local hỗ trợ gửi KIS/QA sau xác nhận và chế độ bật tự nộp cho evaluation. Tự nộp dùng gợi ý do người dùng dán, tạo truy vấn EN, tìm CLIP, rồi dùng model thị giác kiểm chứng ảnh và đọc đáp án QA. Mặc định là Qwen3-VL 2B qua Ollama. Xem README và PROGRESS cho điều kiện chạy và kết quả.
- Theo người dùng, đề thi hiện trên TV và không xuất hiện trên laptop. Luồng thi dự kiến là đọc rồi gõ từng mức bằng tiếng Việt vào Castle Local. Xem [kế hoạch](KE_HOACH_THUC_HIEN.md#2-nhập-đề-từ-tv-và-chuyển-sang-tiếng-anh).
- Không tự gọi/nộp DRES thật; người dùng chủ động dùng tài khoản được cấp. Không lưu password/SessionId. MOCK_ACCEPTED chỉ xác nhận thao tác local.
- Mock tests chỉ kiểm tra logic. Không coi kết quả mock là GPU inference hoặc chất lượng retrieval.

## Trạng thái đã xác minh cuối phiên 20/11/2026

- CLIP ảnh/text CUDA thật thành công; toàn corpus đã build. Bản cuối **4971×512 float32**, L2, không lỗi; Batch1 1824, Batch2 1767, Batch3 1380. 5000 ảnh gốc nguyên trạng, loại 29 bảng màu: 12 khớp hash + 17 path đã xem trong data/exclude.txt. Bản sao danh sách thủ công ở evaluation/calibration_exclusions.txt.
- Server đang chạy localhost:8000 với model cache trên D và HF_HUB_OFFLINE=1; search chạy được không cần gọi HuggingFace. DRES thật vẫn cần mạng khi người dùng tự kết nối.
- **26 tests pass**; Chrome desktop/mobile thực đã kiểm tra gallery, text/image search, neighbors, filters, mock confirmation/history/refresh, lỗi network/model và ứng dụng chưa có index. Không gọi DRES thật.
- Search warm HTTP 9 query: p50 32.164 ms, p95 58.2604 ms; cache query p50 10.262 ms. Nhãn nội bộ thưa: dev Hit@50 1/9, holdout 0/3; chưa đủ để tuyên bố chất lượng thi tốt. Cần mở rộng nhãn nhiều frame theo sự kiện, phân tích query thất bại.
- Doctor CLI trả exit 1 khi có lỗi; scripts doctor/setup dừng đúng. Setup giữ torch/vision/audio/numpy/Pillow đã có; phiên này không package cũ nào bị đổi phiên bản.

Đọc PROGRESS.md trước khi tiếp tục và ghi phát hiện, trạng thái kiểm chứng trước khi kết thúc phiên. [Báo cáo tổng hợp ngày 24/11/2026](BAO_CAO_DU_AN.md) mô tả cấu trúc, cách chạy, tích hợp DRES và lộ trình hai bản.
