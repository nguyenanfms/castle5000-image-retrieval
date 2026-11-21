# QA dựa trên bằng chứng

Chế độ `auto` dùng Gemini lập kế hoạch, CLIP/RegionCLIP tìm ảnh, Gemini phân tích ảnh,
Python xử lý dữ liệu có cấu trúc và Gemini kiểm chứng ảnh trước khi trả lời. API trả
về câu trả lời cùng `claims`, ID ảnh, chú thích và vùng ảnh khi có. Các bước và đầu
ra trung gian của một yêu cầu được giữ trong bộ nhớ theo người dùng, có giới hạn số
lượng và thời gian sống; không lưu ảnh hay khóa API. Câu trả lời đã kiểm chứng có thể
được dùng lại khi câu hỏi, người dùng, phiên bản và ảnh nguồn không thay đổi. Cache
không tồn tại sau khi khởi động lại backend.

`processing_steps` trong kế hoạch chỉ nhận các phép được định nghĩa trong
`qa_aggregation.py`. Phép đếm mặc định đếm số bản ghi ảnh; `visible_count` là số
đối tượng có thể nhìn thấy trong từng ảnh và chỉ dùng khi ảnh được đánh dấu đếm đầy
đủ. Tổng số lần xuất hiện không phải số người khác nhau. Các quan hệ, hành động và
danh tính xuyên ảnh không được tự xác nhận bằng phép nối hay loại trùng theo tên.

Câu hỏi toàn tập phải xét tất cả ảnh trong phạm vi và kiểm chứng kết quả trước khi
đánh dấu `examined_scope.exhaustive`. `QA_CORPUS_MAX_IMAGES` mặc định là 128 và
được giới hạn trong khoảng 1–1000; `QA_REQUEST_TIMEOUT_S` vẫn giới hạn thời gian
thực thi. Nếu vượt giới hạn, hết thời gian, có ảnh thiếu hoặc số đếm không rõ, hệ
thống báo chưa đủ bằng chứng. Phép toán toàn tập ngoài `count_max` cần có các bước
tổng hợp hợp lệ trong kế hoạch. Nếu danh mục người đã kiểm duyệt hiện có bao phủ
mọi ảnh trong phạm vi, hệ thống dùng trực tiếp danh mục đó để tính số người lớn nhất.
Nếu không, hiện chưa có chỉ mục số người được xây dựng tăng dần qua nhiều yêu cầu.

RegionCLIP trả về `region_id` theo phiên bản index và vị trí vùng cùng bounding box;
đó là vùng truy xuất, không phải nhãn người được xác nhận. Chỉ số dò đối tượng có
sẵn có thể cung cấp gợi ý, nhưng Gemini vẫn phải xem và kiểm chứng ảnh. Không cộng
người giữa ảnh, không khẳng định danh tính trùng nhau hoặc thời gian nếu không có
dữ liệu đối chiếu đáng tin cậy.

Kiểm thử: `python -m pytest backend/tests/test_qa_reasoning_sync.py backend/tests/test_qa_aggregation_store.py -q`.
Đây là kiểm thử chức năng với dữ liệu và Gemini giả lập, không phải đánh giá độ chính
xác của mô hình trên dataset thật. Hai file kiểm thử này được cho phép theo dõi bởi Git;
các kiểm thử nghiên cứu khác vẫn tuân theo quy tắc bỏ qua hiện có của dự án.
