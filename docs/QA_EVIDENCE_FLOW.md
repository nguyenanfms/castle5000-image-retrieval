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
đó là vùng truy xuất, không phải nhãn người được xác nhận. Gemini vẫn phải xem
và kiểm chứng ảnh. Không cộng
người giữa ảnh, không khẳng định danh tính trùng nhau hoặc thời gian nếu không có
dữ liệu đối chiếu đáng tin cậy.

Với câu hỏi đếm, hệ thống chạy Grounding DINO theo nhãn cần tìm trên các ảnh
đã truy xuất, loại bớt khung trùng và dùng khung này cho crop. Khung D có thể
sai hoặc thiếu; không có khung phù hợp thì không tự thay bằng khung Gemini và
không kết luận số lượng bằng không. Khung được lưu tạm theo ảnh, nhãn và phiên
bản ảnh trong bộ nhớ, không đọc danh mục phát hiện tạo sẵn. Khi hết thời gian,
ảnh chưa dò được giữ trạng thái chưa xác minh.
Grounding DINO được nạp từ `DETECTION_MODEL_DIR` và phải có sẵn trọng số trên máy;
nếu thiếu, hệ thống giữ ảnh ở trạng thái chưa xác minh, không tự tải mô hình.
Cài `transformers` theo `requirements.txt` và đặt đường dẫn này tới thư mục mô hình
`IDEA-Research/grounding-dino-tiny` đã tải trước. KIS giới hạn số ảnh dò bởi
`DETECTION_MAX_IMAGES` (mặc định 50) và thời gian bởi `DETECTION_TIMEOUT_S`
(45 giây); một lượt suy luận CPU đang chạy có thể vượt ngân sách đó.
Hệ thống dùng loại đối tượng từ kế hoạch, duyệt theo
thứ tự ưu tiên ảnh có nhiều vùng Grounding DINO đề xuất nhất, sau khi đã đọc các ảnh.
Ảnh có cùng số vùng được xử lý theo ID để kết quả ổn định khi đổi thứ tự truy xuất.
Số vùng quan sát chưa chắc là số đối tượng thực tế; bộ phát hiện có thể vẽ sai hoặc bỏ sót.
Với câu hỏi tổng số không
chỉ rõ một ảnh, hệ thống ưu tiên ước lượng đối tượng khác nhau trên top 10 ảnh thay vì
yêu cầu chọn một ảnh. Mỗi crop được mã hóa
bằng image encoder CLIP hiện có; các vector CLIP và RegionCLIP không được trộn lẫn.
Hệ thống so cosine với các vùng trong nhóm đã có: nếu có điểm ít nhất
`QA_DISTINCT_SIMILARITY_THRESHOLD` (mặc định 0.85), vùng được xét vào nhóm đó;
nếu thấp hơn mọi vùng thì tạo nhóm mới. Hai đối tượng cùng ảnh không được gộp.
Vùng thiếu embedding được thống kê riêng. Ngưỡng chỉ để thử nghiệm và cần hiệu chỉnh
trên dữ liệu có nhãn; thứ tự xử lý và ngưỡng có thể làm số nhóm thay đổi.

Giao diện hiển thị số nhóm ước lượng, ảnh crop, ảnh nguồn và những ảnh trong nhóm.
Đây là ước lượng chưa xác minh, không phải danh tính chắc chắn hay xác suất. Tập ảnh
đã đọc có thể chỉ là một phần dataset; không dùng ước lượng này như kết quả toàn tập.
Các nhóm được lưu trong kết quả và bộ nhớ tạm của yêu cầu, không phải cơ sở dữ liệu
danh tính bền vững. `QA_DISTINCT_TOP_K` mặc định 10 ảnh và ngân sách thời gian vẫn
áp dụng. Không có bảo đảm mô hình đánh dấu đủ đối tượng hoặc nhận đúng danh tính.
Chế độ tự động dùng `QA_AUTO_TIMEOUT_S` (mặc định 180 giây); với đếm distinct hệ
thống dành một phần thời gian cho kiểm chứng và tính embedding, và báo tỷ lệ ảnh
đã đọc nếu chưa hết top. Khung `D` là gợi ý từ chỉ mục phát hiện có sẵn, chưa được
Gemini xác minh và không tự đưa vào số đếm.

Kiểm thử: `python -m pytest backend/tests/test_qa_reasoning_sync.py backend/tests/test_qa_aggregation_store.py -q`.
Đây là kiểm thử chức năng với dữ liệu và Gemini giả lập, không phải đánh giá độ chính
xác của mô hình trên dataset thật. Hai file kiểm thử này được cho phép theo dõi bởi Git;
các kiểm thử nghiên cứu khác vẫn tuân theo quy tắc bỏ qua hiện có của dự án.
