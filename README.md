# Tìm ảnh Castle5000

Ứng dụng tìm ảnh và hỏi đáp trên 5.000 ảnh Castle5000. CLIP lấy ảnh ứng viên, RegionCLIP đối chiếu các đối tượng đã chọn và trả về khung trên ảnh gốc. Khi có khóa Gemini, hệ thống kiểm tra lại các ảnh đầu, chỉ ra bằng chứng cho từng yêu cầu, và trả lời câu hỏi về ảnh. Giao diện chạy tại `/search-v2/`.

Sơ đồ hoạt động: mở `docs/so-do-he-thong/index.html` nếu bạn có thư mục này trên máy.

## 1. Chuẩn bị

- Cài [Python 3.13 bản 64-bit](https://www.python.org/downloads/), [Git](https://git-scm.com/install/) và [Git LFS](https://git-lfs.com/).
- Chừa khoảng 15 GB trên ổ đĩa cho dữ liệu, chỉ mục và thư viện. Có thể chạy bằng CPU.
- Cần Internet để clone và cài thư viện. Git LFS tải khoảng 3,5 GiB ảnh, chỉ mục và trọng số trong lần clone đầu tiên.

## 2. Clone dự án

```sh
git --version
git lfs version
git lfs install
git clone https://github.com/nguyenanfms/castle5000-image-retrieval.git
cd castle5000-image-retrieval
git lfs ls-files --size
```

Sau khi clone, `git lfs ls-files --size` sẽ liệt kê ba tệp trong `runtime/` với kích thước ở mức hàng trăm MB hoặc GB. Nếu còn tệp dạng con trỏ, chạy `git lfs pull` trong thư mục vừa clone. Dùng lệnh clone ở trên để lấy các tệp Git LFS.

## 3. Cài và khởi động

**Windows (PowerShell)**

```powershell
py -3.13 --version
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tools/start_local.py
```

Nếu Windows nhận lệnh `python` nhưng thiếu lệnh `py`, kiểm tra `python --version` rồi dùng `python -m venv .venv` khi đó là Python 3.13.

**macOS hoặc Linux**

```sh
python3.13 --version
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tools/start_local.py
```

Lần đầu chương trình giải nén dữ liệu vào `runtime-data/` rồi nạp hai chỉ mục. Giữ terminal mở và đợi dòng `Open http://127.0.0.1:8000/search-v2/`. Mở [trang tìm ảnh](http://127.0.0.1:8000/search-v2/) và đăng nhập bằng tên **student**, mật khẩu **student**. Bấm Ctrl+C trong terminal để dừng web và DRES demo. Những lần mở lại, chạy lại lệnh `tools/start_local.py` bằng Python trong `.venv`.

Bạn có thể mở [trang trạng thái](http://127.0.0.1:8000/health) để kiểm tra web đã nạp đủ ảnh và chỉ mục vùng.

```json
{"status":"ok","images":5000,"region_ready":true}
```

## 4. Tìm ảnh lần đầu

1. Nhập mô tả ảnh bằng tiếng Việt hoặc tiếng Anh rồi bấm **Tìm kiếm**. Dải **Quá trình suy nghĩ** hiện 5 bước, mỗi bước có thời gian và một dòng kết quả: **Hiểu câu → Tìm → Lọc điều kiện → Gemini kiểm → Phóng to**.
2. Bấm vào bước **Hiểu câu** để xem cụm đối tượng và điều kiện Gemini tách ra. Bỏ chọn mục không cần rồi bấm **Tìm lại**.
3. Ảnh có nhãn **Khớp**, **Một phần** hoặc **Không khớp** là ảnh đã được Gemini kiểm. Ảnh khớp được xếp lên đầu, ảnh không khớp xuống cuối, ảnh chưa kiểm giữ thứ tự tìm kiếm. Muốn kiểm tiếp, bấm **Kiểm thêm 4 ảnh** trong bước Gemini kiểm.
4. Mở một ảnh để xem lý do, trạng thái từng yêu cầu và vùng bằng chứng. Dải **Khung lân cận cùng giờ** (hoặc phím ← →) giúp xác nhận đúng sự kiện trước khi nộp.
5. Không có Gemini thì chọn **Nhập điều kiện thủ công**. Ví dụ: nhập `A man in a striped sweater opens a refrigerator` vào **Mô tả toàn cảnh bằng tiếng Anh**, bấm **Thêm cụm** hai lần để nhập `man in a striped sweater` và `refrigerator`, rồi bấm **Áp dụng thay đổi** và **Tìm lại**.

Một sự kiện trên DRES thường gồm nhiều khung hình liên tiếp, và nộp một khung bất kỳ trong đó là được tính đúng. Điều kiện "không có người" được kiểm bằng chỉ mục vùng sẵn có. Các điều kiện đếm khác hiện **Chưa kiểm tra** trừ khi bật `SEARCH_DETECTION_ENABLED`.

Mặc định chương trình dùng DRES demo tại cổng 8090. Khi thử nút gửi ảnh, dùng Run ID `00000000-0000-4000-8000-000000005000`. Phản hồi đến từ máy chủ demo.

## Cấu hình tùy chọn

**Gemini.** Chế độ tìm thủ công nhận mô tả tiếng Anh và điều kiện do bạn nhập. Để Gemini tự phân tích câu tiếng Việt hoặc tiếng Anh, tạo `.env` từ mẫu nếu chưa có.

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Trên macOS hoặc Linux, dùng `test -f .env || cp .env.example .env`. Mở `.env`, điền `SEARCH_PLAN_API_KEY` bằng khóa của bạn rồi khởi động lại web. Có thể đặt `SEARCH_PLAN_MODEL` theo model mà khóa truy cập được. Nếu Gemini lỗi, dùng **Nhập điều kiện thủ công**.

**Gemini dự phòng.** Đặt thêm `QA_API_KEY_BACKUP` (và `SEARCH_PLAN_API_KEY_BACKUP` nếu dùng khóa riêng cho phần tìm ảnh). Khi khóa chính hết quota (429) hoặc quá tải (503), hệ thống gọi lại bằng khóa dự phòng với **cùng model**. Khóa hết quota được nghỉ 10 phút. Nên dùng hai khóa thuộc hai project khác nhau, vì các khóa cùng project dùng chung quota.

**Quota Gemini mỗi thao tác**

| Thao tác | Số lượt gọi |
|---|---|
| Một lần tìm ảnh | tối đa 3: phân tích câu, kiểm 4 ảnh đầu, phóng to bằng chứng |
| Tìm lại đúng câu cũ | 0, lấy từ bộ nhớ tạm |
| **Kiểm thêm 4 ảnh** | 1–2 |
| Một câu hỏi đáp | tùy câu; dừng sớm khi đã đủ bằng chứng |

Tắt công tắc **Gemini kiểm 4 ảnh đầu** dưới ô tìm để mỗi lần tìm chỉ tốn 1 lượt.

**QA.** Đặt `QA_ENABLED=true` trong `.env` và cấu hình nhà cung cấp theo các biến `QA_*` trong `.env.example`.

**Hỏi đáp.** Tab **Hỏi đáp** hiện các bước đang chạy, thời gian, số ảnh đã đọc, và có nút **Dừng**. Grounding DINO dò người và vật trên CPU bằng tối đa 8 luồng, chạy song song với Gemini. Câu đếm người hoặc vật khác nhau có thanh kéo **ngưỡng cosine t** (mặc định 0,76): kéo đến đâu, câu trả lời và các nhóm được tính lại đến đó, không gọi thêm Gemini. Nếu Gemini lỗi giữa chừng, câu trả lời dựa trên các ảnh đã đọc và có ghi chú giới hạn.

**Nộp đáp án QA.** Trong tab **Hỏi đáp**, mở **Nộp đáp án QA**, nhập Run ID hoặc đường dẫn Viewer rồi bấm **Chọn**. Nhập đáp án ngắn gọn và bấm **Nộp đáp án**. Nút **Dùng làm đáp án** ở mỗi câu trả lời giúp đưa nội dung vào ô để chỉnh sửa trước khi nộp. Hệ thống kiểm tra task hiện tại là QA và gửi một trường `text` theo hướng dẫn DRES. Trạng thái **ĐANG CHỜ CHẤM** nghĩa là DRES đã nhận và chờ trọng tài.

Để kiểm thử trên DRES demo, dùng Run ID `00000000-0000-4000-8000-000000005001` cho task QA. Đây là môi trường giả lập cục bộ. Để dùng DRES cuộc thi, chạy cấu hình DRES thật ở mục bên dưới và đăng nhập bằng tài khoản do ban tổ chức cấp.

Trong chế độ hỏi tự động, QA tách điều kiện chọn cảnh khỏi phép xử lý, sau đó kiểm tra từng điều kiện trên ảnh. Khi trả lời số đếm, bấm ảnh bằng chứng để xem các vùng được tính. Câu hỏi tổng số người áp dụng cho người nhìn thấy trong phạm vi ảnh. Nếu danh mục phát hiện có sẵn, QA rà danh mục để chọn ảnh đối chiếu và ghi rõ mức bao phủ. Kết quả tự động có thể báo số tối thiểu dựa trên một ảnh có nhiều người xuất hiện đồng thời.

Để trả tổng số người khác nhau trên một phạm vi đã kiểm duyệt, dùng `QA_PEOPLE_CATALOG_PATH` trỏ tới một tệp JSON cục bộ. Tệp cần có một mục cho **mọi ảnh trong phạm vi**, kể cả ảnh không có người. Dùng cùng mã ẩn danh cho cùng một người qua các ảnh, với tọa độ `[x1, y1, x2, y2]` trong khoảng 0 đến 1 trên ảnh gốc. Ví dụ định dạng:

```json
{
  "images": {
    "08_1002.jpg": [{"label": "P1", "box": [0.1, 0.2, 0.4, 0.9]}],
    "08_1010.jpg": []
  }
}
```

Danh mục này cần được rà thủ công trước khi sử dụng. Khởi động lại ứng dụng sau khi cập nhật tệp. Hai bản ghi trong ví dụ chỉ minh họa định dạng.

**DRES thật.** Đặt `DRES_BASE_URL`, `DRES_EVALUATION_ID` và `SESSION_SECRET` trong `.env`, rồi thêm `--real-dres` vào lệnh chạy. Dùng tài khoản DRES được cấp. Giữ khóa và cấu hình riêng trong `.env`.

**Tinh chỉnh**

| Biến trong `.env` | Mặc định | Ý nghĩa |
|---|---|---|
| `SEARCH_VERIFY_IMAGES` | 4 | Số ảnh đầu Gemini tự kiểm sau mỗi lần tìm |
| `SEARCH_DETECTION_ENABLED` | false | Chạy Grounding DINO cho điều kiện đếm khi tìm ảnh (chậm trên CPU) |
| `SEARCH_PLAN_TIMEOUT_S` | 20 | Thời gian chờ Gemini phân tích câu |
| `QA_CONCURRENCY` | 1 | Số câu hỏi đáp chạy cùng lúc; 2 phù hợp máy 8 nhân |
| `QA_AUTO_TIMEOUT_S` | 180 | Thời gian tối đa cho một câu hỏi đáp (tối đa 300) |
| `QA_EARLY_STOP` | true | Dừng đọc ảnh khi đã đủ bằng chứng |
| `QA_DISTINCT_SIMILARITY_THRESHOLD` | 0,76 | Ngưỡng ghép cùng một người hoặc vật giữa các ảnh |
| `DETECTION_THREADS` | 0 | Số luồng CPU cho Grounding DINO; 0 là số nhân thật, tối đa 8 |

## Dữ liệu đi kèm

| Tệp trong `runtime/` | Nội dung |
|---|---|
| `castle5000-media.zip` | 5.000 ảnh gốc và 5.000 ảnh thu nhỏ có cùng tên với ảnh gốc |
| `castle5000-indexes.zip` | Chỉ mục CLIP, RegionCLIP đã xây dựng, bộ mã hóa văn bản RegionCLIP và dữ liệu đối chiếu điều kiện |
| `open_clip_model.safetensors` | Trọng số CLIP ViT-B/32 dùng cho truy vấn |

Trình chạy dùng dữ liệu đóng gói và trọng số trên máy. Bạn có thể chạy `tools/start_local.py --prepare` với Python để chỉ giải nén. Thư mục `runtime-data/` được tạo tự động.

`docker-compose.yml` dành cho môi trường đã có DRES và proxy HTTPS trên cùng mạng Docker. Giải nén bằng `--prepare` và đặt cấu hình DRES trước khi khởi động dịch vụ. Proxy chuyển tiếp đến cổng 5000 của container.

GitHub tính lượt tải ba tệp trên vào hạn mức Git LFS của chủ repo. [Cách GitHub tính dung lượng Git LFS](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)

## Khi gặp lỗi

- **Thiếu tệp Git LFS hoặc ZIP quá nhỏ**. Chạy `git lfs install` rồi `git lfs pull` trong thư mục clone.
- **Cổng đang được dùng**. Thêm `--port 8005 --dres-port 8095` vào lệnh chạy rồi mở [trang ở cổng 8005](http://127.0.0.1:8005/search-v2/).
- **Chỉ mục chưa sẵn sàng**. Đọc thông báo trong terminal và kiểm tra [trạng thái nạp chỉ mục](http://127.0.0.1:8000/health). Cấu hình chạy này cần thấy `images: 5000` và `region_ready: true` trước khi tìm.
- **Gemini báo lỗi cấu hình**. Kiểm tra `SEARCH_PLAN_API_KEY` và `SEARCH_PLAN_MODEL` hoặc dùng phần chỉnh mô tả tiếng Anh.
- **Gemini 503 (quá tải) hoặc 429 (hết quota)**. 503 thường tự hết sau ít giây; bấm **Kiểm lại** hoặc hỏi lại. Với 429, thêm `QA_API_KEY_BACKUP` hoặc tắt công tắc kiểm tra bằng Gemini. Phần tìm ảnh bằng CLIP và RegionCLIP vẫn chạy khi Gemini lỗi.

Phần tích hợp DRES kế thừa từ [castle5000_contest](https://github.com/linh222/castle5000_contest). Trọng số tìm kiếm toàn ảnh lấy từ [LAION OpenCLIP ViT-B/32](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K).
