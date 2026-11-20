# Tìm ảnh Castle5000

Ứng dụng tìm kiếm trên 5.000 ảnh Castle5000. CLIP lấy ảnh ứng viên, sau đó RegionCLIP đối chiếu các đối tượng đã chọn và trả về khung trên ảnh gốc. Giao diện chạy tại `/search-v2/`.

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

1. Nhập `Không có ai trong ảnh` và bấm **Tìm**. Ở bước **Từ khóa & điều kiện**, giữ mục vắng người rồi bấm **Tìm lại**. Câu mẫu này hoạt động ngay trên máy.
2. Để thử khung đối tượng, chọn **Nhập điều kiện thủ công**. Nhập `A man in a striped sweater opens a refrigerator` vào **Mô tả toàn cảnh bằng tiếng Anh**. Bấm **Thêm cụm** hai lần để nhập `man in a striped sweater` và `refrigerator`.
3. Bấm **Áp dụng thay đổi**. Giữ hai cụm ở bước đối chiếu rồi bấm **Tìm lại**. Mở một ảnh và bấm từng cụm để xem khung tương ứng hoặc trạng thái thiếu bằng chứng.

Mặc định chương trình dùng DRES demo tại cổng 8090. Khi thử nút gửi ảnh, dùng Run ID `00000000-0000-4000-8000-000000005000`. Phản hồi đến từ máy chủ demo.

## Cấu hình tùy chọn

**Gemini.** Tìm thủ công và truy vấn mẫu hoạt động với cấu hình mặc định. Để Gemini tự phân tích câu tiếng Việt hoặc tiếng Anh bất kỳ, tạo `.env` từ mẫu nếu chưa có.

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Trên macOS hoặc Linux, dùng `test -f .env || cp .env.example .env`. Mở `.env`, điền `SEARCH_PLAN_API_KEY` bằng khóa của bạn rồi khởi động lại web. Có thể đặt `SEARCH_PLAN_MODEL` theo model mà khóa truy cập được. Nếu Gemini lỗi, dùng **Nhập điều kiện thủ công**.

**QA.** Đặt `QA_ENABLED=true` trong `.env` và cấu hình nhà cung cấp theo các biến `QA_*` trong `.env.example`.

**DRES thật.** Đặt `DRES_BASE_URL`, `DRES_EVALUATION_ID` và `SESSION_SECRET` trong `.env`, rồi thêm `--real-dres` vào lệnh chạy. Dùng tài khoản DRES được cấp. Giữ khóa và cấu hình riêng trong `.env`.

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

Phần tích hợp DRES kế thừa từ [castle5000_contest](https://github.com/linh222/castle5000_contest). Trọng số tìm kiếm toàn ảnh lấy từ [LAION OpenCLIP ViT-B/32](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K).
