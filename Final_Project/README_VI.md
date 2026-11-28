# Castle Local — DS314

Bản 0.1, chuẩn bị cho thi thử 23/11/2026. Đây là hệ thống local do nhóm có thể tiếp tục phát triển, dùng OpenCLIP pretrained; không phải mô hình được huấn luyện riêng. Tham khảo cách tổ chức bài toán và DRES của thầy, code ứng dụng trong gói này được viết mới.

**Cập nhật 25/11/2026.** Castle Local hỗ trợ nộp **KIS** và **QA**. Cả hai có chế độ gửi sau khi xác nhận và chế độ tự nộp theo bằng chứng. Xem mục 5 để luyện trên máy hoặc kết nối DRES.

## 1. Đặt code đúng chỗ

Mở PowerShell trong `Final_Project/` của nhánh `ph`. Đặt bộ ảnh Castle5000 đã được cấp vào `data/`, giữ nguyên ba thư mục Batch và tên ảnh. File `config.json` và các thư mục `castle`, `web`, `scripts` nằm ngang hàng với `data`. Có thể trỏ `data_dir` trong `config.json` tới thư mục ảnh hiện có để dùng chung dữ liệu trên máy.

```
Final_Project/
  config.json
  requirements.txt
  castle/
  web/
  scripts/
  tests/
  data/
    Batch1/ ... anh ...
    Batch2/ ... anh ...
    Batch3/ ... anh ...
```

Gói không chứa và không sửa ảnh gốc. Chỉ đọc ảnh trong data; ghi index, thumbnail, lịch sử vào artifacts; model cache vào .cache của dự án (trên ổ D nếu dự án ở D).

## 2. Cài bằng venv ml-gpu có sẵn

Mở PowerShell trong VS Code:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup.ps1
```

Script tìm `D:\UIT\.venvs\ml-gpu\Scripts\python.exe`, sau đó `D:\UIT\ml-gpu\Scripts\python.exe`. Nếu venv nằm nơi khác, truyền đường dẫn thật:

```powershell
.\scripts\setup.ps1 -Python "D:\UIT\DUONG_DAN_THAT\ml-gpu\Scripts\python.exe"
```

Script lưu pip freeze trước khi cài ở `.runtime/environment_before.txt`, dùng constraints giữ nguyên các bản torch/torchvision/torchaudio/numpy đã có. Không dùng `--upgrade`, không tự cài lại CUDA/driver, không tạo venv mới. Nếu resolver báo xung đột thì dừng và gửi nguyên lỗi; không tự hạ torch. Lưu ý: cài thêm open_clip vẫn có thể bổ sung/thay phiên bản các dependency khác; nếu venv đang dùng cho công việc quan trọng, cân nhắc venv riêng sau buổi thi thử.

Constraints cũng giữ Pillow hiện có; requirements cho phép Pillow 13 đã kiểm chứng trên máy này. Doctor trả mã lỗi nếu dependency/import/data/mapping không hợp lệ và setup dừng theo mã đó; CPU fallback vẫn được cho phép.

`doctor` hiển thị Python thật, import torch/vision/open_clip, CUDA, GPU, số ảnh mỗi Batch, số mã DRES trùng/chưa nhận dạng. CUDA False vẫn cho chạy CPU. Chỉ kết luận máy chạy GPU sau khi doctor in CUDA True.

## 3. Tạo chỉ mục

```powershell
.\scripts\build_index.ps1
```

Lần đầu cần internet để tải checkpoint ViT-B-32 / laion2b_s34b_b79k. Tải xong, dữ liệu và model nằm trên máy, truy vấn không gọi dịch vụ AI trả phí. Máy chủ DRES vẫn cần internet khi thi.

RTX 3050 Laptop 6 GB: bắt đầu batch 16, không mở thêm model lớn cùng lúc. Nếu OOM, script giảm batch và thử lại; nếu batch 1 vẫn không đủ, dừng ứng dụng GPU khác hoặc dùng CPU:

```powershell
.\scripts\build_index.ps1 -BatchSize 8
.\scripts\build_index.ps1 -Device cpu -BatchSize 4
```

Embedding từng ảnh được cache; ngắt build có thể chạy lại. Index cũ chỉ được thay sau khi thế hệ index mới hoàn tất. Ảnh hỏng được ghi trong `artifacts/build_report.json`; số ảnh lập index có thể thấp hơn số file, cần xem báo cáo. Hai ảnh bảng màu mẫu được loại theo SHA-256 chính xác. Các bảng màu khác không được đảm bảo tự nhận diện: thêm filename hoặc relative path vào `data/exclude.txt`, mỗi dòng một ảnh, rồi build lại. Không xóa ảnh gốc.

Máy local ngày 20/11/2026 có 5000 ảnh đọc được, 12 file trùng hai hash đã biết và 17 bảng màu khác đã xem trực tiếp. Bản index cuối có **4971 ảnh**, 0 lỗi. Danh sách 17 path cụ thể nằm trong `data/exclude.txt`; bản sao ở `evaluation/calibration_exclusions.txt` để khôi phục khi chuyển máy. Không loại theo khoảng frame và không khẳng định đã nhận diện mọi bảng màu khác.

Trước build đầu tiên có thể chạy kiểm tra nhỏ bằng CLIP thật:

```powershell
& "D:\UIT\.venvs\ml-gpu\Scripts\python.exe" scripts\smoke_clip.py
```

Nếu đổi model, pretrained hoặc item_prefix trong config.json, chạy build lại rồi khởi động lại server. Đổi model sẽ tạo cache riêng, không trộn embedding các model.

## 4. Mở ứng dụng

```powershell
.\scripts\start.ps1
```

Mở **http://127.0.0.1:8000** bằng Chrome/Edge. Chạy một server/worker; chỉ lắng nghe localhost. Dừng bằng Ctrl+C. Nếu cổng bận: `.\scripts\start.ps1 -Port 8001`.

**Thử nhanh với index hiện có.** Chọn giờ **08:00** và tìm `a man in a gray sweater adjusting the head camera of a man in black`. Trong lần kiểm tra 22/11, `08_269.jpg` đứng hạng 3 với bộ lọc này. Mở ảnh, so sánh các frame lân cận và kiểm tra mã `day1_Kitchen_08_269`. Dùng chế độ luyện local ở mục 5 để thử nộp ảnh.

- Chưa kết nối DRES vẫn tìm được ảnh.
- Viết mô tả **tiếng Anh ngắn**: hành động, vật thể, quần áo, vị trí. Bản này chưa dịch tự động. CLIP có giới hạn token; mô tả dài bị cắt.
- Lọc theo Batch hoặc phần giờ của tên file; giờ này không được suy diễn thành timestamp chính xác.
- Bấm ảnh để xem lớn, các frame lân cận và tên DRES. Các frame liền nhau chỉ là các ảnh đang có trong tập, không khẳng định liên tiếp theo từng giây.
- “Tìm ảnh tương tự” dùng vector của ảnh đang chọn để tìm lại.
- Điểm cosine không phải xác suất trả lời đúng. Cần nhìn ảnh trước khi nộp.
- Lần tìm chữ đầu tiên nạp model nên có thể chậm. Thử trước giờ thi.

### So CLIP và CLIP + BM25 trên web

Chọn **Phương pháp tìm ảnh** ở thanh bên. Khi metadata đã nạp, web mở ở **CLIP + BM25**. Gõ cùng một câu tiếng Anh và đổi sang **CLIP** để xem lại thứ hạng. Lựa chọn này cũng áp dụng khi bấm **Xử lý gợi ý**.

`partner_repo_dir` trỏ tới thư mục cha chứa `backend/app/metadata_search.py`. `image_metadata_dir` mặc định trỏ tới `runtime-data/image-metadata` ở gốc repo. Nếu đã có metadata mô tả từ bản local trước đó, có thể trỏ `image_metadata_dir` tới thư mục chứa `manifest.json` và `metadata.jsonl` của bản đó. Với clone mới, metadata nằm trong `runtime/castle5000-siglip2.zip`. Có thể tải và giải nén runtime theo [README gốc](../README.md#2-clone-dự-án) rồi dùng `python tools/start_local.py --prepare` từ gốc repo. Castle Local dùng metadata đó với trọng số `metadata_weight=0.5` và tạo chỉ mục CLIP riêng theo mục 3. Nếu thiếu metadata, web dùng CLIP. Sau khi chỉnh cấu hình, khởi động lại server.

## 5. Luyện local và nộp KIS, QA qua DRES

Theo `DRES_HuongDan_Castle5000_QA.pdf`, mục 3–10, DRES Viewer hiển thị đề, đồng hồ và bảng điểm. Bài nộp đi qua API DRES.

| Task | Nhận biết | Mỗi lần nộp | Cách chấm |
|---|---|---|---|
| **KIS** | Tên bắt đầu bằng `KIS`, `taskType` là `LSC Known-Item Search` | Mã của **một** ảnh đúng, gồm `mediaItemName` và `mediaItemCollectionName` | DRES chấm tự động |
| **QA** | Tên bắt đầu bằng `QA`, `taskType` là `LSC Question & Answer Text` | **Một** câu trả lời trong trường `text` | Trọng tài chấm. DRES thường trả `202 INDETERMINATE` khi nhận bài |

### Luyện KIS và QA trên máy

Chọn **Luyện local → Kết nối luyện local**. Trong danh sách buổi thi, chọn bài luyện tương ứng:

- **KIS · Luyện tìm ảnh và nộp thử** (`local-practice`). Mở một ảnh, bấm **Nộp thử local** và xác nhận.
- **QA · Luyện nộp đáp án chữ** (`local-practice-qa`). Tìm và xem ảnh liên quan, rồi nhập **Đáp án QA** ở thanh bên. Bấm **Nộp thử QA** và xác nhận.

`MOCK_ACCEPTED` xác nhận thao tác trên máy và xuất hiện trong **Lịch sử nộp trong phiên**. Mock không chấm đúng/sai và không gửi DRES thật.

### Tự nộp KIS và QA

1. Chuẩn bị model mặc định `qwen3-vl:2b` bằng `ollama pull qwen3-vl:2b` và để Ollama phục vụ tại `127.0.0.1:11434`. Web local kết nối Ollama từ backend. Có thể đặt `CASTLE_OLLAMA_MODEL` trước khi chạy server nếu dùng model thị giác khác.
2. Chọn đúng bài luyện local và bấm **Bật tự nộp**. Dán đề đang hiện vào ô **Đề và các gợi ý đã hiện** rồi bấm **Xử lý gợi ý**. Khi có mức mới, thêm dòng vào cùng ô và xử lý lại. Nếu Viewer đã chuyển task, bấm **Cập nhật task** trước khi dán đề mới.
3. App tạo câu tìm tiếng Anh, tìm ứng viên bằng CLIP và đánh số tối đa tám ảnh để model đối chiếu. Với KIS, app mô tả lại riêng ảnh được chọn rồi kiểm tra mô tả có hỗ trợ chi tiết trong đề. Khi đủ bằng chứng, app tự gửi mã ảnh KIS hoặc câu trả lời QA. Xem **Lịch sử nộp** và **Xem ảnh bằng chứng** để kiểm tra.
4. Nếu app báo thiếu bằng chứng, thêm gợi ý mới. Mỗi task có tối đa một lần tự nộp trong phiên local. **Tắt tự nộp** để chuyển sang thao tác xác nhận bằng tay.

Chế độ DRES thật yêu cầu bật tự nộp riêng cho evaluation. Hãy dùng chế độ luyện để đo chất lượng và thời gian trước. Với QA cần tổng hợp nhiều thời điểm hoặc toàn bộ kho, luồng đọc một ảnh sẽ yêu cầu thêm bằng chứng. Nếu dùng Gemini theo thể lệ cho phép, đặt `CASTLE_AUTO_PROVIDER=gemini`, `CASTLE_GEMINI_MODEL` và `GEMINI_API_KEY` trong môi trường của tiến trình server rồi mở lại app. Ở chế độ đó, câu hỏi và các ảnh ứng viên được gửi tới Gemini.

Câu QA luyện nhanh là **Các chụp đèn treo phía trên đảo bếp có màu gì?** Lần xử lý đầu có thể lâu do nạp cả CLIP và model thị giác. Hãy chạy một lượt luyện trước khi tính thời gian các gợi ý.

### Thử Gemini 4 Hydrogen trên cổng 8001

Mở một cửa sổ PowerShell trong `Final_Project/`. Nhập khóa Gemini API khi được nhắc. Khóa được dùng trong môi trường tiến trình server.

```powershell
$env:CASTLE_AUTO_PROVIDER = 'gemini'
$env:CASTLE_GEMINI_MODEL = 'gemini-4-hydrogen'
$env:GEMINI_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'Gemini API key' -AsSecureString)).Password
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start.ps1 -Port 8001
```

Mở `http://127.0.0.1:8001` trong cửa sổ ẩn danh để thử với phiên luyện riêng. Chọn **Luyện local**, chọn KIS hoặc QA rồi bấm **Bật tự nộp → Xử lý gợi ý**. Hydrogen tạo truy vấn tiếng Anh trước khi CLIP + BM25 tìm ảnh local. Sau khi tìm, Hydrogen đọc các ảnh ứng viên để kiểm chứng KIS hoặc trả lời QA. Adapter dùng mức suy luận **High**. [Tài liệu Gemini 4 Hydrogen](https://ai.google.dev/gemini-api/docs/models/gemini-4-hydrogen) · [Hướng dẫn chuyển model](https://ai.google.dev/gemini-api/docs/whats-new-gemini-4-hydrogen)

### Chuẩn bị cho buổi thi

1. Đăng nhập [DRES](https://dres.2hbread.com) bằng tài khoản BTC cấp. Trên trang **Evaluations**, xác nhận đúng buổi thi, ghi **Run ID** (Evaluation ID) và kiểm tra trạng thái `ACTIVE`.
2. Mở **Viewer** bằng biểu tượng con mắt. Đọc câu hỏi, kiểm tra loại task đang chạy và theo dõi thời gian ở đó. Thống nhất một người nộp bài cho nhóm.
3. Nếu dùng API thủ công, lấy `SessionId` trong [Profile](https://dres.2hbread.com/user) hoặc lấy trường `sessionId` từ phản hồi `POST /api/v2/login`. Giữ thông tin đăng nhập và Session ID riêng, không ghi vào repo hoặc chia sẻ ảnh chụp có chứa chúng. Khi tích hợp vào app, có thể gọi `GET /api/v2/client/evaluation/currentTask/<EVALUATION_ID>?session=<SESSION_ID>` để đọc `taskType`.
4. Kiểm tra luồng nộp trong buổi luyện do BTC mở. Theo PDF, nếu kiểm tra kết nối khi chưa có task và nhận `400` kèm `wrong state`, yêu cầu đã đến DRES nhưng chưa có task nhận bài. Bài sai trong task đang tính điểm bị trừ điểm.

Castle Local lấy task khi chọn buổi thi. Khi Viewer chuyển câu, bấm **Cập nhật task** để hiện đúng ô nộp bài. App kiểm tra lại task ngay trước khi gửi, và yêu cầu xem lại câu hỏi nếu task đã đổi.

### Nộp KIS từ Castle Local

Ngắt kết nối luyện local nếu đang dùng, chuyển sang **DRES thật**, nhập tài khoản BTC cấp và chọn đúng buổi thi. Khi Viewer hiện task **KIS**, mở ảnh đã kiểm tra, xem **Mã nộp DRES**, bấm **Nộp ảnh lên DRES thật** và xác nhận.

Ví dụ `08_269.jpg` có mã `day1_Kitchen_08_269`. Collection là `castle5000`. Một sự kiện có thể có nhiều ảnh được chấp nhận, nhóm chỉ cần nộp một ảnh đúng. Nếu có metadata chứa `imageID` chính thức, dùng đúng giá trị đó. Castle Local chặn mã ảnh trùng hoặc chưa nhận dạng.

### Nộp QA từ Castle Local

Khi Viewer hiện task **QA**, tìm và xem ảnh làm bằng chứng. Đóng khung xem ảnh, nhập đáp án ngắn vào ô **Đáp án QA** ở thanh bên, bấm **Nộp đáp án lên DRES thật** và xác nhận. Dùng ngôn ngữ của câu hỏi, trừ khi BTC quy định khác. Ví dụ nhập `3` khi hỏi số lượng hoặc `red` khi hỏi màu bằng tiếng Anh. App cắt khoảng trắng ở hai đầu và gửi duy nhất trường `text`. Nếu ô QA chưa hiện, bấm **Cập nhật task**.

Trong chế độ tự nộp, model đề xuất đáp án có ảnh làm bằng chứng và app tự gửi khi task vẫn khớp.

### Phương án nộp QA qua API

Trong Postman, chọn **POST**, thay các giá trị giữ chỗ trong URL dưới đây và chọn **Body → raw → JSON** (`Content-Type: application/json`). Kiểm tra lại Viewer đang ở đúng task QA trước khi bấm **Send**. Cả hai loại bài dùng endpoint này và DRES tự gắn bài vào task đang chạy, nên không gửi `taskId`.

```text
https://dres.2hbread.com/api/v2/submit/<EVALUATION_ID>?session=<SESSION_ID>
```

Body QA chỉ có trường `text` trong đối tượng đáp án. Số `3` chỉ minh họa cú pháp.

```json
{"answerSets":[{"answers":[{"text":"3"}]}]}
```

Để đối chiếu, body của KIS chứa mã ảnh và collection.

```json
{"answerSets":[{"answers":[{"mediaItemName":"day1_Kitchen_08_269","mediaItemCollectionName":"castle5000"}]}]}
```

### Đọc phản hồi nộp bài

| Phản hồi | Cách xử lý |
|---|---|
| `200 CORRECT` hoặc `200 WRONG` | Đọc trường `submission` để biết đúng hay sai. QA có thể có kết quả ngay khi cùng đáp án đã được chấm trước đó. |
| `202 INDETERMINATE` | Đã nhận và đang chờ chấm. Điểm QA tính theo lúc gửi bài. Chờ kết quả trên bảng điểm, vì mỗi biến thể đáp án sai có thể bị trừ điểm riêng. |
| `400 wrong state` | Kiểm tra Viewer xem task đã bắt đầu hay hết giờ. |
| `400` hoặc `500` kèm `Could not find media item` | Với KIS, kiểm tra `imageID` và collection `castle5000`. |
| `401` hoặc `404` | Với `401`, đăng nhập lại. Với `404`, kiểm tra Evaluation ID và quyền tham gia. |
| `412` | Đọc chi tiết lỗi để phân biệt sai loại bài, nộp trùng hoặc nhóm đã có bài đúng. |
| Timeout hoặc phản hồi không xác định | Kiểm tra Viewer và trạng thái bài trước khi quyết định gửi lại. |

Castle Local giữ phiên DRES trong RAM, hỏi xác nhận trước khi nộp ảnh hoặc đáp án và không tự gửi lại `POST` khi lỗi kết nối. Nhật ký `artifacts/submissions.jsonl` ghi ảnh hoặc đáp án và kết quả thao tác, không ghi mật khẩu hoặc Session ID.

**Trạng thái kiểm thử trên máy.** Đã kiểm tra CLIP CUDA, điều hướng ảnh và hai luồng nộp KIS/QA bằng mock. Test API giả lập xác minh payload QA và trạng thái `202 INDETERMINATE`. DRES thật cần kiểm tra trong buổi luyện do BTC mở.

## 6. Nếu nhóm có metadata CSV

File CSV không nằm trong các tệp đã cung cấp, nên gói không tự giả định đã có caption.

Trong config.json, đặt `metadata_csv` là đường dẫn CSV tương đối với dự án hoặc đường dẫn tuyệt đối. Cột `imageID` được dùng nguyên giá trị để nộp. Khóa nối ưu tiên `relative_path`, rồi `filename`; nếu chỉ có hour/frame thì dựng tên HH_frame.jpg.

```csv
relative_path,imageID,description
Batch1/08_262.jpg,day1_Kitchen_08_262,People in a kitchen
```

Nếu các Batch có file trùng tên nhưng thuộc ngày/nguồn khác nhau, metadata cần relative_path và imageID đúng cho từng file. Không coi Batch1/2/3 là day1/2/3. Nếu không có metadata, mặc định item_prefix theo tài liệu thầy là day1_Kitchen_; đối chiếu collection thực tế trước khi thi.

Caption metadata hiện chỉ được hiển thị khi xem ảnh, **chưa dùng BM25/reranking** trong phiên bản này.

## 7. Đánh giá nhanh trước thi

Tạo file queries.jsonl, mỗi dòng một JSON. Chọn ảnh và tự viết truy vấn; relevant là danh sách imageID được chấp nhận. Không coi ví dụ dưới đây là nhãn thật đã được BTC xác nhận:

```json
{"query":"a person adjusting another person's head camera","relevant":["day1_Kitchen_08_269"]}
```

Chạy bằng Python của ml-gpu:

```powershell
& "D:\UIT\.venvs\ml-gpu\Scripts\python.exe" -m castle evaluate queries.jsonl
```

Kết quả là Hit@1/5/10/50 và MRR@200, ghi artifacts/evaluation_report.json. Hit@K nghĩa là ít nhất một ảnh đúng vào top K, không gọi nhầm là recall khi một query có nhiều ảnh đúng. Nên tự chuẩn bị khoảng 15–30 mô tả khác nhau, giữ lại một phần để kiểm tra sau khi cải tiến.

Đã tạo bộ nhãn nội bộ ban đầu trong `evaluation/` (9 dev + 3 holdout); nhãn còn thưa và kết quả thấp, chưa đại diện chất lượng cuộc thi. Chi tiết và latency thật xem `notes/EXPERIMENTS.md`. Lưu report riêng theo split vì CLI ghi đè evaluation_report.json mỗi lần.

## 8. Kiểm thử và giới hạn

```powershell
& "D:\UIT\.venvs\ml-gpu\Scripts\python.exe" -m unittest discover -s tests -v
```

Tests dùng encoder giả lập để kiểm tra số học xếp hạng, bộ lọc, thứ tự frame, cache, OOM, index nhất quán, API và DRES mock; không đo chất lượng CLIP trên Castle5000. Gói không kèm 5.000 ảnh hay model weights. Cần kiểm tra inference CUDA, thời gian build và chất lượng truy vấn trên máy thật. Tài liệu VALIDATION.md ghi kết quả kiểm tra bản đóng gói.

Máy Windows này đã qua **26 tests**, CLIP CUDA thật, full build và Chrome thật ngày 20/11/2026; xem phần local trong VALIDATION.md và notes/PROGRESS.md. Kiểm tra trình duyệt lại với server đang chạy:

```powershell
# Nếu chuyển máy: cài Playwright vào thư mục công cụ riêng, dùng Chrome đã có.
& "D:\UIT\.venvs\ml-gpu\Scripts\python.exe" -m pip install --target .runtime\browser-tools playwright
& "D:\UIT\.venvs\ml-gpu\Scripts\python.exe" scripts\smoke_browser.py --expected-count 4971
```

Smoke browser chặn request ngoài localhost và chỉ nộp mock. Khi checkpoint đã tải, có thể đặt `$env:HF_HUB_OFFLINE="1"` trước start để nạp từ cache; không đặt biến này trước lần tải model đầu tiên. Không cần đặt HF token cho baseline công khai.

## Nguồn tham khảo

- Repo giảng viên: https://github.com/linh222/castle5000_contest (đã xem commit fe88a0164ccc13504997ec83efe06ef1d841d5f2).
- OpenCLIP API và checkpoint: https://github.com/mlfoundations/open_clip
- FastAPI: https://fastapi.tiangolo.com/
- Quy tắc nộp KIS và QA: `DRES_HuongDan_Castle5000_QA.pdf`, đặc biệt mục 7–10, do người dùng cung cấp ngày 22/11/2026.

Phần nâng cấp sau thi thử: caption + BM25/hybrid, so sánh model, query rewriting, reranking một tập ứng viên nhỏ. Bản này ưu tiên hoàn tất luồng sử dụng và đo được baseline.
