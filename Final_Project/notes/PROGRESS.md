# Tiến độ — 20/11/2026 (Asia/Saigon)

## Kế hoạch và tiêu chí hoàn thành

1. **Môi trường/dữ liệu — hoàn thành:** xác minh interpreter, GPU inference cơ bản, dependency; thống kê/đọc ảnh, trùng filename/imageID và metadata; doctor và unit tests thành công. Không đổi torch/driver hiện có.
2. **Baseline thật — hoàn thành:** CLIP CUDA thật, full build, matrix/mapping và rebuild cache toàn corpus đã pass.
3. **Ứng dụng — hoàn thành:** Chrome đã pass lại trên bản cuối, các luồng chính, mock và phục hồi; cả trường hợp app chưa có index.
4. **Đánh giá/bàn giao — hoàn thành:** đo dev/holdout trước/sau, giữ baseline, loại 17 bảng màu đã xem, xác minh index/UI cuối và latency cô lập; tài liệu/notes/lệnh mở lại đã cập nhật.

## Kiểm tra ban đầu

- Project root đúng: `D:\UIT\DS314\Final_Project`; code nằm ngang hàng data. Không có Git, AGENTS.md hay notes trước phiên này.
- Đã đọc yêu cầu đính kèm UTF-8, README_VI.md, VALIDATION.md, toàn bộ Python và scripts; đọc frontend/tests. VALIDATION cũ chỉ là kết quả workspace Linux, không phải máy này.
- PowerShell đếm đệ quy: Batch1 **1853**, Batch2 **1767**, Batch3 **1380**, tổng **5000 JPG**, không có subdirectory trong Batch. Chưa kết luận ảnh đều đọc được.
- `rg --files -g '*.csv' -g '*.json' -g '*.pdf' ...` trong DS314: không thấy metadata hoặc PDF hướng dẫn DRES; chỉ config của project và PDF bài lab. Mapping theo hướng dẫn người dùng, chưa đối chiếu collection thật.
- Interpreter thật: `D:\UIT\.venvs\ml-gpu\Scripts\python.exe`, Python **3.14.3**. Đường dẫn dự phòng không tồn tại.
- torch **2.14.1+cu130**, NumPy **2.5.4**, Pillow **13.0.0**, httpx **0.28.1**. Thiếu torchvision, torchaudio, OpenCLIP, FastAPI, uvicorn.
- `nvidia-smi`: RTX 3050 Laptop **6144 MiB**, driver 581.86. `torch.cuda.is_available()` True; tensor matmul CUDA thực sự thành công. Chưa chạy CLIP.
- `python -m pip check`: No broken requirements found. D còn khoảng 229 GiB trống; C khoảng 23 GiB. Giữ cache/model trên D.

## Phát hiện và điều chỉnh kế hoạch

- requirements hiện yêu cầu Pillow <13 trong khi môi trường có 13.0.0; cần kiểm tra tương thích và tránh hạ package dùng chung nếu không cần.
- Thiếu torchvision: cần xem resolver và phiên bản tương thích trước khi cài; không cho pip thay torch để đáp ứng OpenCLIP.

### Kiểm tra corpus và resolver

- Audit dùng `castle.catalog.scan` + Pillow `Image.load()` trên toàn bộ ảnh: **5000/5000 đọc được**, đều 1280×720; không trùng filename hoặc imageID, không có ID chưa nhận dạng. Báo cáo `artifacts/data_audit.json`; khoảng 89.85 giây.
- Có **12 file** khớp đúng hai SHA-256 bảng màu đã biết (nhiều bản sao với tên khác), không chỉ hai filename mẫu. Dự kiến index **4988** ảnh. Không mở rộng quy tắc loại ảnh.
- Pip dry-run có constraints hoàn tất, chỉ bổ sung package mới; không thay package đã có. Plan lưu `.runtime/dependency_plan.json`. Đang chạy setup bằng interpreter đã xác minh.
- Điều chỉnh `requirements.txt` cho phép Pillow 13 (`<14`) và constraints bảo vệ Pillow hiện có. Cần xác nhận bằng tests/build trên bản này.

### Giai đoạn 1 hoàn thành

- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup.ps1`: thành công. torchvision 0.29.2, OpenCLIP 3.3.0, FastAPI 0.165.0, uvicorn 0.54.0; torch/torchvision/open_clip import OK, CUDA True. torch/NumPy/Pillow giữ nguyên bản trước cài.
- `python -m unittest discover -s tests -v`: **21/21 pass**, 1.844 giây; đây là test logic bằng fake encoder, không phải đo retrieval.
- `python -m pip check`: không có dependency lỗi. Snapshot `.runtime/environment_initial.txt`, `environment_before.txt`, `environment_after.txt`.
- Bổ sung `scripts/smoke_clip.py` kiểm tra ảnh/text thật, shape, finite/L2 và GPU memory trước build. Chuẩn bị Playwright trong `.runtime/browser-tools` riêng (không cài vào venv dùng chung), dùng Chrome sẵn có, không tải browser mới.

### Giai đoạn 2: CLIP thật đã qua kiểm tra nhỏ

- `python -u scripts/smoke_clip.py`: **pass CUDA thật**, 3 ảnh từ 3 Batch, output (3,512), text (512,), finite và L2 đúng. Tải/nạp lần đầu 153.172 giây, mã hóa 3 ảnh + 1 query 0.894 giây, peak allocated 596.2 MiB. Không diễn giải cosine thành chất lượng retrieval.
- Checkpoint lưu `.cache/open_clip` trên D. HuggingFace cảnh báo thiếu symlink trên Windows nhưng tải/cache/inference thành công; không cần quyền admin.
- Đang chạy `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_index.ps1`, batch 16, device auto → CUDA, log `artifacts/build.log`.
- Playwright 1.65.2 được cài riêng vào `.runtime/browser-tools`; sẽ chạy Chrome sẵn có. Script `scripts/smoke_browser.py` chặn request ngoài localhost và chỉ nộp MOCK_ACCEPTED.
- Chrome **158.0.8127.36** headless thực sự khởi chạy được. Thêm hai regression tests cho lỗi khởi tạo encoder và lỗi mã hóa: vẫn browse/image search được và query tiếp có thể thành công. **23/23 tests pass**, 1.381 giây.
- Đã trực tiếp xem ảnh mẫu của 13 giờ và 08_262/266/269; chuẩn bị `evaluation/dev.jsonl` (9 query), `holdout.jsonl` (3 query), nhãn nội bộ từng ảnh đã xem. Query chỉnh camera khớp 08_269; 08_266 thực tế chưa thể hiện thao tác chỉnh nên không dùng nhãn ví dụ trong README cũ.

### Giai đoạn 2 hoàn thành

- Build CUDA batch 16: **218.63 giây**, **4988×512 float32**, 5000 scanned − 12 known calibration = 4988 indexed, **0 errors**, không OOM. GPU memory quan sát khi build ~787 MiB (không phải peak profiler).
- Kiểm tra index: finite, L2 norm, 4988 key/ID duy nhất, mọi source còn tồn tại, mapping filename đúng, không ambiguous/excluded lọt vào index. Batch1 indexed 1841; Batch2 1767; Batch3 1380. Báo cáo `artifacts/index_validation.json`.
- Đối chiếu snapshot pip freeze trước/sau: **không package đã có nào đổi phiên bản**. Torch/driver giữ nguyên.

### Giai đoạn 3 và phát hiện mới

- `python -u scripts/smoke_browser.py`: **pass Chrome thật**. Text search CLIP thật, filters, thumb/full 1280px, neighbors, image-to-image, empty state, lỗi model/network có chủ đích và phục hồi, mock login/select/cancel/confirm/history/refresh/logout, mobile 390px không tràn ngang. Không pageerror, không request browser ngoài localhost; mock gateway không gọi DRES.
- Server `http://127.0.0.1:8000`, start bằng Python venv và `Start-Process -WindowStyle Hidden`; PID launcher trong `.runtime/server_pid.txt`, log server trong artifacts.
- Xem screenshot baseline phát hiện bảng màu chưa khớp hai hash. Contact sheet từ ảnh gốc `artifacts/calibration_review.jpg` xác nhận **17 file bổ sung**: 08_12/17/35/55/90/99/102/104/122/124/149/155/182/195/209/232/250. 08_262/266/269 và các ảnh bếp tiếp theo được giữ. Không loại dựa trên khoảng frame.
- Thêm 17 path cụ thể vào `data/exclude.txt`; bản sao có thể version-control ở `evaluation/calibration_exclusions.txt` vì data/ gitignored. Dự kiến cuối: 5000 − 12 hash − 17 manual = **4971**.
- Baseline 4988 ảnh đã lưu dev/holdout trước sửa. Dev 9 query: Hit@1/5/10 = 0, Hit@50 = 1/9, MRR@200 = 0.00652135; holdout 3 query Hit@K/MRR = 0. Nhãn thưa (đa số chỉ một frame), chưa thể coi mọi kết quả phía trên là sai; cần mở rộng nhãn sự kiện trước so model.
- Bổ sung tiến độ thumbnail/source mỗi 1000 ảnh và flush log vì bản cũ im lặng lâu khi redirect trên PowerShell; không đổi embedding/retrieval.
- Rebuild không đổi dữ liệu **4988 vector**, device `cache`, **140.23 giây**, không cần nạp CLIP/mã hóa lại, 0 lỗi. Sau đó mới bắt đầu build curated 4971 để tránh hai build ghi CURRENT cùng lúc.
- `python -m unittest discover -s tests -v` sau thay đổi: **23/23 pass**, 1.762 giây.
- Build curated hoàn tất: **4971×512**, 29 excluded (12 hash + 17 manual), 0 errors; **4971 cache reused, pending 0**, **44.13 giây**. Generation baseline được giữ nguyên trong artifacts/indexes.
- Phát hiện doctor cũ chỉ in lỗi rồi trả exit 0, setup vẫn báo thành công. Đã sửa CLI doctor trả exit 1 khi thiếu package/import/data hoặc mapping lỗi; PowerShell doctor/setup dừng theo mã lỗi. CPU fallback không bị coi là lỗi. Ba regression tests mới xác minh các trường hợp này: **26/26 tests pass**, 1.521 giây. Doctor thật sau sửa cũng pass, CUDA True, exclusions 29, IDs lỗi/trùng 0.
- Đã khởi động lại server của phiên này để nạp generation 4971; đặt `HF_HUB_OFFLINE=1` cho process server nhằm xác minh nạp model từ cache không cần HF khi search.

## Bàn giao — 17:24 ngày 20/11/2026, Asia/Saigon

- Server **đang chạy** `http://127.0.0.1:8000`, actual server PID **3272** (launcher 20484), CUDA, **4971 ảnh**, chưa đăng nhập DRES. Kiểm tra HTTP cuối ready=true, text search/thumbnail 200, không lỗi server. Lưu PID launcher `.runtime/server_pid.txt`; có thể xem listener bằng `Get-NetTCPConnection -LocalPort 8000 -State Listen`.
- Index cuối **4971×512**, 5000 nguồn, loại 29 bảng màu, 0 ảnh hỏng, 0 ID trùng/chưa biết. Batch1 1824 / Batch2 1767 / Batch3 1380. `artifacts/final_validation.json` đối chiếu scan, paths, exclusions, norm và các ảnh bếp 08_262/266/269 được giữ.
- Chrome final `python scripts\smoke_browser.py --expected-count 4971`: pass; gồm no-index app riêng, desktop/mobile, search thật và mock. Reports/screenshots `artifacts/browser/`. Lỗi model/network là injection có chủ đích; không trình bày như lỗi GPU thật.
- **26/26 unit tests pass**; `compileall -q castle scripts tests` pass; `pip check` sạch. Pip snapshot xác nhận không package ban đầu nào đổi phiên bản.
- Dev/holdout trước và sau curate **không đổi**: dev Hit@50 1/9, MRR@200 0.00652135; holdout 0/3. Không tuyên bố cải thiện retrieval từ dọn bảng màu. Nhãn thưa nên cần mở rộng relevance theo sự kiện. Warm HTTP k=200, 9 query không cache: p50 **32.164 ms**, p95 **58.2604 ms**; query cache p50 **10.262 ms**, không gồm cold load/rendering. Không có build/evaluate/browser đồng thời trong phép đo cuối.
- Đã cập nhật README_VI.md, VALIDATION.md, PROJECT_CONTEXT.md, DECISIONS.md, EXPERIMENTS.md; thêm AGENTS.md cho phiên sau.

### Giới hạn và ưu tiên tiếp theo

1. Chưa login/nộp DRES thật hoặc đối chiếu collection qua tài khoản BTC. Không tự động thực hiện; người dùng mở UI, chọn DRES thật và chủ động kiểm tra evaluation/task/imageID. Không lưu credentials vào notes.
2. Mở rộng nhãn từ 12 query thưa thành nhiều frame phù hợp theo sự kiện; xem top kết quả của các query thất bại, giữ tập kiểm tra riêng. Baseline hoạt động không chứng minh chất lượng đủ thi.
3. Không bảo đảm phát hiện mọi bảng màu khác; chỉ 29 file cụ thể đã xác định. Mọi bổ sung phải có bằng chứng ảnh thật.
4. Chưa thêm model, caption, reranker, fine-tune hoặc đổi GPU stack. Tiếp tục dùng baseline hiện có trước khi thử cải tiến retrieval.

### Mở lại và kiểm tra

Server đang chạy thì mở URL trực tiếp. Sau khi dừng hoặc reboot:

```powershell
cd D:\UIT\DS314\Final_Project
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start.ps1
```

Không chạy thêm server cùng cổng khi server hiện tại còn hoạt động. Có thể dừng process của phiên này bằng `Stop-Process -Id 3272` sau khi xác nhận PID vẫn là listener của Castle; khi khởi động lại PID sẽ đổi.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\doctor.ps1
& D:\UIT\.venvs\ml-gpu\Scripts\python.exe -m unittest discover -s tests -v
& D:\UIT\.venvs\ml-gpu\Scripts\python.exe scripts\smoke_browser.py --expected-count 4971
```

Model/cache đã có trên D. Khi cần ép search chạy từ cache: `$env:HF_HUB_OFFLINE='1'` trước start; không dùng trước lần tải model đầu tiên trên máy khác. Thay dữ liệu/config/exclude thì build lại, rồi khởi động lại server để nạp generation mới.

- Kiểm tra lại khi hướng dẫn thử hệ thống ngày 20/11/2026. `/api/status` báo ready, 4971 ảnh, CUDA. `/api/search` với `a man in a gray sweater adjusting the head camera of a man in black`, lọc 08:00, `k=60` trả `08_269.jpg` ở hạng 3, mã `day1_Kitchen_08_269`.

## Đối chiếu repo bạn cùng nhóm — 20/11/2026

- Clone tại `D:\UIT\DS314\castle5000-image-retrieval`, nhánh `main`, commit `ca9f06d733a3540be7576d2b09df95b959c53daf`. `git status --short --branch` sạch. Mã nguồn có đủ. Đã phục hồi trọng số CLIP 605 MB từ cache local sau khi xác nhận cùng SHA-256 với con trỏ LFS `ac4f8c4b88...` và kiểm tra lại SHA-256 sau checkout. Hai ZIP LFS tổng ~3,2 GB tải rất chậm qua hai lượt thử nên dừng. `py -3.14 tools/start_local.py --prepare` xác nhận thiếu media ZIP, exit 1. Máy có Python 3.14 và 3.11, README của repo khuyến nghị 3.13. Chưa cài dependencies hoặc benchmark toàn app của repo bạn.
- Kiểm tra cú pháp 25 file Python và CLI `tools/start_local.py --help` thành công. Kiểm tra `tools.mock_dres` riêng bằng FastAPI TestClient trên máy, login demo và payload `day1_Kitchen_08_269` trả HTTP 200 / `CORRECT`. Demo trả `CORRECT` mặc định cho ID thông thường. Không dùng verdict demo làm nhãn retrieval.
- Hai bản dùng ViT-B/32 cùng checkpoint LAION. Bản bạn dùng FAISS `IndexFlatIP` để lấy ứng viên chính xác, RegionCLIP-RN50/ONNX CPU để so vùng trong top 100 mặc định và Grounding DINO detections đóng gói để kiểm tra một số điều kiện. Có overlay khung, so rank CLIP, điều kiện chỉnh tay, query planner Gemini tùy chọn và QA tắt mặc định. Điều kiện hành động được trả `unknown` trong code. Builder công khai chỉ build CLIP toàn ảnh từ thư mục phẳng. Launcher yêu cầu 5000 ID khớp manifest vùng. File `backend/app/search.py` chọn MPS/CPU cho text lúc serve, `tools/build_index.py` hỗ trợ CUDA.
- Final_Project dùng NumPy exact cosine, CUDA đã chạy thật, đọc đệ quy ba Batch, loại bảng màu có bằng chứng, cache theo ảnh, báo cáo lỗi và generation nguyên tử. Giao diện có lọc Batch/giờ, frame lân cận và image-to-image trước khi đăng nhập. DRES ưu tiên metadata nếu cấu hình, chặn mã trùng/chưa nhận dạng, xác nhận nộp, kiểm tra evaluation/task và ghi lịch sử; chưa đối chiếu live. Bản bạn có mock HTTP riêng, chọn/dán Run ID, và tạo ID bằng prefix + stem.
- Đề xuất đối chiếu filename/hash để xác định tập giao với 4971 ảnh hợp lệ của Final_Project, mở rộng nhãn nhiều frame theo sự kiện và so ba chế độ trên cùng query: Final_Project CLIP, repo bạn CLIP, repo bạn CLIP + vùng với cụm chi tiết chuẩn bị sẵn. Đo query có relevant trong top ứng viên, Hit@K/MRR, latency riêng cold/warm và thời gian người dùng chọn frame. Tập dev hiện chỉ có 1/9 query thấy ảnh đã gán nhãn trong top 100 của Final_Project. Đây là số liệu nhãn thưa của bản mình, không phải số của partner. Rerank cùng tập ứng viên không thể tìm ảnh nằm ngoài tập đó, nên cần kiểm tra recall trước khi chốt ngưỡng 100.
- Ưu tiên dùng Final_Project làm baseline vận hành. Bổ sung trước trải nghiệm chọn ảnh (phím trái/phải khi xem frame và ghim ứng viên để so sánh). Thử RegionCLIP/khung bằng chứng như chế độ tùy chọn nếu tăng chất lượng có đo. Chia việc: bạn cùng nhóm phụ trách vùng và điều kiện, bên mình phụ trách dữ liệu/ID, workflow nộp và đánh giá chung. Tạm hoãn Gemini/QA để tập trung luồng thi thử. Chưa merge hai codebase.

Để tải tiếp hai gói còn thiếu trong clone, chạy `git lfs pull --include="runtime/*.zip"` khi mạng phù hợp. Khi đã có runtime và venv riêng theo README, chạy `tools/start_local.py --port 8005 --dres-port 8095` để dùng mock và tránh trùng cổng 8000 của Final_Project.

- Làm rõ phạm vi QA. Thư mục DS314 chưa có rubric đồ án. README và [hướng dẫn sinh viên công khai](https://github.com/linh222/castle5000_contest/blob/main/docs/STUDENT_GUIDE.md) mô tả tìm ảnh và nộp mã ảnh qua DRES. Yêu cầu QA của đồ án và định dạng nộp QA vẫn cần xác nhận từ đề bài. Repo bạn cùng nhóm có `/api/qa`, hỗ trợ hỏi ảnh đã chọn. Hỏi trên cả bộ ảnh cần bật thêm `QA_RETRIEVAL_ENABLED`. Cả hai cần cấu hình nhà cung cấp model và chưa được chạy thật trên máy này. Luồng submit của repo đó hiện gửi `mediaItemName` cho bài tìm ảnh. Nếu QA bắt buộc, ưu tiên kiểm thử module chung trên ảnh đã chọn và kiểm tra câu trả lời có dẫn ảnh, rồi đánh giá retrieval QA nếu đề yêu cầu tìm bằng chứng trong cả bộ.

## So sánh kỹ thuật truy vấn — 20/11/2026

- Đọc mã cả hai pipeline, xác minh `/search-v2/` dùng shortlist rerank có điều kiện, API `legacy` có nhánh RRF khác. `IndexFlatIP` và NumPy dot-product trên vector L2 đều là tìm chính xác. Cùng đầu vào vector cho cùng thứ tự về mặt toán học, có thể lệch do tie hoặc sai số. Chất lượng tăng hay giảm do corpus, embedding, query và xếp hạng lại, không suy trực tiếp từ FAISS.
- Chạy CLIP CUDA offline trên 9 query dev và đủ 4971 ảnh. Số query có frame đã gán nhãn tại K=100/200/300/500/1000/2000 lần lượt 1/2/3/5/6/9. MRR@200 khớp trước. Xem trực tiếp montage cho thấy top 3 của query bếp vắng người phù hợp dù nhãn thiếu. Không sửa nhãn hoặc chạm holdout.
- Lưu report và ví dụ trong `artifacts/retrieval_shortlist_dev.json` và `artifacts/retrieval_shortlist_examples.jpg`. Chi tiết tại `notes/EXPERIMENTS.md`. Repo bạn vẫn thiếu hai ZIP LFS, chưa đo latency hoặc chất lượng của RegionCLIP. Baseline đang ready trên CUDA.

## Kiểm tra trước thi thử — 22/11/2026

- Server đã dừng lúc kiểm tra đầu phiên. Khởi động lại bằng `D:\UIT\.venvs\ml-gpu\Scripts\python.exe -m castle serve --port 8000` và để app chạy. `/api/status` ready, 4971 ảnh. Smoke qua HTTP xác nhận CLIP thật dùng CUDA, query chỉnh camera lọc 08:00 đưa `08_269.jpg` lên hạng 3 với `day1_Kitchen_08_269`. Thumbnail/ảnh lớn là JPEG, trả 9 frame lân cận và tìm ảnh tương tự dùng được.
- Luyện local pass login, chọn `local-practice`, chặn nộp khi chưa xác nhận, trả `MOCK_ACCEPTED` khi đã xác nhận, có lịch sử và đăng xuất được. DRES thật chưa được kiểm thử.
- QA chưa sẵn sàng trên laptop này. `Final_Project` chỉ có retrieval. Clone của bạn cùng nhóm có code QA nhưng vẫn thiếu hai ZIP LFS, chưa có venv hoặc `.env`. Ollama localhost trả danh sách model rỗng. Để thử QA thật, ưu tiên module trên máy bạn cùng nhóm nếu đã chạy, hỏi trên ảnh biết đáp án trước rồi thử hỏi không chọn ảnh nếu đề yêu cầu tìm bằng chứng toàn bộ corpus. Yêu cầu và định dạng nộp QA vẫn cần đối chiếu đề bài.

## Cập nhật hướng dẫn DRES KIS và QA — 22/11/2026

- Đọc PDF `DRES_HuongDan_Castle5000_QA.pdf` do người dùng cung cấp, đủ 9 trang. PDF xác nhận cả KIS và QA trong cuộc thi, cùng endpoint nộp bài. KIS gửi mã ảnh và collection, QA chỉ gửi `text`, thường nhận `202 INDETERMINATE` rồi chờ trọng tài. Cập nhật các thông tin trước đó còn bỏ ngỏ về loại task và payload QA.
- Cập nhật `README_VI.md` với cách luyện KIS, phân biệt task bằng Viewer hoặc `taskType`, cách nộp QA thủ công qua Postman, ví dụ hai JSON và bảng đọc phản hồi. Đồng bộ `PROJECT_CONTEXT.md`. Phạm vi là tài liệu, không triển khai sinh hoặc nộp QA trong app, không thử submit DRES thật.

## Tích hợp nộp QA — 22/11/2026

- Thêm **QA · Luyện nộp đáp án chữ** (`local-practice-qa`) và form **Đáp án QA**. Thành viên dùng tìm kiếm ảnh để lấy bằng chứng rồi tự nhập câu trả lời. Luồng nộp ảnh KIS vẫn dùng `local-practice` khi luyện.
- Thêm `/api/dres/submit-text` dùng chung kiểm tra evaluation `ACTIVE`, task hiện tại, xác nhận, chặn trùng và ghi lịch sử với KIS. Payload DRES QA chỉ có `text` đã trim. UI đọc loại task từ `/api/dres/task`, có nút **Cập nhật task** và kiểm tra lại trước khi nộp. Backend đối chiếu mã task lúc xác nhận và lúc gửi để chặn bài cũ. QA 202 hiển thị `INDETERMINATE`, mock hiển thị `MOCK_ACCEPTED`. Timeout giữ `UNKNOWN` và chặn gửi lại y hệt trong task hiện tại.
- `python -m unittest discover -s tests -v` pass **30/30**. TestClient kiểm tra QA mock dù chưa có index, bắt buộc xác nhận, trim, lịch sử, chặn trùng. `httpx.MockTransport` kiểm tra body QA, 202, sai loại, task đổi và timeout. Không gọi DRES thật.
- Khởi động lại server tại `http://127.0.0.1:8000` với `HF_HUB_OFFLINE=1`, PID 22792 lúc bàn giao. `python scripts/smoke_browser.py --expected-count 4971` pass với Chrome thật. Xác nhận CLIP dùng CUDA sau khi tìm, KIS mock và QA mock đều qua thao tác hủy/xác nhận, QA khóa nộp ảnh, lịch sử còn sau tải lại trang. Không page error hoặc request browser ngoài localhost. Báo cáo `artifacts/browser/report.json`.
- Cập nhật `README_VI.md` và `PROJECT_CONTEXT.md` theo luồng mới. Chưa kiểm thử với tài khoản/task DRES thật. QA trong bản này là nộp đáp án do thành viên nhập.

## Mở app trong buổi test — 23/11/2026

- Cổng 8000 ban đầu không có server. Mở lại Castle Local với `HF_HUB_OFFLINE=1`, PID 14596. Index 4.971 ảnh sẵn sàng. Lần tìm đầu vượt timeout 30 giây của phép kiểm tra khi model chưa nạp. Lần tiếp theo hoàn tất trên CUDA, query chỉnh camera lọc 08:00 vẫn đưa `08_269.jpg` vào top 3. Không gửi DRES thật trong kiểm tra này.
- Với mô tả người đàn ông rửa tay ở bồn rửa đảo bếp, thử ba cách diễn đạt qua `/api/search` local và xem ảnh toàn cảnh, crop, frame lân cận. Query `a man washing his hands at the sink in the kitchen island` lọc 09:00 đưa `09_2635.jpg` lên đầu. Đây là ứng viên cần người dùng kiểm tra vì các frame gần đó có rửa dụng cụ. Chưa có nhãn chính thức và không gửi bài.

## Báo cáo hiện trạng — 24/11/2026

- Tạo `notes/BAO_CAO_DU_AN.md` với cây thư mục local, luồng build và truy vấn ảnh, cấu trúc FastAPI và web, cách chạy bằng PowerShell, payload KIS và QA, kết quả kiểm chứng, so sánh hai codebase và phần phát triển tiếp theo.
- Kiểm kê lại 5.000 ảnh nguồn và nạp được ma trận index `4971 × 512` `float32`, chuẩn L2, 4.971 mã ảnh duy nhất. `doctor` thấy CUDA True, `pip check` sạch, **30/30 tests pass**. Bản clone vẫn ở `c1953e5` và thiếu hai ZIP Git LFS. Cổng 8000 không có server khi kiểm kê, lệnh mở lại nằm trong báo cáo.

## Khôi phục web local — 24/11/2026

- Cốc Cốc báo `ERR_CONNECTION_REFUSED` tại `127.0.0.1:8000`. Kiểm tra xác nhận cổng 8000 không có tiến trình lắng nghe. Khởi động Castle Local ở chế độ nền với `HF_HUB_OFFLINE=1`, Uvicorn PID 7832. Trang chủ và trạng thái index trả HTTP 200, 4.971 ảnh. Kiểm tra truy vấn thật qua CUDA trả HTTP 200, có `08_269.jpg` trong top 3 ở bộ lọc 08:00. Để server chạy cho người dùng. Sau khi server dừng hoặc máy khởi động lại, mở bằng `scripts/start.ps1` và giữ terminal mở.

## Kế hoạch thi với gợi ý tăng dần — 24/11/2026

- Lập [kế hoạch thực hiện đồ án](KE_HOACH_THUC_HIEN.md), liên kết từ báo cáo. Các pha có người phụ trách, phụ thuộc, ngưỡng thử local, lựa chọn khi trượt ngưỡng và điều kiện đưa vào bản thi. Mốc 30 giây và số mức 4–5 lấy từ thể thức người dùng quan sát. Ngày thi và quyền dùng dịch vụ ngoài máy chờ BTC xác nhận.
- Đối chiếu PDF Castle5000, mục 1, 3.3, 6.2, 8.4, với schema client DRES v2.0.5. `currentTask` có thông tin nhận diện và thời lượng, schema không có nội dung hint. Đề xuất paste VI trước, OCR vùng đang hiện sau, chỉ thêm adapter API lấy chữ khi có nguồn cho đội thi. Dịch local bằng VinAI và QA bằng Qwen2.5-VL hoặc Gemini là các phương án cần thử. Ollama local có 0 model, hai ZIP LFS của partner vẫn là con trỏ 135 byte.
- Kiểm tra lại bằng `scripts/doctor.ps1`, `python -m unittest discover -s tests -q` và `python scripts/smoke_browser.py --expected-count 4971` trong venv ml-gpu. Doctor đạt, CUDA hoạt động, web có index 4.971 ảnh, 30/30 test đạt. Chrome tìm ảnh thật và chạy KIS/QA mock, không có lỗi trang hoặc yêu cầu mạng ngoài localhost.
- Trên generation `5abe8458b4f54ed982c399f023fbf72c`, gọi `POST /api/search` bốn lần với bốn mức EN tự soạn, `k=200`, bỏ lọc giờ/Batch. Hạng của frame dev `day1_Kitchen_08_269` lần lượt ngoài top 200, 155, 15, 19. RRF trọng số đều, hằng số 60, gộp dần danh sách đã thấy ra vắng, 239, 66, 33. Câu đầu vào và cách tính được ghi trong kế hoạch để tái hiện. Đây là một sự kiện kiểm tra thiết kế, chưa đo OCR, chất lượng dịch hoặc khả năng giải đề thi.
- Bước triển khai kế tiếp là bộ đề KIS/QA phát lại theo đồng hồ local và bộ nhãn theo sự kiện. Sau đó tích hợp ô dán VI, hiển thị bản dịch EN và tìm lại toàn kho ở mỗi mức.

## Đối chiếu phương pháp với GitHub của nhóm — 24/11/2026

- `git rev-parse HEAD` cho bản clone `c1953e5`. `git ls-remote origin refs/heads/main` xác nhận GitHub ở `b317148`, đã thêm 23 commit. Đọc README, `backend/app/search.py`, `backend/app/metadata_search.py` và nhật ký QA/KIS qua `git show origin/main:<đường dẫn>`. Bản clone đang dùng và index Castle Local vẫn ở phiên bản đã kiểm tra.
- Cập nhật pha 3 trong [kế hoạch](KE_HOACH_THUC_HIEN.md) để thử SigLIP 2 có/không có điểm BM25 từ metadata trên cùng câu hỏi lũy tiến. Kết quả R@5 0,10 lên 0,41 là số do repo partner báo trên tập riêng. Lượt đối chiếu này chỉ đọc mã nguồn và tài liệu.

## Thiết kế tự nộp KIS và QA — 25/11/2026

- Đọc hai route nộp trong `castle/app.py` và luồng kiểm tra task trong `castle/dres.py`. Cả KIS và QA hiện yêu cầu `confirmed` từ giao diện. Gateway đã có kiểm tra task, loại bài và chặn gửi trùng đáp án.
- Cập nhật [kế hoạch](KE_HOACH_THUC_HIEN.md) với chế độ bật một lần cho evaluation, quyết định tự nộp riêng cho KIS và QA, kiểm tra phiên bản gợi ý/task, test mock và phép so với nút xác nhận. Bước này cập nhật thiết kế. Code tự chọn và tự gửi còn phải triển khai.

## Bản thử tự nộp trên Castle Local — 25/11/2026

- Thêm nút **Bật tự nộp** theo từng evaluation. Người dùng dán các gợi ý đã hiện và bấm **Xử lý gợi ý**. App tạo truy vấn EN, tìm trên index CLIP và cho VLM đối chiếu tối đa tám ảnh đánh số. KIS kiểm tra lại ở tập ảnh nhỏ khi cần, sau đó đối chiếu đề với mô tả độc lập của riêng frame. QA trả lời ngắn từ một ảnh đã tìm. Khi có quyết định, giao diện gọi gửi tự động. Bản nộp tay và lịch sử cùng dùng gateway.
- Backend giữ quyết định một lần trong phiên, kiểm tra task và trạng thái tự nộp trước khi gửi. Đã kiểm tra chống gửi lại, nộp tay đồng thời, gợi ý mới, task đổi, tắt chế độ, QA chờ chấm và timeout bằng TestClient và HTTP giả lập. **35/35 test đạt**, `compileall` đạt, `pip check` sạch. Chrome smoke tìm kiếm thật, gửi KIS/QA bằng nút và bật/tắt tự nộp đạt.
- Cài `qwen3-vl:2b` qua Ollama trên ổ D. Tag này dùng renderer thinking. Dùng assistant prefill để nhận JSON trực tiếp và nâng ngữ cảnh lên 12.288 vì một lượt tám ảnh thực tế vượt 4.096 token. Gemini là tùy chọn cấu hình. Lượt thử xác thực khóa Gemini có sẵn trả 401 nên kiểm chứng thực tế dùng Ollama.
- Chrome với CLIP CUDA, Ollama và ảnh thật tự gửi QA **trắng** cho câu về chụp đèn trên đảo bếp, ảnh bằng chứng `15_1459.jpg` và verdict `MOCK_ACCEPTED`. Lần chạy cuối mất 9,49 giây.
- KIS đã qua kiểm thử chọn và gửi tự động với phản hồi model giả lập. Trên ảnh thật, bản đầu đã chọn nhầm ảnh cho câu về người đưa tay về camera và nhầm động tác cầm thớt. Bản hiện tại giữ kết quả ở trạng thái thiếu bằng chứng với các ca kiểm tra đó. Lần phát lại Chrome cuối dừng ở cả ba câu KIS tự soạn, từ 15 đến 27 giây mỗi câu. Chưa có đủ bằng chứng về độ chính xác tự nộp KIS qua VLM trên kho thật. Các case mock được tính là kiểm thử đường gửi, không dùng để đánh giá retrieval.
- Mở app bản mới tại `http://127.0.0.1:8000`, dùng index hiện có 4.971 ảnh. Mọi lượt gửi trong phiên phát triển dùng mock hoặc HTTP giả lập. [Cách thử](../README_VI.md#tự-nộp-kis-và-qa) và [kế hoạch](KE_HOACH_THUC_HIEN.md#tự-nộp-kis-và-qa) đã cập nhật.

## Chọn hướng tích hợp hai bản — 25/11/2026

- `git ls-remote origin refs/heads/main` xác nhận partner vẫn ở `b317148`. Đối chiếu README, nhật ký và `tools/package_release.py` tại commit này. Gói SigLIP 2 chứa `image-metadata/metadata.jsonl` cùng các manifest. Bản clone local vẫn ở `c1953e5`, hai ZIP đang có chỉ là con trỏ LFS. Chưa có kết quả chạy đối chứng của bản mới trên laptop.
- Đề xuất dùng Castle Local làm nền và thử thêm metadata + BM25 của partner trên cùng 4.971 ID, giữ CLIP làm mốc. Đo ảnh đúng trong top 5/20 ở từng gợi ý và thời gian tới kết quả. Kiểm tra `/api/status` cho thấy app đang sẵn sàng, CUDA, 4.971 ảnh và Ollama. Model tự nộp KIS mới kiểm chứng đường gửi bằng giả lập. KIS khi làm bài thật ưu tiên người chọn ảnh đến khi có số đo đủ tin cậy. QA local đã gửi mock từ ảnh thật một câu. So QA nhiều frame của partner trên cùng bộ câu và theo thời gian trước khi chọn làm mặc định.

## Đề luyện local cho KIS và QA — 25/11/2026

- Xác nhận cổng 8000 sẵn sàng, CUDA, 4.971 ảnh và Ollama có `qwen3-vl:2b`. Qua `POST /api/search`, tìm KIS chỉnh camera với bốn mức EN trong top 60. Ảnh mục tiêu `08_269.jpg` lần lượt nằm ngoài top 60, ngoài top 60, hạng 15 và hạng 3. Mức cuối dùng lọc 08:00. Mở ảnh trả HTTP 200 và gửi đúng mã `day1_Kitchen_08_269` trong mock, nhận `MOCK_ACCEPTED`.
- QA tự nộp với câu “Các chụp đèn treo phía trên đảo bếp có màu gì?” qua `/api/dres/auto/prepare` và `/api/dres/auto/commit`. Trả lời **trắng**, ảnh bằng chứng `day1_Kitchen_15_1459`, `MOCK_ACCEPTED`, 15,27 giây. Hai bài dùng evaluation luyện local và đề mô phỏng.

## Tách chất lượng tìm ảnh và kiểm chứng KIS — 25/11/2026

- Kiểm tra ảnh gốc và thấy cả `08_269` lẫn `08_272` đều khớp câu chỉnh camera trước đó. Tạo đề tìm đúng một frame với ground truth `day1_Kitchen_20_3449` chọn trước khi viết bốn mức. Xem trực tiếp 42 ảnh khác nhau trong hợp frame lân cận, ảnh gần nhất theo embedding và top 20 của gợi ý cuối. Chỉ frame đích thỏa trọn bố cục trong nhóm đã xem. Ghi VI/EN và ảnh đối chiếu trong [evaluation/README.md](../evaluation/README.md#đề-kis-tìm-đúng-một-frame).
- Gọi `POST /api/search` với `k=200`, toàn kho 4.971 ảnh. Bốn mức hạng ngoài top 200, 88, 1, 1. Tái lập đúng thông báo trong ảnh người dùng với gợi ý “a man in a gray sweater next to a man in black in a kitchen”. Ở mức cuối của đề mới, `/api/dres/auto/prepare` vẫn trả ảnh đích hạng 1 nhưng `preview=null` sau 13,24 giây. Lượt này dùng mock để kiểm chứng, không có lần gửi.
- Đối chiếu GitHub partner `b317148`. Ưu tiên giữ gợi ý gốc và kiểm chứng từng chi tiết của top ứng viên trong Castle Local. Thử BM25 metadata hoặc tầng thị giác của partner trên cùng nhãn để đo riêng ảnh đích có được tìm thấy và được chọn để nộp hay không.

## Đề KIS laptop có mã QR — 25/11/2026

- Chọn ảnh gốc `Batch1/08_353.jpg` làm ground truth trước khi viết mô tả. Xem 24 ảnh khác nhau trong nhóm lân cận và ảnh gần nhất theo embedding, cùng top 20 của bản mô tả chi tiết ban đầu. `08_358.jpg` và `13_341.jpg` có laptop với bố cục khác. [Bốn mức truy vấn](../evaluation/README.md#đề-kis-laptop-có-mã-qr)
- Gọi `POST /api/search` trên cả 4.971 ảnh với `k=200`. Ảnh đích lần lượt ngoài top 200, hạng 30, hạng 1 và hạng 1. Câu VI cuối qua `/api/dres/auto/prepare` cũng đưa ảnh đích lên hạng 1 và trả `preview=null` sau 12,52 giây. LLM viết lại “áo ca rô đỏ” thành “red robe”. Lượt thử dùng phiên mock, dừng sau bước chuẩn bị.

## Kiểm tra lại quyết định tự nộp ảnh laptop — 25/11/2026

- Tái lập câu EN đang hiện trên ảnh giao diện qua `AutoAnswer.search_query`, `POST /api/search` và `AutoAnswer.verify`. `08_353.jpg` đứng đầu (0,371), ảnh thứ hai 0,270. Qwen chọn đúng ảnh 1 khi xem tám ứng viên. Ở bước mô tả độc lập một ảnh, model ghi nhận laptop bạc và mã QR xanh nhưng bỏ vị trí sát ống kính và nền xanh đậm của màn hình. Kết quả so khớp là `matches=false`, do đó quyết định tự nộp rỗng. Lượt này chỉ đọc kết quả model.

## Kiểm tra tùy chọn Gemini — 25/11/2026

- `GET http://127.0.0.1:8000/api/status` xác nhận app đang dùng `ollama`. Kiểm tra danh mục model Gemini qua `GET /v1beta/models/{model}` với header API key từ môi trường. Cả `gemini-2.5-flash-lite` và `gemini-3.5-flash-lite` trả HTTP 401. Lượt kiểm tra chỉ gọi metadata model. Castle Local đã đọc được `CASTLE_AUTO_PROVIDER` và `CASTLE_GEMINI_MODEL` ở lúc mở server, hỗ trợ chạy thêm bản thử bằng `scripts/start.ps1 -Port 8001` sau khi có khóa xác thực được.

## Cập nhật clone, trích metadata và luồng nhập từ TV — 25/11/2026

- Fast-forward `D:\UIT\DS314\castle5000-image-retrieval` từ `c1953e5` lên `b317148`, sạch và bằng `origin/main`. Nhánh từ xa `improve-qa-and-search-verification` đã có trong lịch sử main. Giữ các gói lớn ở dạng LFS.
- Đọc từng khoảng byte của ZIP SigLIP 2 ghim theo commit. Trích và kiểm tra CRC ba tệp metadata với khoảng 657 KB tải xuống từ ZIP 729 MB. Lưu tại `.runtime/partner_metadata/b317148`. Có 5.000 dòng hợp lệ, 5.000 tên JPG trùng dữ liệu gốc và 4.971/4.971 ảnh trong chỉ mục có metadata. Metadata `08_353` nhắc laptop có QR ở tiền cảnh và người áo ca rô cạnh bếp. `08_358` và `13_341` cũng có laptop và QR.
- Chạy `MetadataStore` và `MetadataMatcher.bonus` từ clone, ghép theo `filename`, cộng điểm BM25 vào cosine CLIP với trọng số partner `0,5`. Lượt laptop ở mức 30 giây lên hạng 1 từ 30. Lượt sát camera lên hạng 4 từ 88. Cảnh chỉnh camera có hai frame hợp lệ, hạng tốt nhất từ 15 lên 2. Tổng hit top 5 từ 4/9 lên 7/9. Đây là 9 lượt gợi ý từ ba ca nội bộ trên CUDA. Bản web hiện tại chưa nạp metadata. [Báo cáo thử nghiệm](../artifacts/partner_metadata_probe_b317148.json)
- Trường `frame_type` còn đánh dấu 21 `test_card` trong 4.971 ảnh chỉ mục. Xem contact sheet và xác nhận cả 21 là bảng màu. Khởi động lại web cổng 8000 sau khi thấy cổng trống. Truy vấn “a silver laptop with a QR code held close to the camera” trả `08_353` hạng 1, hai bảng màu `20_3453` và `20_3584` hạng 2 và 3. Đây là các ứng viên có thể lọc ở tầng tìm kiếm. [Ảnh kiểm chứng](../artifacts/partner_metadata_test_cards.jpg)
- Theo thông tin đề chỉ chiếu trên TV, sửa [kế hoạch](KE_HOACH_THUC_HIEN.md) sang đọc và nhập từng gợi ý VI, giữ câu hỏi QA đầy đủ và đo thời gian từ TV đến lưới. LLM dịch/tách chi tiết và kiểm tra ảnh ứng viên hoặc trả lời QA. CLIP + BM25 tìm local, mã gửi qua gateway DRES. Ưu tiên thử Gemini 3.5 Flash-Lite với shortlist khi có khóa hợp lệ và được phép gửi dữ liệu. Kiểm tra hiện tại với khóa trong môi trường vẫn trả 401. OpenAI GPT-5.4 mini là đối chứng cần adapter.

## Tích hợp metadata vào web Castle Local — 25/11/2026

- Giữ app Castle Local làm giao diện và gateway. Thêm `castle/partner_metadata.py` để tải trực tiếp `MetadataStore` và `MetadataMatcher` từ clone `b317148` theo cấu hình. Đọc metadata theo tên file. Lượt khởi tạo thực nạp 4.971/4.971 bản ghi trong khoảng 3,5 giây. Nếu repo hoặc metadata thiếu, tìm ảnh rơi về CLIP và web báo trạng thái.
- Web có **Phương pháp tìm ảnh** với **CLIP** và **CLIP + BM25**. Dùng lựa chọn đó cho tìm văn bản và lấy ứng viên ở bước **Xử lý gợi ý**. API thiếu tham số `mode` dùng CLIP, giúp chạy lại đối chứng cũ. Với thử nghiệm A/B, hai mode dùng cùng index 4.971 ảnh và hệ số metadata 0,5.
- `python -m unittest discover -s tests -q` qua 37/37 test. `compileall -q castle tests` đạt. Chrome đổi CLIP + BM25 → CLIP → CLIP + BM25 trên truy vấn laptop và nhận hạng ảnh đích 1 → 30 → 1. Mã `day1_Kitchen_08_353` gửi mock trả `MOCK_ACCEPTED`. Smoke Chrome tổng thể đạt ở cổng 8000, gồm lọc Batch/giờ, ảnh tương tự, KIS/QA mock và giao diện mobile.
- Thử `/api/dres/auto/prepare` với `mode=hybrid` và phiên mock. Lượt đầu trả 503. Lượt chạy lại trả HTTP 200, dùng hybrid, `08_353.jpg` hạng 1. Qwen trả `preview=null` sau 37,51 giây nên không có bài trong lịch sử. Unit test cũng xác nhận ảnh dùng làm bằng chứng lấy từ kết quả hybrid khi model giả lập chọn ảnh.
- Mở bản tích hợp ở `http://127.0.0.1:8000`. [Hướng dẫn thử trên web](../README_VI.md#so-clip-và-clip--bm25-trên-web) và [phương án ghép hai bản](KE_HOACH_THUC_HIEN.md#phương-pháp-tham-khảo-từ-github-của-nhóm) đã cập nhật.

## Chuẩn bị thử Gemini 4 Hydrogen — 27/11/2026

- Xác minh tài liệu Google cho ID `gemini-4-hydrogen`, ảnh đầu vào, JSON có schema và REST `generateContent`. Adapter chọn `thinkingLevel=LOW` riêng cho Hydrogen, bỏ `temperature` và giới hạn 512 token để tránh cắt phần trả lời do token suy luận được tính vào cùng ngân sách. Các giai đoạn gọi model là chuẩn hóa truy vấn trước tìm kiếm và đọc ảnh ứng viên sau tìm kiếm.
- Chạy `python -m unittest discover -s tests -q` đạt 38/38, gồm HTTP giả lập cho request văn bản và ảnh qua adapter Hydrogen. `python -m compileall -q castle tests` đạt. Kiểm tra danh mục Gemini và metadata model Hydrogen bằng khóa đang có đều trả 401. Suy luận thật chờ khóa xác thực được.
- Mở lại bản local cổng 8000. `/api/status` báo sẵn sàng, Ollama và metadata 4.971 ảnh. Thêm [hướng dẫn chạy Hydrogen riêng ở cổng 8001](../README_VI.md#thử-gemini-4-hydrogen-trên-cổng-8001).

## Hydrogen High và kiểm tra kết nối web local — 27/11/2026

- Đổi request `gemini-4-hydrogen` sang `thinkingLevel=HIGH`, cập nhật hướng dẫn và assertion của test HTTP giả lập. `python -m unittest discover -s tests -q` đạt 38/38, `compileall -q castle tests` đạt. `GET /v1beta/models/gemini-4-hydrogen` với khóa hiện có trả 401. Cổng 8001 chưa có tiến trình lắng nghe.
- Kiểm tra ảnh báo `ERR_CONNECTION_REFUSED`. Tại thời điểm kiểm tra, Uvicorn PID 23160 đã lắng nghe `127.0.0.1:8000` và log startup hoàn tất. Chrome mới mở trang trả HTTP 200, tải lưới ảnh và tìm laptop bằng CLIP + BM25 trả HTTP 200 trên CUDA. `08_353.jpg` đứng đầu, metadata 4.971 ảnh, không có lỗi JavaScript. Không tái lập lỗi kết nối ở thời điểm kiểm tra.

## Xác minh server Gemini do người dùng mở — 27/11/2026

- `GET /` và `GET /api/status` ở cổng 8001 đều trả 200. API báo sẵn sàng, provider `gemini`, 4.971 ảnh chỉ mục và 4.971 metadata. Cổng 8000 cũng trả 200 với Ollama.
- PID 23248 khởi động trước mốc cập nhật cuối của `castle/auto_answer.py`. Để chắc chắn tiến trình dùng `thinkingLevel=HIGH`, chạy lại lệnh start trong chính cửa sổ PowerShell đã khai báo biến môi trường. Ở bước này chỉ kiểm tra startup và trạng thái provider. Khóa trong server 8001 sẽ được kiểm tra khi người dùng gửi truy vấn Gemini.

## Thử QA tiếng Anh trên Gemini — 27/11/2026

- Dùng phiên mock mới ở cổng 8001, provider Gemini, metadata 4.971 ảnh, hỏi “What color are the pendant lights above the kitchen island?” qua `/api/dres/auto/prepare` với `mode=hybrid`. Lượt đầu báo upstream 503 sau 6,97 giây. Lượt thử lại trả lỗi model không phản hồi sau 151,98 giây tính từ đầu yêu cầu. Cổng 8001 báo CUDA sau lượt thứ hai. Hai lượt không có đáp án để commit.
- Đối chứng qua `POST /api/search` ở cổng 8000 với “pendant lights above the kitchen island” và hybrid. Ảnh đầu là `Batch2/16_622.jpg`, 51 ms theo phản hồi server. Xem ảnh xác nhận chụp đèn trắng. Dùng `white` làm đáp án mẫu tiếng Anh cho ca này.

## Chuẩn bị bàn giao qua nhánh `ph` · 28/11/2026

- Baseline local qua 38/38 unit test. Nhánh `ph` được tạo từ `origin/main` tại `b317148` trong worktree `D:\UIT\DS314\castle5000-ph`. Đặt Castle Local dưới `Final_Project/` và dùng mã BM25 từ thư mục cha của cùng checkout. Cấu hình trên nhánh đọc metadata từ `runtime-data/image-metadata`.
- Bản ở worktree qua 38/38 unit test và `compileall -q castle tests scripts`. Lượt kiểm tra API ban đầu thiếu cache OpenCLIP trong worktree. Chạy lại với cache, index và ảnh sẵn có ở bản local, đồng thời vẫn dùng phần tải BM25 và đường dẫn metadata của nhánh mới. Web trả HTTP 200, nạp 4.971 metadata, tìm trên CUDA và đưa `08_353.jpg` lên hạng 30 với CLIP, hạng 1 với CLIP + BM25. Các bài nộp trong unit test dùng mock hoặc HTTP giả lập.
- Commit bàn giao gồm source, cấu hình, hướng dẫn, test và nhãn đánh giá đã dùng để tái hiện. Ảnh, cache, index, metadata runtime, nhật ký nộp và thông tin xác thực được giữ ngoài commit.

## Lệnh tiếp tục

```powershell
cd D:\UIT\DS314\Final_Project
& D:\UIT\.venvs\ml-gpu\Scripts\python.exe -m pip check
.\scripts\doctor.ps1
& D:\UIT\.venvs\ml-gpu\Scripts\python.exe -m unittest discover -s tests -v
```
