# Thực nghiệm

## Baseline dự kiến

- OpenCLIP ViT-B-32 / laion2b_s34b_b79k, ảnh L2 float32, NumPy exact search.
- Máy thực: Windows, Python 3.14.3, torch 2.14.1+cu130, RTX 3050 Laptop 6 GB.
- Chưa đo CLIP retrieval ở thời điểm lập notes. Không sử dụng số đo từ fake encoder làm kết quả chất lượng.
- Tập query sẽ được tạo sau khi xem ảnh gốc, ghi rõ nhãn nội bộ, chưa đầy đủ ảnh liên quan và không phải nhãn BTC. Giữ lại một phần trước cải tiến.

## Kết quả thực — 20/11/2026

Windows, RAM vật lý 12,578,578,432 byte (~11.71 GiB), RTX 3050 Laptop 6 GiB, driver 581.86, torch 2.14.1+cu130, torchvision 0.29.2, OpenCLIP 3.3.0, NumPy 2.5.4, Pillow 13.0.0. Không thay model/checkpoint trong các lần đo.

### Khởi tạo và build

- Smoke CUDA thật: 3 ảnh + 1 query, shape (3,512), L2/finite pass; tải/nạp lần đầu 153.172 s; inference 0.894 s; peak allocated 596.2 MiB. Đây là sanity check, không phải metric retrieval.
- Build đầu tiên: 5000 scanned, 12 hash exclusions, 4988 vector, 0 lỗi, batch 16, CUDA, 218.63 s. GPU memory được quan sát ~787 MiB, không khẳng định đó là peak.
- Rebuild cache trước curate: 4988 vector, 0 pending, 140.23 s; có hoạt động browser/evaluate cùng phiên nên không dùng so tốc độ build cô lập.
- Sau xem trực tiếp thêm 17 bảng màu: 4971 vector, 29 exclusions, 0 lỗi, toàn bộ dùng cache, 44.13 s. Chênh lệch thời gian còn phụ thuộc OS file cache và tác vụ đồng thời.

### Nhãn và chất lượng

`evaluation/dev.jsonl` có 9 query, `holdout.jsonl` có 3 query cố định trước cải tiến. Query được viết từ ảnh gốc đã xem; đa số chỉ có một frame relevant, riêng empty kitchen có 3 frame đã xem. Không phải nhãn BTC và **không đầy đủ ảnh phù hợp**. Cùng camera/bối cảnh; holdout nhỏ không đại diện chất lượng cuộc thi. Không dùng holdout để chọn model; thay đổi duy nhất sau baseline là loại bảng màu nhìn thấy rõ.

| Corpus / split | n | Hit@1 | Hit@5 | Hit@10 | Hit@50 | MRR@200 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline 4988 / dev | 9 | 0 | 0 | 0 | 0.111111 | 0.00652135 |
| Baseline 4988 / holdout | 3 | 0 | 0 | 0 | 0 | 0 |
| Curated 4971 / dev | 9 | 0 | 0 | 0 | 0.111111 | 0.00652135 |
| Curated 4971 / holdout | 3 | 0 | 0 | 0 | 0 | 0 |

CLI thật: `python -m castle evaluate evaluation/dev.jsonl` và tương tự holdout; các lần curated và holdout chạy với `HF_HUB_OFFLINE=1`. Reports `artifacts/baseline_dev.json`, `baseline_holdout.json`, `curated_dev.json`, `curated_holdout.json`.

Phân tích: query chỉnh camera có frame đã gán nhãn 08_269 ở rank 19; empty kitchen có relevant đầu ở rank 165. Bảy query dev khác và ba holdout chưa tìm thấy frame đã gán nhãn trong top 200. CLIP chưa bảo đảm chọn đúng khoảnh khắc. Do nhãn còn thưa, không được suy ra mọi frame xếp trên đều sai; cần xem/gán nhãn nhiều frame phù hợp theo sự kiện trước khi so model. Loại bảng màu làm sạch gallery nhưng **không tăng các metric này**, không tuyên bố nâng chất lượng retrieval.

### Latency HTTP thực, bản cuối 4971

- Model đã nạp CUDA, localhost, k=200, 9 query dev chưa có vector trong cache của server mới; không build/evaluate/browser đồng thời khi đo cuối.
- Wall time p50 **32.164 ms**, p95 **58.2604 ms** (n=9, mẫu nhỏ). Lặp cùng query dùng vector cache: p50 **10.262 ms**. Không gồm nạp model lạnh hay rendering browser.
- Báo cáo `artifacts/final_latency.json`. Báo cáo cũ `baseline_latency.json` có thể chồng thời gian với CLI holdout/cache rebuild; chỉ giữ để truy vết, không dùng như benchmark cô lập.

### Bước tiếp theo

Mở rộng nhãn các query thất bại thành nhiều frame/sự kiện đã xem, bổ sung 15–30 mô tả và tập kiểm tra riêng; đo lại baseline trước query rewriting, caption/hybrid hoặc model khác. Live DRES và collection mapping chưa được kiểm chứng bằng tài khoản thật; người dùng chủ động thao tác trước thi thử.

### Đo độ bao phủ tập ứng viên — 20/11/2026

Chạy CLIP CUDA từ cache trên generation `5abe8458b4f54ed982c399f023fbf72c`, gọi `SearchIndex.search` cho đủ 4971 ảnh với từng query trong `evaluation/dev.jsonl`. Giữ nguyên nhãn gốc, không dùng holdout. Ma trận toàn ảnh là 4971 × 512 float32, 9,709 MiB.

| K | Số query có ít nhất một frame đã gán nhãn trong top K |
|---:|---:|
| 10 | 0/9 |
| 20 | 1/9 |
| 50 | 1/9 |
| 100 | 1/9 |
| 200 | 2/9 |
| 300 | 3/9 |
| 500 | 5/9 |
| 1000 | 6/9 |
| 2000 | 9/9 |

Hạng frame đã gán nhãn đầu tiên theo thứ tự dev là 19, 381, 291, 165, 1954, 718, 1195, 350, 1379. MRR@200 khớp kết quả trước, 0,00652135. Query chỉnh camera có ảnh `08_269.jpg` hạng 19 trên toàn bộ và hạng 3 khi biết để lọc 08:00, còn 343 ảnh. Top 1 toàn bộ là `20_3449.jpg`, người áo xám đưa tay về camera, không có thao tác chỉnh camera trên đầu người đối diện trong frame này.

Ba ảnh đầu của query bếp vắng người (`20_3280`, `20_3315`, `20_3335`) trông phù hợp khi xem, dù không nằm trong nhãn ban đầu. Contact sheet và report lần đo lần lượt ở `artifacts/retrieval_shortlist_examples.jpg` và `artifacts/retrieval_shortlist_dev.json`. Đây là bằng chứng cần gán nhãn thêm nhiều frame hợp lệ và xem ảnh chưa gán nhãn mà cả hai hệ thống trả về trước khi chấm. Chưa sửa nhãn từ top kết quả trong phép đo này.

Bản bạn cùng nhóm tại `ca9f06d` dùng `IndexFlatIP` ở tầng CLIP và xếp lại shortlist bằng RegionCLIP. Giao diện `/search-v2/` gửi `mode=clip_region`, trả 24 ảnh, kiểm tra tối đa 100 ứng viên mặc định. API `legacy` có nhánh `rank_fused` khác, nên cần ghi mode khi so. Chưa có hai ZIP LFS trên máy để đo RegionCLIP. Dùng CLIP-only và RegionCLIP của cùng repo, cùng tập ID và mô tả toàn ảnh để đo riêng hiệu ứng rerank. Xem thêm độ bao phủ của bước CLIP tại mỗi K và đánh giá riêng câu có ràng buộc vật thể, hành động hoặc phủ định.
