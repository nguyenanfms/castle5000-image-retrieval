# Kế hoạch thực hiện đồ án DS314 Castle5000

**Bản lập ngày 24/11/2026, cập nhật ngày 25/11/2026.** Chọn Castle Local làm ứng dụng thi. Mỗi gợi ý mới giúp tìm thêm ứng viên và giữ các ảnh đang cân nhắc. KIS gửi một mã ảnh. QA gửi một đáp án ngắn có ảnh làm bằng chứng trong app. Bản hiện tại gửi sau khi thành viên xác nhận. Bản phát triển thêm chế độ **Tự nộp** để app chọn và gửi đáp án khi người dùng đã bật chế độ cho buổi thi.

**Thứ tự đầu tư cho hai người.** Đề thi chiếu trên TV theo thông tin của người dùng. Ưu tiên luồng *gõ gợi ý tiếng Việt → dịch → tìm ảnh → bổ sung gợi ý*. Dùng metadata của partner để thử xếp hạng lại, rồi đo chất lượng kiểm chứng ảnh và QA. Trước ngày thi một ngày, khóa cấu hình đã thắng trong diễn tập. Ngày thi cụ thể và việc dùng dịch vụ AI ngoài máy cần đối chiếu với BTC.

## 1. Điểm xuất phát trên laptop

| Thành phần | Đã kiểm tra | Hệ quả cho thiết kế |
|---|---|---|
| Castle Local | Web cổng 8000, CUDA, 4.971 ảnh có vector 512 chiều từ 5.000 ảnh gốc. Bản tự nộp qua 35 test. Chrome thực đã tìm ảnh và luyện nộp KIS, QA qua mock. | Dùng chung web, chỉ mục và luồng nộp hiện có. Mỗi thay đổi đều phải chạy lại được đường này. |
| Tìm kiếm | OpenCLIP ViT-B-32, cosine chính xác bằng NumPy. p95 HTTP đã đo khoảng 58 ms trên 9 truy vấn khi model đã nạp. Có lần tìm đầu vượt 30 giây khi model còn lạnh. | Làm nóng bằng một truy vấn thật trước thi. Đo riêng thời gian đọc đề, dịch, tìm và vẽ ảnh lên trình duyệt. |
| Chất lượng | Nhãn cũ rất thưa. Dev chỉ tìm lại được frame đã gán nhãn trong top 50 ở 1/9 câu, top 200 ở 2/9 câu. | Gán nhiều frame hợp lệ cho mỗi sự kiện và chấm từng mức gợi ý trước khi chọn thuật toán. |
| QA và repo partner | Castle Local có QA bằng Ollama và ô nộp tay. Bản clone partner ở `b317148`, có SigLIP 2, BM25 và Gemini QA. Đã trích metadata LFS, ghép được đủ 4.971 ảnh chỉ mục local. | Tách thử nghiệm metadata và CLIP local khỏi phép đo toàn app partner. |
| Tài nguyên | RTX 3050 Laptop 6 GiB, RAM vật lý khoảng 11,7 GiB. Ollama cổng 11434 có Qwen3-VL 2B. | Đo từng model với CLIP đang phục vụ. Đặt cache/trọng số trên ổ D và tách môi trường dịch/QA nếu thư viện xung đột với venv GPU. |

Nguồn hiện trạng và số đo gốc nằm trong [báo cáo dự án](BAO_CAO_DU_AN.md), [thực nghiệm](EXPERIMENTS.md) và [báo cáo Chrome local](../artifacts/browser/report.json).

## 2. Nhập đề từ TV và chuyển sang tiếng Anh

**Luồng thi chính là gõ gợi ý tiếng Việt từ TV.** Người A nhập các chi tiết phân biệt ngay khi xuất hiện, người B xem ứng viên và chuẩn bị nộp. Với KIS, có thể gõ ngắn như “laptop bạc, sát camera”, rồi thêm “QR sáng, màn hình xanh đậm”. Với QA, ghi nguyên câu hỏi hoặc giữ đủ từ để xác định câu hỏi đang hỏi **màu, số lượng hay hành động**. Nút **Cập nhật task** đọc tên và loại task từ DRES. Câu chữ dùng cho truy vấn đến từ người nhập. [Đặc tả client DRES v2.0.5](https://github.com/dres-dev/DRES/blob/v2.0.5/doc/oas-client.json) · [Tích hợp DRES hiện tại](../castle/dres.py)

**Giao diện hiện tại** nhận nhiều gợi ý trong ô **Đề và các gợi ý đã hiện**, xử lý khi bấm nút. **Giao diện đề xuất** dùng ô một dòng để thêm một mức bằng Enter, cho sửa từng mức, hiện bản dịch cạnh gợi ý gốc và giữ danh sách ảnh đang cân nhắc. Tập phản xạ bằng một màn hình phát đề giả riêng với laptop tìm ảnh. Đo cả thời gian đọc và gõ từ lúc câu xuất hiện trên TV.

## 3. Một task với bốn hoặc năm mức gợi ý

| Thời điểm diễn tập | Castle Local cần làm | Người thi cần thấy |
|---|---|---|
| 0 giây, mức 1 | Dịch mô tả ngắn, tìm trên cả 4.971 ảnh và trả ngay lưới đầu. | Câu gốc, bản tiếng Anh có thể sửa, 24 ảnh đầu, nút ghim. |
| 30 giây, mức 2 | Nhận chi tiết mới. Tìm lại toàn kho, cập nhật ứng viên, giữ các ảnh đã ghim. | Gợi ý nào vừa đổi, ứng viên nào mới xuất hiện, ảnh đang mở vẫn ở nguyên vị trí. |
| 60 giây, mức 3 | Tìm theo hành động, vật hoặc quan hệ phân biệt. Nếu đã qua kiểm thử, chạy xếp hạng theo vùng ở nền. | Có thể mở ảnh lớn, xem tay/vật và các frame lân cận trong lúc chờ. |
| 90 giây, mức 4 | Đối chiếu chi tiết với các ảnh đang cân nhắc. QA có thể đọc ảnh ghim hoặc nhóm ảnh truy xuất được. | KIS có mã ảnh. QA có đáp án nháp, ảnh nguồn và vùng cần kiểm tra. |
| 120 giây, mức 5 nếu có | Chạy cùng quy trình khi có câu mới. Đội có thể nộp từ bất kỳ mức nào khi đã đủ bằng chứng. | Theo dõi đồng hồ do BTC chiếu. Mức cuối và thời điểm hết giờ là hai mốc riêng. |

Một task giữ `evaluation`, `task_ref`, lượt chạy của task, loại KIS/QA, danh sách gợi ý VI/EN, phiên bản nội dung, nguồn và thời điểm nhận. Một câu có thể **bổ sung**, **hiển thị lại cả đoạn cũ kèm chi tiết mới**, hoặc **sửa thông tin cũ**. Dùng ba thao tác thêm, sửa, bỏ. Bóc phần lặp ở đoạn cộng dồn. Khi sửa, vô hiệu hóa bản dịch và kết quả bị thay thế. Khi gặp câu mâu thuẫn chưa rõ, cho người thi xem hai bản và chọn. Khi tiếp tục giữa chừng, bắt đầu ngay từ chữ đang hiện.

Mỗi tác vụ dịch, tìm hay QA mang cặp `(lượt task, phiên bản gợi ý)`. Khi gợi ý đổi, kết quả chậm của phiên bản cũ không ghi đè lưới hoặc ô đáp án. Khi BTC chuyển task, kết thúc phiên cũ, hủy việc đang chờ, bỏ ghim/đáp án đang sửa, khóa nộp tới khi đồng bộ lại. Có nút bắt đầu lượt mới cho trường hợp BTC chạy lại cùng tên task. Cả nộp theo nút hiện tại lẫn chế độ tự nộp dự kiến đều đi qua kiểm tra task của backend. [Luồng xác nhận hiện tại](../web/app.js)

## 4. Thiết kế dịch, tìm ảnh và trả lời QA

### Dịch và tạo câu tìm kiếm

- Giữ nguyên câu tiếng Việt trên màn hình. Mỗi lần có gợi ý, tạo một bản dịch sát nghĩa và một mô tả thị giác ngắn cho CLIP từ những chi tiết đã xuất hiện. Giữ số lượng, màu, trái/phải, đồ vật, phủ định và ai làm gì với ai. Với đoạn dài, chia theo mệnh đề để tránh bị tokenizer CLIP cắt mất chi tiết cuối. Lưu bản dịch đã duyệt theo nội dung, ngôn ngữ và phiên bản model.
- Ví dụ KIS: “Một người đàn ông đang rửa tay ở bồn rửa trên đảo bếp” → `a man washing his hands at the sink in the kitchen island`. Ví dụ QA tự soạn: “Người đang rửa tay mặc áo màu gì?” → câu tìm `a person washing their hands at the kitchen island sink`, câu hỏi cho model đọc ảnh vẫn là nguyên văn. Chỉ điền màu khi đã quan sát được áo.
- Mẫu thử offline đầu tiên là **VinAI Translate VI→EN**. Chạy thử bằng CPU ở tiến trình riêng, chỉ chuyển sang GPU nếu độ trễ và VRAM đo được cho phép. Có thể thử **Qwen2.5 3B** qua Ollama để rút gọn câu hoặc tách thuộc tính, so với luật đơn giản và bản dịch của người. Khi dịch lỗi hoặc quá hạn, cho sửa EN trực tiếp. [Model dịch của VinAI](https://huggingface.co/vinai/vinai-translate-vi2en-v2) · [Qwen2.5 3B](https://ollama.com/library/qwen2.5:3b)

### Hai kỹ thuật truy vấn trên bản clone

1. **CLIP toàn ảnh.** Với mỗi mức, mã hóa mô tả EN đã cập nhật thành vector đơn vị. Tính `score = image_vector · text_vector` trên toàn kho, sắp giảm dần. Trả danh sách mới nhất ngay. Chỉ dùng bộ lọc giờ hoặc Batch khi có bằng chứng từ đề hoặc do người thi chủ động chọn. Corpus nhỏ nên có thể cho một ảnh quay lại ở mức 3 dù vắng ở top 200 mức 1. [Mã tìm kiếm](../castle/index.py)
2. **CLIP toàn ảnh rồi xếp lại theo vùng.** Lấy ứng viên trên toàn kho như bước 1. Thử RegionCLIP của repo bạn trên top 100, 200, 500 để so các vùng theo cụm từ như “tay bên bồn”, “áo xám” hoặc “bát đỏ”. Ảnh đúng phải lọt vào tập ứng viên trước khi rerank. Dùng cùng bộ ID, cùng câu EN, cùng checkpoint CLIP. Chạy CLIP-only và CLIP+region trong cùng bản partner rồi mới so giao diện hai app. Với top 500, đo bên trong backend hoặc bằng công cụ đánh giá vì API web Castle Local hiện giới hạn `k ≤ 200`. [Tìm kiếm của repo bạn](../../castle5000-image-retrieval/backend/app/search.py) · [RegionCLIP](../../castle5000-image-retrieval/backend/app/regionclip.py)

So thêm **cách kết hợp các mức** độc lập với việc có dùng RegionCLIP. Đối chứng là chỉ dùng mô tả mới nhất. Biến thể A dùng một mô tả ngắn tổng hợp tất cả dữ kiện còn hiệu lực. Biến thể B hợp nhất các bảng hạng bằng Reciprocal Rank Fusion (RRF): mỗi ảnh được cộng `w_i / (60 + hạng_i)` cho danh sách có ảnh, hạng bắt đầu từ 1. Thử trọng số đều trước, sau đó thử tăng trọng số cho mô tả mới nhất trên dev. Loại bản lặp trước khi cộng. Giao diện giữ nút xem danh sách mới nhất và nhóm ảnh ghim. [Bài báo RRF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)

**Phép thử đã chạy ngày 24/11.** Một sự kiện dev có frame nội bộ `day1_Kitchen_08_269`. Bốn mô tả EN dưới đây được nhóm mô phỏng từ nhãn, tìm qua web local, không lọc giờ hay Batch. RRF dùng top 200 mỗi mức, trọng số đều, hòa điểm xếp theo mã ảnh.

| Mức | Mô tả đưa vào CLIP | Hạng với mô tả hiện tại | Hạng RRF trên hợp các mức đã hiện |
|---|---|---:|---:|
| 1 | two men in a kitchen | Ngoài top 200 | Chưa vào tập hợp |
| 2 | a man in a gray sweater next to a man in black in a kitchen | 155 | 239 |
| 3 | a man in a gray sweater raises his hands near the head of a man in black | 15 | 66 |
| 4 | a man in a gray sweater adjusting the head camera of a man in black | 19 | 33 |

RRF đã đẩy frame mục tiêu xuống ở ví dụ này. Mức 3 cũng xếp frame cao hơn mức 4. Vì mới có một sự kiện và bản EN viết sẵn, chỉ dùng phép thử này để chọn đối chứng và cách kiểm tra, chưa dùng để chọn kỹ thuật cho cả cuộc thi.

### Phương pháp tham khảo từ GitHub của nhóm

Kế hoạch ban đầu tham khảo bản `c1953e5` cho hai kỹ thuật trên. Đối chiếu ngày 24/11 cho thấy nhánh `main` trên GitHub đã tới `b317148`, thêm 23 commit. Ở bản mới, SigLIP 2 chấm mức khớp giữa câu và ảnh toàn cảnh. BM25 chấm thêm mức khớp giữa từ trong câu và metadata từng ảnh về cảnh, người, quần áo, đồ vật. Metadata được Gemini tạo trước, nên bước khớp từ khi có gợi ý mới chạy được tại máy. Bản mới còn có chỉ mục vùng SigLIP 2 và luồng QA dùng ảnh để kiểm chứng đáp án. [Mã kết hợp điểm](https://github.com/nguyenanfms/castle5000-image-retrieval/blob/b317148/backend/app/search.py) · [Mã BM25](https://github.com/nguyenanfms/castle5000-image-retrieval/blob/b317148/backend/app/metadata_search.py) · [Nhật ký phương pháp và kết quả của nhóm](https://github.com/nguyenanfms/castle5000-image-retrieval/blob/b317148/docs/NHAT_KY_2026-11-23.md)

**Thử trên laptop ngày 25/11.** Đã trích 5.000 mô tả và ghép đủ 4.971 ảnh trong chỉ mục. Dùng cùng vector CLIP và thêm điểm từ `MetadataMatcher` của partner với trọng số thử `0,5`. Ở mức 30 giây, laptop từ hạng 30 lên 1 và cảnh sát camera từ 88 lên 4. Hai gợi ý mở đầu rất chung còn ngoài top 200. Trong 9 lượt gợi ý từ ba ca đã biết, số lượt có ảnh đích ở top 5 từ 4 lên 7. Đây là phép thử thăm dò để chọn hướng, dùng tập giữ lại lớn hơn để chọn cấu hình thi. [Kết quả thô](../artifacts/partner_metadata_probe_b317148.json)

**Bản tích hợp web.** Dùng Castle Local làm giao diện và gateway. Web có hai lựa chọn **CLIP** và **CLIP + BM25** trên cùng bộ chỉ mục 4.971 ảnh. Cả ô tìm ảnh và luồng **Xử lý gợi ý** dùng lựa chọn này để lấy ứng viên. [Cách so trực tiếp](../README_VI.md#so-clip-và-clip--bm25-trên-web)

**Thử tiếp ở pha 3.** Giữ cùng ID ảnh và bản EN. Đo CLIP bật/tắt BM25, sau đó tách phép đo SigLIP 2 bật/tắt BM25 khi có runtime. Ghi riêng phần đóng góp của gom sự kiện, tìm vùng và kiểm chứng Gemini. Metadata có từ mô tả người, hành động, vật và vị trí để tìm nhanh. Câu hỏi QA và quyết định tự nộp cần đối chiếu bằng ảnh gốc.

### QA tự động và lựa chọn model

**Luồng QA trước thi:** tìm các frame liên quan → chọn 3–5 frame đại diện các khoảnh khắc khác nhau, hoặc dùng trực tiếp ảnh người thi đã ghim → đưa câu hỏi gốc cùng ảnh toàn cảnh vào model thị giác-ngôn ngữ (VLM) → khi cần, đọc thêm vùng tay, áo hoặc đồ vật ở độ phân giải gốc → hiển thị đáp án ngắn và ảnh chứng minh. Nếu ảnh mờ, chi tiết bị che hoặc ảnh mâu thuẫn, trả trạng thái *chưa đủ bằng chứng* và cho xem frame lân cận. Với câu hỏi đếm trong một ảnh, đếm trong chính frame đó. Câu hỏi toàn bộ sự kiện hoặc toàn bộ corpus cần ghi rõ phạm vi đã xét trước khi kết luận.

| Chế độ | Dùng vào việc gì | Điều kiện chọn khi thi |
|---|---|---|
| Local hoàn toàn | Castle Local đã chạy Qwen3-VL 2B qua Ollama để chuẩn hóa truy vấn và đọc ảnh. Có thể thử VinAI cho bước dịch riêng để giữ phần tìm kiếm nhẹ. | Đo tải nguội, p95 lúc nóng, đỉnh VRAM/RAM và độ chính xác khi CLIP cùng hoạt động. |
| Local + Gemini API | Ưu tiên `gemini-3.5-flash-lite` để thử chuẩn hóa gợi ý VI và đọc top ảnh bằng chứng. Tận dụng mẫu đầu ra theo từng điều kiện của partner. | Dùng khi quy định cho phép đưa đề/ảnh tới nhà cung cấp và tài khoản nhóm xác thực được. Đo số lượt gọi, độ trễ, lỗi và chất lượng bản dịch, kiểm chứng KIS, QA. Hủy việc cũ khi có gợi ý mới. [Luồng QA của partner](../../castle5000-image-retrieval/docs/QA_EVIDENCE_FLOW.md) · [Gemini 3.5 Flash-Lite](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite) |
| Local + OpenAI API | `gpt-5.4-mini` hỗ trợ đầu vào ảnh và đầu ra có cấu trúc. Dùng làm phương án đối chứng trên cùng câu và ảnh. | Cần thêm adapter và tài khoản hợp lệ. [Thông tin model](https://developers.openai.com/api/docs/models/gpt-5.4-mini) |
| Người đọc ảnh | Bản dịch EN có thể chỉnh, CLIP, ảnh lớn, neighbors và ô QA hiện có. | Dùng ngay khi thiếu model, API quá hạn, mất mạng ngoài hoặc model trả lời chưa đủ căn cứ. Nộp DRES thật vẫn cần kết nối tới DRES. |

Đầu ra QA nội bộ gồm `answer`, `evidence_ids`, `status` và chi tiết nhìn thấy. Backend chỉ nhận ID thuộc tập ảnh thật đã gửi cho model. Hiển thị bằng chứng cạnh đáp án, cho sửa rồi xác nhận. Theo mục 8.4 PDF của lớp, trả lời theo ngôn ngữ của đề, trừ khi BTC chỉ định khác. Ví dụ với đề tiếng Việt có thể nộp **đỏ** hoặc **3** tùy câu hỏi. DRES chỉ nhận trường `text` cho QA. KIS gửi đúng một `mediaItemName` với collection `castle5000`. [Hướng dẫn nộp của dự án](../README_VI.md)

### Tự nộp KIS và QA

**Bản thử đã triển khai ngày 25/11/2026.** Castle Local nhận gợi ý người dùng dán, tạo câu tìm tiếng Anh, truy xuất bằng CLIP và dùng Qwen3-VL 2B hoặc Gemini để đọc tối đa tám ảnh ứng viên. Phần nhận gợi ý tự động từ Viewer và benchmark trên nhiều task tiếp tục theo các pha bên dưới. [Cách chạy bản thử](../README_VI.md#tự-nộp-kis-và-qa)

Người dùng chọn **Tự nộp** một lần cho evaluation đang tham gia. Từ đó app quyết định theo gợi ý đã hiện và có thể gửi ngay ở bất kỳ mức nào đủ bằng chứng. Có nút tắt để chuyển về gửi theo xác nhận. Thử luồng này trước với mock local và kiểm tra thể lệ của buổi thi khi chọn chế độ vận hành.

- **KIS.** Tìm ảnh sau mỗi gợi ý. Kiểm chứng ứng viên tốt nhất bằng các chi tiết phân biệt của đề và lấy đúng mã ảnh. Xác lập tiêu chí tự gửi từ số lần đúng, sai và bỏ qua trên tập diễn tập. Rank 1 của CLIP chỉ dùng để chọn ứng viên kiểm chứng.
- **QA.** Dùng truy xuất và VLM để tìm và đọc ảnh. Khi câu trả lời đã qua kiểm chứng bằng ảnh, chuẩn hóa thành một đáp án ngắn theo ngôn ngữ đề rồi gửi trường `text`. Với câu mơ hồ hoặc ảnh thiếu bằng chứng, app đợi gợi ý sau. Tách tiêu chí tự gửi QA khỏi KIS.
- **Gửi và dừng.** Ngay trước khi gửi, kiểm tra evaluation, task, loại bài và phiên bản gợi ý còn khớp. Dùng cùng gateway, lịch sử và khóa chống gửi trùng cho hai chế độ. Bản đầu cho tối đa một lần tự nộp mỗi lượt task. Sau khi nộp thành công, QA đang chờ chấm hoặc timeout chưa rõ kết quả, app dừng tự nộp cho lượt đó. Lịch sử hiển thị đáp án, bằng chứng và phản hồi DRES. [Gateway hiện tại](../castle/dres.py)
- **Test mock.** Phát lại KIS và QA có gợi ý tăng dần, đếm số lần gọi nộp. Kiểm tra một bài đủ bằng chứng tự gửi đúng một lần, bài chưa đủ thì đợi, gợi ý mới đến trong lúc kiểm chứng, task đổi, bấm tay trùng thời điểm, tắt chế độ trong lúc chờ, QA trả 202 và timeout. So thời gian, số lần đúng, sai và bỏ qua với lúc gửi theo xác nhận.

## 5. Các giai đoạn thực hiện và tiêu chí hoàn thành

Lịch dưới đây là phương án cho hai người bắt đầu từ 24/11. A phụ trách web, đồng bộ gợi ý và DRES. B phụ trách dữ liệu, retrieval và model. Hai người đổi vai trong diễn tập, mỗi lượt chỉ định một người bấm nộp. Nếu lịch thi thay đổi, giữ nguyên thứ tự phụ thuộc và dời mốc khóa bản theo ngày thi.

| Giai đoạn | Công việc và đầu ra | Kiểm thử và điều kiện chuyển giai đoạn |
|---|---|---|
| **0 · Chốt baseline, 24/11** | A ghi interpreter, generation, số ảnh, cách mở app. B chạy query chuẩn, lưu thứ hạng từng mức. | Doctor, 30 test hiện có, Chrome tìm ảnh thật và nộp mock KIS/QA. Xem được ảnh lớn và neighbors. Các kết quả lượt này nằm trong [tiến độ](PROGRESS.md). |
| **1 · Đề và nhãn local, 25/11** | B cùng A tạo tối thiểu 12 task luyện gồm 6 KIS, 6 QA, mỗi task 4 hoặc 5 gợi ý VI có thời điểm mở. Mục tiêu cho báo cáo là 30 task cân bằng nếu đủ thời gian. Mock phát lần lượt theo đồng hồ, có chế độ tua nhanh, tạm dừng và chuyển task. | Mỗi KIS có ít nhất một frame đã xem, ghi thêm mọi frame chấp nhận được đã kiểm tra. Mỗi QA có đáp án, biến thể hợp lệ và ảnh chứng minh. Khóa 1/3 số sự kiện mới làm holdout trước khi chỉnh thuật toán. Tự động kiểm tra tại 29,9/30 và 59,9/60 giây để gợi ý chưa mở không vào frontend, dịch, CLIP hay VLM. |
| **2 · Nhập từ TV và dịch, 25–26/11** | A làm luồng gõ tiếng Việt, thêm mức mới, sửa EN và giữ ảnh ghim. B đo thời gian dịch. Bản dùng trong diễn tập là gõ → dịch → tìm → nộp mock. | Phát một task 4 mức và một task 5 mức trên màn hình khác. Thử gõ cụm ngắn, gõ nhầm rồi sửa, refresh, đổi task khi dịch chậm. Đo riêng thời gian từ lúc đề hiện tới khi nhập xong và từ lúc nhập xong tới lưới ảnh. |
| **3 · So truy vấn với metadata, 26–27/11** | B so CLIP và CLIP + BM25 trên cùng kho và gợi ý. Thử SigLIP 2 và tìm theo vùng ở máy có runtime. A đo thời gian nhập từng mức và kiểm tra giao diện còn phản hồi trong lúc model kiểm chứng ảnh. | Báo số sự kiện tìm đúng theo từng mức, lỗi mất ứng viên và thời gian. Dùng CLIP làm mốc để chọn trọng số metadata trên tập dev rồi kiểm tra tập giữ lại. |
| **4 · QA có bằng chứng, 27–28/11** | B tích hợp một VLM sau giao diện chung. A thêm nút gợi ý đáp án, ảnh nguồn và sửa trước khi nộp. So với người đọc ảnh. | Chấm lượt 1 với ảnh đúng đã chọn, lượt 2 để app tự tìm. Đo đúng đáp án, đúng bằng chứng, từ chối đúng khi thiếu ảnh, p95 và lỗi khi API 429/timeout, GPU thiếu bộ nhớ. Mục tiêu đề xuất QA trong ≤ 15 giây p95 khi đã có ảnh, mọi đáp án được đề xuất có ID ảnh hợp lệ. |
| **5 · Tự nộp và thi thử local, 29/11 hoặc sớm hơn** | Nối quyết định KIS/QA với gateway ở mock. Chạy hai buổi trên máy thi, chia task chưa thấy thành hai tập tương đương. So tự nộp với gửi theo xác nhận và đảo thứ tự giữa hai người. | Ghi thời gian gửi, đúng, sai, bỏ qua và mức gợi ý đầu tiên đủ bằng chứng. Chạy ca mất Internet, nhập muộn, khởi động lại và task đổi trong lúc kiểm chứng. Cần 0 lần nộp nhầm task/loại hoặc nộp trùng do hai chế độ cùng gửi trong mock. |
| **6 · Diễn tập DRES và khóa bản, trước thi 1 ngày** | Thành viên đăng nhập tài khoản được cấp khi BTC mở lượt luyện. Đối chiếu task hiện hành với đề chiếu trên TV, tên ảnh và quy tắc dùng API AI. Chốt một cấu hình và một cấu hình dự phòng đã luyện. | Nếu BTC mở đủ hai loại task, thành viên xác minh mỗi luồng bằng một lần nộp hợp lệ trong buổi luyện và ghi verdict. Nếu thiếu một loại, ghi chính xác phần chỉ mới qua mock. Khởi động lại app từ hướng dẫn và làm nóng thành công trước khi khóa. |
| **7 · Báo cáo và bàn giao** | Hai thành viên ghi phần đóng góp, sơ đồ, bộ câu/nhãn tự tạo, cấu hình, giấy phép model dùng, bảng so sánh, ví dụ thành công/thất bại và hướng dẫn chạy lại. | Một thành viên khác làm theo README và tái hiện được buổi mock, ít nhất một kết quả KIS, một QA và bảng đo từ cấu hình đã khóa. Đối chiếu hình thức nộp source/báo cáo với rubric môn. |

Pha 3 và 4 có thể phát triển song song sau khi có bộ nhãn và giao diện ở pha 2. Tự nộp KIS cần bước kiểm chứng ứng viên. Tự nộp QA cần VLM và bằng chứng từ pha 4. Pha 5 đo cả hai và giữ luồng bấm xác nhận làm chế độ dự phòng. Sau mỗi pha, cập nhật lệnh chạy, kết quả thực tế, lỗi và cấu hình được giữ trong [PROGRESS.md](PROGRESS.md). Lỗi của một thành phần phải được xử lý hoặc quay về cấu hình đã qua kiểm tra trước khi dùng nó ở pha sau.

### Đưa bản hiện tại lên nhánh `ph` · 28/11/2026

1. Kiểm tra bản local và repo chung. Hoàn thành khi bộ test baseline đạt, `origin/main` đã xác định và tên nhánh `ph` còn trống.
2. Đặt Castle Local tại `Final_Project/` trên nhánh mới. Hoàn thành khi bộ test và tìm kiếm thử qua cấu trúc mới đạt, hướng dẫn chạy có trên repo, và tập thay đổi chỉ gồm mã nguồn, cấu hình và tài liệu cần bàn giao.
3. Push `ph` lên repo nhóm. Hoàn thành khi commit đầu nhánh GitHub khớp commit local và SHA của `main` khớp mốc trước lần push.

## 6. Đo hiệu quả và quyết định bản đem thi

- **Bộ dữ liệu đánh giá.** Mỗi task có mã sự kiện, loại, các cặp thời điểm–gợi ý, ảnh đúng đã xem và đáp án QA nếu có. Giữ đề và tệp chấm ở hai phía của mock. Chia theo sự kiện, giữ các frame liền nhau của cùng cảnh trong cùng split. Hai người xem ảnh và thống nhất các trường hợp rửa tay/rửa dụng cụ, số người bị che, trái/phải theo góc nhìn. Báo riêng các ảnh chưa chấm. Cố định tập giao các mã ảnh và ghi số lượng khi so với repo partner.
- **KIS.** Với từng mức, báo Hit@1/10/50, MRR và tỉ lệ sự kiện có ảnh đúng trong top 100/200/500 trước rerank. Ghi số đếm dạng `n/N`, mức đầu tìm trúng, thời gian tới ảnh đúng đầu tiên, thời gian người thật nộp đúng, số lượt sai, số task hết giờ. Khi timestamp bị thiếu, ghi là thiếu. Nếu chỉ báo thời gian trên task đã giải thì báo kèm tỉ lệ giải trên toàn tập.
- **Nhập và dịch.** So EN do người viết với VI nhập nguyên câu và VI nhập cụm ngắn từ TV. Chấm thủ công việc giữ số lượng, màu, phủ định, quan hệ và chủ thể. Báo thời gian nhập, tỷ lệ cần sửa bản dịch và thời gian sửa. Khi dịch đúng mà truy vấn vẫn sai, ghi lỗi ở tầng tìm ảnh.
- **QA.** Với ảnh đã biết đúng, chấm đáp án sau chuẩn hóa hoặc theo người chấm với bộ biến thể ngắn. Với QA đầu-cuối, báo đồng thời câu trả lời đúng, ảnh dẫn chứng đúng, số câu có trả lời/tổng câu và số câu sai nhưng model vẫn đề xuất. Ảnh đúng + đáp án sai → xem lại đọc ảnh. Ảnh sai + đáp án tình cờ đúng → đánh dấu thiếu bằng chứng. Thêm ca không thể trả lời được từ ảnh đang có.
- **Hiệu năng.** Ghi thời điểm thấy chữ, có EN, lưới ảnh vẽ xong, QA xong, người bấm nộp và lúc có phản hồi. Tính p50/p95 trên tối thiểu 30 lần cập nhật khi có thể, ghi kích thước mẫu, tách cold/warm và cache hit/miss. Với local VLM ghi peak VRAM, RAM và ảnh hưởng đến CLIP. Với API ghi số ảnh, số request, mã lỗi, chi phí thực và điều kiện mạng.
- **Giao thức nộp.** Trong mock, kiểm tra KIS đúng/sai, QA 202 chờ chấm, QA 200 có verdict ngay, 401 hết phiên, 412 trùng bài, timeout sau khi máy chủ có thể đã nhận và chuyển task giữa lúc xác nhận. Đối chiếu payload và số lần gửi. Mọi lần thử lỗi dùng host local hoặc HTTP giả lập. Trước khi vào thi, người nộp phải phân biệt được `MOCK_ACCEPTED`, `CORRECT`, `WRONG`, `INDETERMINATE` và `UNKNOWN`.
- **Quyết định.** Chọn cấu hình trên dev và khóa trước khi mở holdout. Đưa cải tiến vào bản thi khi giúp giải đúng thêm task hoặc rút ngắn thời gian trên các task cùng giải, đồng thời giữ tỉ lệ nộp sai và độ trễ trong ngưỡng đã thống nhất. Nếu hai cấu hình đổi tốc độ lấy độ chính xác, trình bày cả hai con số và áp dụng quy tắc phạt BTC công bố. Nếu mẫu ít hoặc kết quả ngang nhau, mang bản đơn giản đã luyện. Báo cáo giữ nguyên kết quả holdout và các trường hợp thất bại.

## 7. Điểm tích hợp và thao tác ngày thi

| Vị trí | Phần cần phát triển |
|---|---|
| [Web local](../web/app.js) và [HTML](../web/index.html) | Nhập VI từ TV, sửa EN, danh sách mức, ảnh ghim, đáp án QA và nguồn ảnh. Giữ riêng hai nút nộp theo loại task. |
| [FastAPI](../castle/app.py) và [chỉ mục](../castle/index.py) | Tái dùng `POST /api/search`, `GET /api/image/{key}`, `GET /api/neighbors/{key}`. Thêm dịch/hợp hạng/đọc ảnh qua các đầu vào tách biệt, trả kết quả kèm phiên bản gợi ý. Cache vector theo model và truy vấn. |
| [Gateway DRES](../castle/dres.py) | Tái dùng task hiện hành, xác nhận và lịch sử. Giữ SessionId ở backend và loại khỏi log/ảnh chụp. |
| [Evaluation](../evaluation/README.md), [tests](../tests/test_system.py) và [smoke trình duyệt](../scripts/smoke_browser.py) | Thêm bộ đề theo mức, đồng hồ giả lập, chấm theo sự kiện, số liệu thời gian và test UI với phản hồi đến sai thứ tự. |

**Lệnh hiện có để kiểm tra local** chạy từ PowerShell trong `D:\UIT\DS314\Final_Project`. Nếu cổng 8000 đang phục vụ app, dùng luôn server đó cho bước Chrome.

```powershell
cd D:\UIT\DS314\Final_Project
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\doctor.ps1
& D:\UIT\.venvs\ml-gpu\Scripts\python.exe -m unittest discover -s tests -v
# Khi server chưa chạy, mở trong một terminal riêng và giữ terminal đó mở.
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start.ps1
# Trong terminal chạy kiểm tra:
& D:\UIT\.venvs\ml-gpu\Scripts\python.exe scripts\smoke_browser.py --expected-count 4971
```

**Trước giờ thi.** Mở web local, xác nhận index sẵn sàng, chạy một tìm kiếm làm nóng, mở thử ảnh lớn, kiểm tra chế độ luyện/thật và chỉ định người nộp. Hai người luyện một lượt đọc TV, nhập gợi ý và nộp mock. Nếu dùng VLM, đo thời gian với một ảnh luyện.

**Trong task.** Người A đọc gợi ý trên TV, gõ tiếng Việt và kiểm tra bản EN. Người B xem ứng viên và lịch sử nộp. Trong chế độ tự nộp, app chọn và gửi một mã ảnh KIS hoặc câu trả lời QA khi đủ bằng chứng. Trong chế độ xác nhận, người B kiểm tra ảnh lớn, neighbors hoặc câu trả lời QA rồi bấm gửi. Sau `202 INDETERMINATE`, ghi nhận đang chờ trọng tài. Sau timeout `UNKNOWN`, xem lịch sử và trạng thái task trước khi quyết định gửi lại. Ảnh đang cân nhắc vẫn xem được qua danh sách ghim cho tới khi đổi task.

**Bàn giao.** Lưu phiên bản code, generation index, danh sách ID, phiên bản model, cấu hình provider và kết quả đo đã loại thông tin phiên. Bàn giao hướng dẫn khởi động và dữ liệu đánh giá theo quy định của lớp. Bản thử đã có dịch qua model, QA bằng VLM và tự nộp. Luồng nhập nhanh từ TV, phát lại nhiều mức và hợp hạng tiếp tục theo các pha triển khai.
