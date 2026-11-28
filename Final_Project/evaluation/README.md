# Bộ query nội bộ, 20/11/2026

Các mô tả do trợ lý viết sau khi trực tiếp xem ảnh gốc 1280×720, **không phải nhãn BTC**. File JSONL lưu imageID theo quy tắc người dùng cung cấp, không phải metadata chính thức. Không dùng mô tả này làm caption của corpus.

- `dev.jsonl`: 9 query để phân tích lỗi.
- `holdout.jsonl`: 3 query cố định trước cải tiến. Chưa dùng chọn thay đổi retrieval. Bộ nhỏ, cùng camera/bối cảnh; không chứng minh khái quát hóa.
- Nhãn chỉ gồm ảnh trực tiếp xác minh; còn nhiều ảnh phù hợp chưa gán nhãn. Với query “empty kitchen” đã xác nhận ba ảnh, các query khác một ảnh. Hit@K thấp có thể do nhãn thiếu, không có nghĩa mọi ảnh xếp trên đều sai.
- Query chỉnh camera dùng **08_269**, đã nhìn thấy hai tay đang chỉnh camera trên trán người đối diện. Không dùng ví dụ 08_266 làm nhãn hành động này vì ảnh thật đó chưa thể hiện thao tác chỉnh.
- Chọn ảnh giữa mỗi giờ để có mốc tái lập, thêm 08_269 cho hành động chi tiết. Không dựa vào rank để chọn nhãn, không tự coi frame cạnh nhau là relevant.

```powershell
& D:\UIT\.venvs\ml-gpu\Scripts\python.exe -m castle evaluate evaluation\dev.jsonl
& D:\UIT\.venvs\ml-gpu\Scripts\python.exe -m castle evaluate evaluation\holdout.jsonl
```

CLI hiện ghi `artifacts/evaluation_report.json` mỗi lần; sao chép report theo split trước lần kế tiếp. Kết quả/điều kiện đo xem `notes/EXPERIMENTS.md`.

## Đề KIS tìm đúng một frame

- **Ground truth** là [ảnh Batch2/20_3449.jpg](../data/Batch2/20_3449.jpg), mã `day1_Kitchen_20_3449`. Chọn ảnh trước khi soạn gợi ý. Hai mép ảnh bị hai bàn tay che ở các vị trí khác nhau. Mặt người áo khoác xám ở sát ống kính, còn sofa ca rô đỏ nằm phía sau bên trái.
- Đã xem 42 ảnh khác nhau trong hợp của 11 frame lân cận gồm cả ảnh đích, 15 hàng xóm gần nhất theo embedding ảnh có tính ảnh đích và 20 kết quả CLIP từ gợi ý cuối. Chỉ ảnh đích có đầy đủ bố cục nêu trên trong nhóm đã xem. [Frame lân cận](../artifacts/kis_gt_20_3449_neighbors.jpg) · [Ảnh gần theo embedding](../artifacts/kis_gt_20_3449_similar.jpg) · [Top 20 gợi ý cuối](../artifacts/kis_gt_20_3449_final_top20.jpg)
- Ca trước mô tả một sự kiện chỉnh camera. Cả `08_269.jpg` và `08_272.jpg` thể hiện thao tác đó, nên câu cũ hợp để luyện tìm sự kiện. [Frame hai ảnh chỉnh camera](../artifacts/kis_gt_08_269_neighbors.jpg)

| Giây | Gợi ý VI ở mức hiện tại | Câu EN đã chạy qua CLIP | Hạng ground truth |
|---:|---|---|---:|
| 0 | Một người đàn ông trong bếp. | a man in a kitchen | Ngoài top 200 |
| 30 | Một người đàn ông mặc áo khoác xám có khóa kéo trong bếp. | a man wearing a gray zip-up jacket in a kitchen | 88 |
| 60 | Người mặc áo khoác xám đứng sát camera, nhìn thẳng vào ống kính và đưa cả hai tay về phía trước. | a man in a gray zip-up jacket close to the camera, looking straight into the lens and reaching toward it with both arms | 1 |
| 90 | Cận mặt người áo khoác xám nhìn vào ống kính, hai tay vươn ra. Một bàn tay che mép phải, tay kia lọt vào góc dưới bên trái. Phía sau có sofa ca rô đỏ. | close-up of a man in a gray zip-up jacket looking into the lens with both arms extended. A hand obscures the right edge and another enters at the lower left. A red plaid sofa is behind him | 1 |

Hạng tính trên toàn bộ 4.971 ảnh, bỏ lọc giờ và Batch. Dùng ô **Tìm ảnh** với bản EN để tái lập thứ hạng. Dùng **Xử lý gợi ý** để thử thêm bước kiểm chứng tự nộp. Trong lượt chạy ngày 25/11, gợi ý cuối vẫn đưa ảnh đích lên hạng 1 qua luồng tự nộp. Model chưa xác nhận bằng chứng nên giao diện báo **Chưa đủ bằng chứng** sau 13,24 giây.

## Đề KIS laptop có mã QR

Ground truth là [Batch1/08_353.jpg](../data/Batch1/08_353.jpg), mã `day1_Kitchen_08_353`. Ảnh có laptop bạc hướng thẳng vào camera, mã QR sáng giữa màn hình xanh đậm, laptop che gần hết mặt người cầm và người áo ca rô đỏ đứng cạnh bếp nấu. Hai ảnh `08_358` và `13_341` cũng có laptop, với hướng màn hình và vị trí trong ảnh khác frame đích. [Ảnh lân cận và ảnh gần theo embedding](../artifacts/kis_gt_08_353_comparison.jpg)

| Giây | Câu tìm tiếng Anh | Hạng ảnh đích |
|---:|---|---:|
| 0 | several people in a kitchen | Ngoài top 200 |
| 30 | a person holding an open silver laptop in a kitchen | 30 |
| 60 | an open silver laptop held close to the camera in a kitchen with a bright QR code on a dark blue screen | 1 |
| 90 | a silver laptop held directly in front of the camera, with a bright QR code centered on its dark blue screen. The keyboard is in the foreground and the laptop hides most of the holder's face. A man in a red plaid shirt stands by the stove behind it | 1 |

Dùng ô **Tìm ảnh** với 200 kết quả và bỏ lọc Batch/giờ để lặp lại. Câu cuối bản VI là “Một chiếc laptop bạc được giơ sát và hướng thẳng vào camera. Mã QR sáng nằm giữa màn hình xanh đậm, bàn phím ở tiền cảnh, laptop che gần hết mặt người cầm. Phía sau có người mặc áo ca rô đỏ đứng cạnh bếp nấu.” Khi chạy qua **Xử lý gợi ý**, ảnh đích vẫn hạng 1 và model báo **Chưa đủ bằng chứng**.
