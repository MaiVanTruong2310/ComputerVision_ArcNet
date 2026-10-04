# Bộ Công Cụ Kiểm Thử Căng Thẳng (Stress Testing Suite) cho ArcFace

Thư mục này chứa toàn bộ các kịch bản kiểm thử khắt khe nhằm đánh giá tính bảo mật, độ tin cậy và tốc độ xử lý của mô hình `backbone_best.pth` đã huấn luyện trước khi tích hợp vào ứng dụng di động PassNote.

---

## 📂 Cấu trúc thư mục kết quả
Sau khi chạy các file kiểm thử, kết quả số liệu và biểu đồ sẽ được tự động ghi vào thư mục:
*   `robustness_testing/results/`

Các tệp kết quả bao gồm:
*   **Biểu đồ tác động (.png)**: Thể hiện trực quan ảnh hưởng của ánh sáng, mờ nhòe, độ phân giải thấp, góc xoay nghiêng và đa luồng lên mô hình.
*   **Bảng số liệu thô (.csv)**: Lưu trữ các chỉ số để nạp vào Excel hoặc phân tích khoa học sau này.
*   **Báo cáo tóm tắt (.md)**: Báo cáo tổng hợp kết quả chi tiết từng phần.

---

## 🚀 Hướng dẫn chạy các kịch bản kiểm thử

### 1. Đánh giá độ bền môi trường (Ánh sáng, Mờ nhòe, Độ phân giải, Che khuất, Xoay nghiêng)
Script này tải tập dữ liệu LFW, áp dụng các phép làm nhiễu vật lý lập trình, và đo đạc sự biến thiên của Accuracy (tại ngưỡng cố định `1.60`) và EER.
```bash
python robustness_testing/perturbation_tests.py
```
*   *Đầu ra*: Tạo ra 5 biểu đồ tác động tương ứng và file báo cáo `perturbation_report.md` tại thư mục `results/`.

### 2. Đánh giá Tốc độ và Ép tải Đa luồng (Speed & Concurrency)
Đo tốc độ phản hồi (Latency), thông lượng (FPS) ở các Batch Size từ 1 đến 16 trên cả CPU và GPU. Đồng thời giả lập đa luồng gửi yêu cầu đồng thời để đo độ chịu tải và mức RAM/VRAM sử dụng.
```bash
python robustness_testing/speed_stress_test.py
```
*   *Đầu ra*: Tạo ra biểu đồ phân bổ độ trễ `latency_stress_distribution.png`, biểu đồ `fps_vs_batch_size.png` và file báo cáo hiệu năng `speed_stress_report.md` tại thư mục `results/`.

### 3. Đánh giá các cặp dễ nhận nhầm (Hard Negatives) & Tự kiểm thử ảnh Gia đình/Anh em
Script này tìm ra 20 cặp hình ảnh khác người trong LFW có khuôn mặt giống nhau nhất và kiểm tra xem mô hình có bị nhận nhầm ở các ngưỡng bảo mật hay không.
```bash
python robustness_testing/lookalike_twins_test.py
```
*   *Đầu ra*: Danh sách 20 cặp kèm chỉ số Cosine và Euclidean lưu tại `results/lfw_hard_negatives_report.md`.

#### 👥 Hướng dẫn tự kiểm thử ảnh Gia đình / Anh em / Sinh đôi tự chọn:
1. Bạn hãy chụp hoặc chọn các ảnh cận cảnh khuôn mặt của bản thân, anh em hoặc người thân trong gia đình.
2. Cắt (crop) sát khuôn mặt (giống kiểu ảnh thẻ hoặc Passport) để đạt hiệu quả tối đa (mô hình mong muốn dữ liệu đầu vào đã căn chỉnh khuôn mặt).
3. Đặt tất cả các ảnh đó trực tiếp vào thư mục:
    `robustness_testing/custom_test_images/`
4. Chạy lại lệnh:
    ```bash
    python robustness_testing/lookalike_twins_test.py
    ```
5. Mở file báo cáo kết quả [custom_comparison_report.md](file:///d:/Mon%20Hoc/Lap%20trinh%20AI/ArcFace/robustness_testing/results/custom_comparison_report.md) trong thư mục `results/` để xem mô hình đánh giá các cặp ảnh của bạn là **CÙNG MỘT NGƯỜI (SAME)** hay **KHÁC NHAU (DIFF)** ở các mức bảo mật khác nhau (`ngưỡng 1.60`, `1.40`, hoặc `1.20`).
