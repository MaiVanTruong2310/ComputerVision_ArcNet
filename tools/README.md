# 🛠️ Bộ Công Cụ Tiện Ích ArcNet (Tools)

Thư mục này tập hợp các kịch bản tiện ích phụ trợ cho quá trình xuất mô hình, đánh giá hàng loạt và kiểm chuẩn tốc độ:

---

## 📌 Danh Sách Công Cụ

### 1. `export_onnx.py`
Chuyển đổi trọng số mô hình PyTorch (`.pth`) sang định dạng **ONNX (`.onnx`)** với cấu trúc đồ thị tính toán tối ưu, dynamic batching và hằng số được gấp gọn (constant folding):
```bash
python tools/export_onnx.py --checkpoint checkpoints/finetuned_best.pth --output outputs/finetuned_best.onnx
```

### 2. `generate_report.py`
Tự động tính toán toàn diện các chỉ số phân biệt sinh trắc học (**Accuracy, EER, AUC, F1-Score, TPR @ FAR = 1%, TPR @ FAR = 0.1%, Fisher Ratio FDR**) và trích xuất biểu đồ phân tích sang thư mục `evaluation_reports/`:
```bash
python tools/generate_report.py --checkpoint checkpoints/finetuned_best.pth --target lfw
```

### 3. `evaluate_results.py`
Quét và đánh giá độ chính xác hàng loạt qua toàn bộ các checkpoint có trong thư mục `checkpoints/` để vẽ biểu đồ tiến độ hội tụ và tìm ra epoch tốt nhất:
```bash
python tools/evaluate_results.py --checkpoint checkpoints --dataset dataset/raw/lfw.bin
```

### 4. `test_speed.py`
Đo đạc tốc độ đọc nạp dữ liệu của **DataLoader** và đo tốc độ suy luận (forward & backward throughput / iterations per second) của mạng trên GPU CUDA:
```bash
python tools/test_speed.py
```
