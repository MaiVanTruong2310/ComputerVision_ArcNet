# ArcFace Training Pipeline with IResNet-50

Dự án này triển khai quy trình huấn luyện nhận diện khuôn mặt sử dụng loss function **ArcFace** và kiến trúc backbone **IResNet-50** (Improved ResNet). 

Đặc biệt, dự án sử dụng bộ đọc **Pure Python RecordIO** tự viết, giúp bạn huấn luyện trực tiếp từ file `.rec` của InsightFace mà **KHÔNG cần cài đặt thư viện MXNet** (tránh được các lỗi tương thích trên Windows và Python 3.10+).

## 🛠️ Yêu cầu hệ thống

- **Python 3.8+** (Đã kiểm tra ổn định trên 3.10 và 3.11)
- **GPU hỗ trợ CUDA** (Khuyến nghị để huấn luyện nhanh)
- **Thư viện chính**: PyTorch, Torchvision, OpenCV, Scikit-learn, Scipy, Pillow

## 🚀 Cài đặt nhanh

1. Cài đặt các thư viện cần thiết:
```powershell
pip install -r requirements.txt
```

2. Chuẩn bị dữ liệu:
Tạo thư mục `dataset/` và đặt các file dữ liệu vào trong. Dự án sẽ tự động tìm kiếm trong `dataset/` hoặc `dataset/raw/`:
- `train.rec`, `train.idx` (Dữ liệu huấn luyện)
- `lfw.bin`, `agedb_30.bin` (Dữ liệu đánh giá)

3. Kiểm tra thông tin Dataset:
Sử dụng công cụ kiểm tra để đếm số lượng Class (ID) và tổng số ảnh:
```powershell
python src/inspect_dataset.py
```

## 📁 Cấu trúc thư mục

```text
ArcNet_Production/
├── src/                               # Mã nguồn cốt lõi (Core Training Pipeline)
│   ├── model.py                      # Kiến trúc IResNet-50 & ArcFace Loss
│   ├── recordio.py                   # Pure Python RecordIO reader (Zero MXNet)
│   ├── dataset.py                    # DataLoader & Augmentations
│   ├── train.py                      # Huấn luyện mô hình, Optimizer, Scheduler
│   ├── eval.py                       # Đánh giá Accuracy & Validation metrics
│   ├── ema.py                        # Exponential Moving Average cho weights
│   ├── config.py                     # Cấu hình hyperparameters
│   ├── inspect_dataset.py            # Công cụ kiểm tra metadata file .rec
│   └── utils.py                      # Tiện ích đọc dữ liệu .bin (LFW, AgeDB, CFP)
├── fine_tune/                         # Quy trình Fine-tuning chuyên sâu
│   ├── config_finetune.py            # Cấu hình fine-tuning
│   ├── dataset_finetune.py           # Dataset loader cho fine-tuning
│   └── train_finetune.py             # Script huấn luyện fine-tune
├── robustness_testing/                # Đánh giá độ bền vững và stress test
│   ├── lookalike_twins_test.py       # Test phân biệt người giống nhau / sinh đôi
│   ├── perturbation_tests.py         # Test nhiễu, mờ, độ sáng, góc xoay
│   ├── speed_stress_test.py          # Benchmark tốc độ và độ trễ FPS
│   └── custom_test_images/           # Ảnh thử nghiệm kiểm thử thực tế
├── Web/                               # Giao diện Web Demo & Dashboard trực quan
│   ├── index.html                    # Trang chủ hệ thống ArcNet
│   ├── live_demo.html                # Nhận diện khuôn mặt thời gian thực
│   ├── webcam_realtime.html          # WebCam live verification
│   ├── robustness_evaluation.html    # Báo cáo đánh giá robustness
│   ├── sota_techniques.html          # So sánh công nghệ SOTA
│   ├── training_process.html         # Trực quan hóa tiến trình huấn luyện
│   ├── app.js                        # Xử lý logic phía client
│   └── style.css                     # Giao diện hiện đại, responsive
├── information/                       # Báo cáo kỹ thuật và biểu đồ huấn luyện
│   ├── model_summary.md              # Báo cáo chi tiết thông số mô hình
│   └── training_curve.png            # Biểu đồ loss & accuracy theo epochs
├── main.py                            # Entry point chính (Train / Evaluation)
├── export_onnx.py                     # Chuyển đổi mô hình PyTorch sang định dạng ONNX
├── test_speed.py                      # Đo tốc độ suy luận (Inference Speed Benchmark)
├── evaluate_results.py                # Tổng hợp số liệu và sinh báo cáo đánh giá
├── generate_report.py                 # Tạo báo cáo tự động
├── requirements.txt                   # Danh sách thư viện phụ thuộc
├── run_train.bat                      # Script chạy huấn luyện 1-click trên Windows
└── README.md                          # Tài liệu hướng dẫn sử dụng
```

## 🏋️ Huấn luyện (Training)

Chạy lệnh sau để bắt đầu huấn luyện:
```powershell
python main.py --mode train
```
Hoặc sử dụng file script tiện lợi trên Windows:
- Click đúp vào `run_train.bat`

## 📊 Đánh giá (Evaluation)

Để kiểm tra độ chính xác của một checkpoint trên tập dữ liệu kiểm thử (ví dụ LFW):
```powershell
python main.py --mode eval --backbone checkpoints/backbone_epoch_2.pth --target lfw
```

## 🌐 Chạy Web Dashboard & Demo

Để xem báo cáo và trải nghiệm giao diện demo:
1. Mở file `Web/index.html` trực tiếp trên trình duyệt hoặc sử dụng Live Server (VS Code).
2. Trải nghiệm các trang:
   - **Live Demo & Webcam**: Thử nghiệm nhận diện khuôn mặt qua webcam hoặc tải ảnh lên.
   - **Robustness Evaluation**: Xem kết quả thử nghiệm độ bền vững trước nhiễu và các cặp khuôn mặt tương đồng.
   - **Training Process**: Xem biểu đồ đường cong hội tụ của mô hình.

## 📝 Đặc điểm nổi bật

- **Zero MXNet Dependency**: Bộ đọc RecordIO thuần Python giải quyết triệt để lỗi xung đột môi trường trên Windows & Python 3.10+.
- **IResNet-50 Backbone**: Kiến trúc tối ưu cho trích xuất vector đặc trưng khuôn mặt (Feature Embedding).
- **Robust ArcFace Loss**: Xử lý góc lớn ($\theta + m > \pi$) giúp gradient ổn định và model hội tụ mượt mà.
- **ONNX Export & Fast Inference**: Hỗ trợ xuất sang ONNX runtime để tối ưu hóa tốc độ triển khai thực tế.
- **Complete Suite**: Tích hợp đầy đủ từ Data Inspection, Training, Fine-tuning, Stress Testing đến Web Dashboard.

