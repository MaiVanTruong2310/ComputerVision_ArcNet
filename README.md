# 🌟 ArcNet: Production Face Recognition Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX%20Runtime-Web%20%7C%20CPU%20%7C%20GPU-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/Repository-ComputerVision__ArcNet-blue?logo=github)](https://github.com/MaiVanTruong2310/ComputerVision_ArcNet.git)

Hệ thống nhận diện khuôn mặt hoàn chỉnh chất lượng công nghiệp (**Production-Ready Face Recognition Pipeline**) triển khai kiến trúc **IResNet-50** kết hợp hàm mất mát biên độ góc **ArcFace (Additive Angular Margin Loss)**.

Đặc biệt, dự án tích hợp bộ nạp dữ liệu **Pure Python RecordIO Reader** tự phát triển, cho phép đọc và giải mã trực tiếp định dạng `.rec` của InsightFace mà **hoàn toàn không cần cài đặt thư viện MXNet** (giải quyết triệt để vấn đề tương thích môi trường trên Windows 10/11 và Python 3.10+).

---

## 📌 Mục lục
- [📥 Tải Mô Hình Đã Huấn Luyện (Model Weights)](#-tải-mô-hình-đã-huấn-luyện-model-weights)
- [⚙️ Thông Số Kỹ Thuật & Cấu Hình Huấn Luyện](#️-thông-số-kỹ-thuật--cấu-hình-huấn-luyện)
- [📊 Kết Quả Đánh Giá & Benchmark Chi Tiết](#-kết-quả-đánh-giá--benchmark-chi-tiết)
- [📁 Cấu Trúc Thư Mục Dự Án](#-cấu-trúc-thư-mục-dự-án)
- [🛠️ Cài Đặt Môi Trường](#️-cài-đặt-môi-trường)
- [📦 Chuẩn Bị Dữ Liệu](#-chuẩn-bị-dữ-liệu)
- [🚀 Hướng Dẫn Sử Dụng](#-hướng-dẫn-sử-dụng)
- [🌐 Web Dashboard & Realtime Demo](#-web-dashboard--realtime-demo)
- [🛡️ Kiểm Thử Độ Bền Vững (Robustness Suite)](#️-kiểm-thử-độ-bền-vững-robustness-suite)

---

## 📥 Tải Mô Hình Đã Huấn Luyện (Model Weights)

Để thuận tiện cho việc kiểm thử và triển khai mà không làm nặng kho lưu trữ Git, toàn bộ trọng số mô hình đã được tải lên Google Drive:

| Mô hình | Định dạng | Kích thước | Mô tả | Liên kết tải về |
| :--- | :---: | :---: | :--- | :---: |
| **Fine-tuned Model (`finetuned_best.pth`)** | `.pth` / `.onnx` | ~166 MB | Checkpoint tối ưu nhất sau quy trình Fine-tuning | [👉 Tải Model Fine-tune](https://drive.google.com/file/d/1AYGvCqHuC36G0FcGBhAtCMOg5Xh_wJ4y/view?usp=drive_link) |
| **Best Backbone (`backbone_best.pth`)** | `.pth` | ~166 MB | Trọng số Backbone tốt nhất thu được sau quá trình huấn luyện gốc | [👉 Tải Best Backbone](https://drive.google.com/file/d/1BqXkClfXS3Skcd8WNJBo9C0GsKz01Tc-/view?usp=drive_link) |

> [!TIP]
> **Vị trí đặt checkpoint sau khi tải về:**
> - Cho huấn luyện / đánh giá PyTorch: lưu vào thư mục `checkpoints/` (ví dụ: `checkpoints/finetuned_best.pth`).
> - Cho Web Demo nhận diện client: xuất hoặc đặt file ONNX vào thư mục `Web/models/finetuned_best.onnx`.

---

## ⚙️ Thông Số Kỹ Thuật & Cấu Hình Huấn Luyện

Toàn bộ các siêu tham số (Hyperparameters) và đặc tả kỹ thuật được cấu hình tối ưu cho năng lực trích xuất đặc trưng và hội tụ ổn định:

### 1. Kiến Trúc Mạng (Model Architecture)
- **Backbone**: `IResNet-50` (Improved ResNet-50 với cấu trúc residual đảo ngược và pre-activation).
- **Tổng số tham số (Total Parameters)**: **43,599,744** (~43.6 triệu tham số).
- **Kích thước ảnh đầu vào**: $112 \times 112 \times 3$ (Khuôn mặt căn chỉnh 5 điểm mốc theo chuẩn InsightFace).
- **Kích thước Vector Đặc trưng (Embedding Size)**: **512 chiều** (512-D Float Vector).
- **Chuẩn hóa vector**: L2 Normalization ($\|x\|_2 = 1$).
- **Độ phức tạp tính toán**: ~6.3 GFLOPs.

### 2. Hàm Mất Mát ArcFace (Additive Angular Margin Loss)
$$\mathcal{L}_{Arc} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cdot \cos \theta_j}}$$
- **Hypersphere Scale ($s$)**: `64.0` (Bán kính siêu cầu nhằm kéo giãn độ dốc phân biệt).
- **Angular Margin ($m$)**: `0.4` rad (~23° ép góc margin phạt để phân tách class).
- **Chống bùng nổ góc**: Cơ chế Cosine Expansion xử lý biên độ $\theta + m > \pi$.

### 3. Tối Ưu Hóa & Lịch Trình Huấn Luyện (Optimization Pipeline)
- **Tập dữ liệu huấn luyện**: CASIA-WebFace / MS1MV2 aligned (112x112), gồm **10,572 người (IDs)** và **494,149 ảnh**.
- **Optimizer**: `SGD` (Stochastic Gradient Descent) với Động lượng `momentum = 0.9`.
- **Weight Decay**: `5e-4` (L2 Regularization).
- **Gradient Clipping**: Cắt tỉa chuẩn Gradient tại ngưỡng `5.0`.
- **Base Learning Rate**: `0.03` (Khởi đầu từ `1e-4`).
- **LR Scheduler**: `CosineAnnealingLR` kết hợp **LR Warmup (3 epochs)**.
- **Margin Warmup**: Tăng dần tham số margin $m$ từ `0.0` đến `0.4` trong **5 epochs đầu** nhằm tránh sốc mô hình trong pha sơ khởi.
- **Kỹ thuật Smooth Weights**: Exponential Moving Average (`EMA` với decay `0.999`).
- **Tăng tốc phần cứng**: Automatic Mixed Precision (`AMP FP16`) trên GPU NVIDIA GeForce RTX 3090 Ti (Thời gian huấn luyện ~10.4 phút / epoch).

---

## 📊 Kết Quả Đánh Giá & Benchmark Chi Tiết

Mô hình được thẩm định độc lập trên 3 tập kiểm chuẩn quốc tế khắt khe: **LFW** (Ảnh tự nhiên thực tế), **AgeDB-30** (Biến thiên độ tuổi lớn), và **CFP-FP** (Góc nhìn nghiêng Frontal-to-Profile):

### 1. Bảng So Sánh Hiệu Năng: Pretrained vs Fine-tuned

| Tập kiểm thử | Chỉ số đánh giá | Pretrained (`backbone_best.pth`) | Fine-tuned (`finetuned_best.pth`) | Chênh lệch / Đánh giá |
| :--- | :--- | :---: | :---: | :---: |
| **LFW** | **Độ chính xác (Accuracy)** | 98.40% ± 0.54% | **98.53% ± 0.64%** | <font color="#28a745">**+0.13%**</font> |
| | **Equal Error Rate (EER)** | 0.0180 | **0.0170** | <font color="#28a745">**-0.0010 (Tốt hơn)**</font> |
| | **Area Under Curve (AUC)** | 0.9972 | **0.9970** | Duy trì mức hoàn hảo |
| | **F1-Score** | 0.9841 | **0.9852** | <font color="#28a745">**+0.0011**</font> |
| | **Best Cosine Threshold** | 0.197 | **0.215** | Ngưỡng phân tách rõ nét |
| | **Best L2 Threshold** | 1.60 | **1.57** | Gom cụm chặt hơn |
| | **TPR @ FAR = 1%** | 97.70% | **98.00%** | <font color="#28a745">**+0.30%**</font> |
| | **TPR @ FAR = 0.1%** | 94.70% | **94.73%** | Tăng độ bảo mật |
| | **Fisher Ratio (FDR)** | 8.4440 | **8.7614** | <font color="#28a745">**+0.3174 (Tách cụm tốt hơn)**</font> |
| **AgeDB-30** | **Độ chính xác (Accuracy)** | 87.10% ± 1.98% | **87.23% ± 1.88%** | <font color="#28a745">**+0.13%**</font> |
| | **Area Under Curve (AUC)** | 0.9440 | **0.9447** | <font color="#28a745">**+0.0007**</font> |
| | **Equal Error Rate (EER)** | 0.1278 | **0.1278** | Ổn định |
| **CFP-FP** | **Độ chính xác (Accuracy)** | 88.13% ± 1.72% | **88.30% ± 1.73%** | <font color="#28a745">**+0.17%**</font> |
| | **Area Under Curve (AUC)** | 0.9393 | **0.9425** | <font color="#28a745">**+0.0032**</font> |
| | **Equal Error Rate (EER)** | 0.1284 | **0.1266** | <font color="#28a745">**-0.0018 (Tốt hơn)**</font> |

### 2. Độ Bền Bỉ Ngoại Cảnh (Robustness Metrics)
- **Pose Robustness (Khả năng chịu góc quay mặt LFW vs CFP-FP)**:
  - Độ suy giảm chính xác: **10.23%** (ở bản fine-tuned, cải thiện tốt hơn mức 10.27% của bản gốc).
- **Age Robustness (Khả năng chịu thay đổi độ tuổi LFW vs AgeDB-30)**:
  - Độ suy giảm chính xác: **11.30%** (khẳng định vector embedding giữ vững định danh sinh trắc học qua thời gian).

### 3. Tốc Độ Suy Luận & Hiệu Năng Thực Tế (Inference Speed)
- **GPU (PyTorch FP16 - RTX 3090 Ti)**: **~4.2 ms / ảnh** (~238 FPS).
- **CPU (ONNX Runtime WebAssembly - SIMD Multi-threaded)**: **~18 ms / ảnh** (~55 FPS).

---

## 📁 Cấu Trúc Thư Mục Dự Án

Cấu trúc dự án được chuẩn hóa tinh gọn, tách biệt rõ ràng giữa mã nguồn huấn luyện, module kiểm thử, tài nguyên demo và tài liệu kỹ thuật:

```text
ComputerVision_ArcNet/
├── src/                               # Thư viện lõi huấn luyện & mô hình
│   ├── model.py                      # Định nghĩa IResNet-50 & ArcMarginProduct
│   ├── recordio.py                   # Pure Python RecordIO reader (Zero MXNet)
│   ├── dataset.py                    # DataLoader nạp đa luồng & Augmentations
│   ├── train.py                      # Pipeline huấn luyện, Warmup, AMP, Optimizer
│   ├── eval.py                       # Thuật toán đánh giá Accuracy & 10-fold CV
│   ├── ema.py                        # Exponential Moving Average cho weights
│   ├── config.py                     # Quản lý siêu tham số hệ thống
│   ├── inspect_dataset.py            # Công cụ chẩn đoán và trích xuất metadata .rec
│   └── utils.py                      # Tiện ích giải mã các tệp kiểm chuẩn .bin
├── tools/                             # Bộ công cụ phụ trợ (Scripts & Utilities)
│   ├── export_onnx.py                # Chuyển đổi mô hình PyTorch sang định dạng ONNX
│   ├── generate_report.py            # Tự động xuất biểu đồ và báo cáo chỉ số
│   ├── evaluate_results.py           # Quét và đánh giá hàng loạt checkpoint
│   ├── test_speed.py                 # Benchmark tốc độ DataLoader & GPU forward/backward
│   └── README.md                     # Tài liệu hướng dẫn sử dụng các tools
├── fine_tune/                         # Pipeline tinh chỉnh (Transfer Learning)
│   ├── config_finetune.py            # Cấu hình siêu tham số fine-tuning
│   ├── dataset_finetune.py           # Dataset loader cho tập fine-tune
│   ├── train_finetune.py             # Script thực thi quá trình fine-tuning
│   └── README.md                     # Hướng dẫn quy trình fine-tuning
├── robustness_testing/                # Bộ công cụ Stress Testing chuyên sâu
│   ├── perturbation_tests.py         # Kiểm thử nhiễu, mờ, độ sáng, góc xoay
│   ├── speed_stress_test.py          # Benchmark độ trễ, FPS và ép tải đa luồng
│   ├── lookalike_twins_test.py       # Kiểm thử các cặp sinh đôi và người giống nhau
│   ├── custom_test_images/           # Ảnh mẫu phục vụ thử nghiệm ngoại cảnh
│   └── README.md                     # Tài liệu hướng dẫn bộ công cụ test
├── Web/                               # Ứng dụng Web Dashboard & Realtime Demo
│   ├── index.html                    # Trang điều khiển chính ArcNet
│   ├── live_demo.html                # So sánh đối sánh 2 khuôn mặt thời gian thực
│   ├── webcam_realtime.html          # Nhận diện trực tiếp qua Webcam (WebAssembly)
│   ├── robustness_evaluation.html    # Báo cáo trực quan hóa độ bền vững
│   ├── training_process.html         # Trực quan hóa tiến trình huấn luyện mô hình
│   ├── sota_techniques.html          # So sánh các kỹ thuật SOTA trong Deep Face
│   ├── app.js                        # Logic xử lý client, ONNX Runtime Web & Canvas
│   ├── style.css                     # Giao diện responsive hiện đại
│   ├── roc_curves.png                # Biểu đồ đường cong ROC
│   └── similarity_distributions.png  # Phân phối độ tương đồng Cosine
├── information/                       # Báo cáo kỹ thuật phân tích chuyên sâu
│   ├── model_summary.md              # Thống kê chi tiết các tầng mạng và trọng số
│   └── training_curve.png            # Biểu đồ đường cong hội tụ Loss & Accuracy
├── main.py                            # Điểm nhập lệnh duy nhất (Train, Eval, Export)
├── run_train.bat                      # Script chạy huấn luyện 1-click trên Windows
├── requirements.txt                   # Danh sách gói phụ thuộc môi trường Python
├── .gitignore                         # Loại trừ dataset, checkpoint nặng và logs
└── README.md                          # Tài liệu hướng dẫn tổng quan dự án
```

---

## 🛠️ Cài Đặt Môi Trường

### 1. Clone Kho Lưu Trữ
```bash
git clone https://github.com/MaiVanTruong2310/ComputerVision_ArcNet.git
cd ComputerVision_ArcNet
```

### 2. Tạo Môi Trường Ảo & Cài Đặt Thư Viện
Khuyến nghị sử dụng **Python 3.10** hoặc **3.11**:
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📦 Chuẩn Bị Dữ Liệu

Tạo thư mục `dataset/` (hoặc `dataset/raw/`) và đặt dữ liệu vào vị trí tương ứng:
```text
dataset/
├── train.rec                         # File RecordIO chứa toàn bộ ảnh huấn luyện
├── train.idx                         # File Index đánh dấu chỉ mục
├── train.lst                         # Danh sách nhãn và định danh
├── lfw.bin                           # Tập kiểm chuẩn LFW
├── agedb_30.bin                      # Tập kiểm chuẩn AgeDB-30
└── cfp_fp.bin                        # Tập kiểm chuẩn CFP-FP
```

Kiểm tra tính toàn vẹn và số lượng danh tính (classes) của dataset:
```powershell
python src/inspect_dataset.py
```

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Huấn Luyện Mô Hình (Training)
Chạy trực tiếp qua lệnh CLI:
```powershell
python main.py --mode train
```
Hoặc trên Windows, bạn chỉ cần nhấp đúp chuột vào file `run_train.bat`.

### 2. Đánh Giá Checkpoint (Evaluation)
Đánh giá độ chính xác của một checkpoint cụ thể trên tập kiểm chuẩn LFW:
```powershell
python main.py --mode eval --backbone checkpoints/finetuned_best.pth --target lfw
```

Hoặc đánh giá toàn diện và tự động xuất báo cáo, biểu đồ:
```powershell
python tools/generate_report.py --checkpoint checkpoints/finetuned_best.pth --target lfw
```

### 3. Xuất Mô Hình Sang Định Dạng ONNX
Để chuyển đổi checkpoint sang định dạng ONNX phục vụ triển khai Web hoặc C++ Embedded:
```powershell
python tools/export_onnx.py --checkpoint checkpoints/finetuned_best.pth --output outputs/finetuned_best.onnx
```
*(Hoặc dùng lệnh ngắn gọn: `python main.py --mode export --backbone checkpoints/finetuned_best.pth`)*

### 4. Tinh Chỉnh Mô Hình (Fine-tuning)
Để huấn luyện tiếp mô hình trên tập dữ liệu đặc thù của riêng bạn:
```powershell
cd fine_tune
python train_finetune.py
```

---

## 🌐 Web Dashboard & Realtime Demo

Hệ thống đi kèm một giao diện Web tương tác hoàn chỉnh sử dụng công nghệ **ONNX Runtime Web (WASM)**, cho phép suy luận trực tiếp trên trình duyệt mà không cần máy chủ GPU:

1. Đặt file ONNX đã xuất vào thư mục `Web/models/finetuned_best.onnx`.
2. Khởi chạy HTTP Server cục bộ:
   ```powershell
   python -m http.server 8000 --directory Web
   ```
3. Mở trình duyệt và truy cập: `http://localhost:8000`
   - **Live Face Verification**: Kéo thả 2 bức ảnh bất kỳ để tính toán Cosine Similarity và xác định cùng người / khác người.
   - **Realtime Webcam**: Trích xuất khuôn mặt qua camera theo thời gian thực và tự động nhận diện danh tính đã đăng ký.
   - **Visual Analytics**: Xem biểu đồ ROC, biểu đồ phân phối khoảng cách và độ bền vững.

---

## 🛡️ Kiểm Thử Độ Bền Vững (Robustness Suite)

Kho lưu trữ cung cấp đầy đủ các kịch bản kiểm thử bảo mật và áp lực trong thư mục `robustness_testing/`:
- **Biến dạng ảnh (Perturbations)**: Đo lường tác động của nhiễu Gaussian, mờ chuyển động, thay đổi độ sáng và góc xoay nghiêng:
  ```powershell
  python robustness_testing/perturbation_tests.py
  ```
- **Ép tải & Đa luồng (Stress & Concurrency)**: Kiểm tra FPS, thông lượng và khả năng chịu tải đa luồng:
  ```powershell
  python robustness_testing/speed_stress_test.py
  ```
- **Phân biệt sinh đôi / Người giống nhau (Lookalike Twins)**: Kiểm tra các ca khó (Hard Negatives):
  ```powershell
  python robustness_testing/lookalike_twins_test.py
  ```

---

## 📄 Bản Quyền & Trích Dẫn (License)

Dự án được phân phối dưới giấy phép **MIT License**. Mọi đóng góp, báo cáo lỗi (Issues) hoặc Pull Requests đều được chào đón nồng nhiệt tại [GitHub Repository](https://github.com/MaiVanTruong2310/ComputerVision_ArcNet.git).
