# Hướng dẫn Fine-tune Mô hình Facenet

Thư mục này chứa code để bạn tự đưa dữ liệu khuôn mặt cá nhân/công ty vào huấn luyện tiếp (fine-tune) từ mô hình đã được pre-train.

## 1. Chuẩn bị dữ liệu
Theo yêu cầu, mô hình sẽ tiếp tục sử dụng định dạng file `.rec` gốc. 
Bạn hãy chắc chắn rằng đường dẫn `dataset_path` trong `config_finetune.py` đang trỏ đúng đến thư mục chứa file `.rec` và `.lst` (mặc định là `../dataset`).

## 2. Cấu hình Fine-tune
Mở file `fine_tune/config_finetune.py` để chỉnh sửa:
- `num_classes = 10572`: Nếu bạn dùng tập dataset cũ, giữ nguyên số này. Nếu bạn dùng một tập `.rec` mới, bạn phải thay đổi số này cho phù hợp với số lượng classes của tập mới.
- `freeze_backbone = False`: Mặc định là `False` để mô hình cập nhật lại toàn bộ weights với learning rate nhỏ.
- `epochs = 20`: Số vòng lặp huấn luyện.
- `batch_size = 64`: Có thể tăng/giảm tùy lượng RAM của GPU.

## 3. Chạy Fine-tune
Mở Terminal, trỏ đường dẫn về thư mục `fine_tune` và chạy lệnh:
```bash
cd fine_tune
python train_finetune.py
```

## 4. Kết quả
Sau khi train xong, mô hình best sẽ được lưu tại:
- `fine_tune/checkpoints/finetuned_best.pth` (Dùng file này để deploy vào ứng dụng nhận diện của bạn).
