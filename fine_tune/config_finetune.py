import os
import torch

class FinetuneConfig:
    # ─── Dataset Paths ────────────────────────────────────────────────────────
    dataset_path = '../dataset/raw' # Trỏ về dataset .rec gốc hoặc thư mục chứa file .rec mới
    output_dir = 'outputs'
    checkpoint_dir = 'checkpoints'
    log_dir = 'logs'
    
    # Số lượng class (Sẽ được tự động đếm qua file .lst hoặc quét file .rec)
    num_classes = 10572
    # ─── Model Settings ───────────────────────────────────────────────────────
    # Phải trùng khớp với pre-trained model
    backbone_name = 'iresnet50' 
    embedding_size = 512
    dropout = 0.4
    pretrained_path = '../checkpoints/backbone_best.pth'
    
    # ─── Fine-tune Settings ───────────────────────────────────────────────────
    freeze_backbone = False # Bật True nếu data quá ít (dưới 50 ảnh/người)
    load_pretrained = True  # Bật True để fine-tune từ backbone pre-trained tốt nhất
    subsample_ratio = 0.2   # Chạy trên 2/10 tập dữ liệu ảo (2.114 classes)
    grad_accumulate_steps = 4 # Tích lũy gradient (giả lập batch size 256 từ batch size 64)
    
    # ─── ArcFace Settings ─────────────────────────────────────────────────────
    s = 64.0
    m = 0.5   # Đặt margin là 0.5 để siết chặt biên giới phân loại cho đầu phân loại mới
    
    # ─── Training Hyperparameters ─────────────────────────────────────────────
    batch_size = 64   # Hạ batch size xuống 64 để phù hợp với RTX 3050 (6GB VRAM)
    epochs = 25       # Chạy 25 epochs
    
    # Learning rate chuẩn (SGD sẽ được cập nhật)
    base_lr = 0.003   # Hạ LR xuống 0.003 cho fine-tune để đảm bảo an toàn cho đặc trưng của backbone
    momentum = 0.9
    weight_decay = 5e-4
    ema_decay = 0.999
    
    # Learning rate schedule
    warmup_epochs = 3
    margin_warmup_epochs = 5
    warmup_init_lr = 1e-4
    lr_milestones = [] # Có thể dùng thay cho Cosine
    lr_gamma = 0.1
    
    # ─── System Settings ──────────────────────────────────────────────────────
    num_workers = 4   # Tối ưu luồng CPU cho HĐH Windows
    pin_memory = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_grad_norm = 5.0
    
    # ─── Validation Settings ──────────────────────────────────────────────────
    val_targets = ['lfw', 'agedb_30', 'cfp_fp']
    val_batch_size = 256
    
    @classmethod
    def create_dirs(cls):
        for d in [cls.output_dir, cls.checkpoint_dir, cls.log_dir]:
            os.makedirs(d, exist_ok=True)

FinetuneConfig.create_dirs()
