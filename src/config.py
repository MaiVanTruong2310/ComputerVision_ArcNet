# ==============================================================================
# Tên file: config.py
# Công dụng: Định nghĩa lớp Config chứa toàn bộ siêu tham số (Hyperparameters)
#            và cấu hình hệ thống cho huấn luyện và kiểm thử.
#            Quản lý đường dẫn dữ liệu, thông số mạng backbone, các thông số ArcFace (s, m),
#            tốc độ học (learning rate), số epoch, cấu hình GPU, số luồng DataLoader.
# ==============================================================================
import os
import torch

class Config:
    # ─── Dataset Paths ────────────────────────────────────────────────────────
    dataset_path = 'dataset'
    output_dir = 'outputs'
    checkpoint_dir = 'checkpoints'
    log_dir = 'logs'
    num_classes = 10572
    
    # ─── Model Settings ───────────────────────────────────────────────────────
    backbone_name = 'iresnet50'
    embedding_size = 512
    dropout = 0.4
    
    # ─── ArcFace Settings ─────────────────────────────────────────────────────
    s = 64.0
    m = 0.4
    
    # ─── Training Hyperparameters ─────────────────────────────────────────────
    batch_size = 64 
    epochs = 25
    base_lr = 0.03
    momentum = 0.9
    weight_decay = 5e-4 # Cơ chế Regularization (Cơ chế Điều hòa hóa)
    ema_decay = 0.999
    
    # Learning rate schedule
    warmup_epochs = 3
    margin_warmup_epochs = 5
    warmup_init_lr = 1e-4
    lr_milestones = [15, 24, 28]
    lr_gamma = 0.1
    
    # ─── System Settings ──────────────────────────────────────────────────────
    num_workers = 8
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

# Khởi tạo các thư mục cần thiết
Config.create_dirs()