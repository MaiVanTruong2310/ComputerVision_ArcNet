# ==============================================================================
# Tên file: train.py
# Công dụng: Kịch bản huấn luyện chính (Main Training Pipeline) của mô hình ArcFace.
#            Tích hợp các kỹ thuật hiện đại: Khởi động mềm LR/Margin (Warmup),
#            tính toán dấu phẩy động hỗn hợp tự động (AMP FP16), cắt đạo hàm (Gradient Clipping),
#            trung bình động lũy thừa trọng số (EMA), và lưu trữ checkpoint phục hồi.
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from config import Config
from ema import EMA
from dataset import get_dataloader
from model import get_backbone, ArcMarginProduct
from utils import load_bin
from eval import perform_val
import os
import time
from tqdm import tqdm
from datetime import timedelta

def train():
    # ── Device & Config ─────────────────────────────────────────────────────
    device = Config.device
    print(f"--- Device: {device} | GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'} ---")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True # Tự động tìm thuật toán tích chập tối ưu nhất cho ảnh kích cỡ cố định
        # Tối ưu cho GPU Ampere (RTX 30 series, RTX 40 series)
        torch.backends.cuda.matmul.allow_tf32 = True # Cho phép dùng định dạng TF32 cho phép nhân ma trận trên GPU Ampere trở lên
        torch.backends.cudnn.allow_tf32 = True # Cho phép dùng định dạng TF32 cho phép tích chập cuDNN để tăng tốc độ train
        # => Giúp tăng tốc độ huấn luyện và rất hữu ích khi sử dụng các GPU hiện đại

    train_loader = get_dataloader( # Khởi tạo Dataloader để load dữ liệu theo batch và hỗ trợ đa luồng (multi-processing)
        Config.dataset_path,
        Config.batch_size,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory
    )
    steps_per_epoch = len(train_loader)
    warmup_steps    = Config.warmup_epochs * steps_per_epoch
    margin_warmup_steps = Config.margin_warmup_epochs * steps_per_epoch
    print(f"--- Training: {Config.epochs} epochs | LR Warmup: {Config.warmup_epochs} | Margin Warmup: {Config.margin_warmup_epochs} ---")
    print(f"--- LR: {Config.base_lr} (init: {Config.warmup_init_lr}) | Margin: {Config.m} ---")

    # Load validation data (LFW, AgeDB-30, CFP-FP)
    val_data_list = []
    for target in Config.val_targets:
        bin_path = os.path.join(Config.dataset_path, 'raw', f'{target}.bin')
        if not os.path.exists(bin_path):
            bin_path = os.path.join(Config.dataset_path, f'{target}.bin')
            
        if os.path.exists(bin_path):
            print(f"Loading validation set: {target}")
            data, issame = load_bin(bin_path)
            val_data_list.append((target, data, issame))

    # ── Model ────────────────────────────────────────────────────────────────
    backbone = get_backbone( # Trích xuất đặc trưng khuôn mặt (IResNet)
        name=Config.backbone_name, 
        embedding_size=Config.embedding_size,
        dropout=Config.dropout
    ).to(device)
    header = ArcMarginProduct(Config.embedding_size, Config.num_classes, s=Config.s, m=Config.m).to(device) # Đầu phân loại ArcFace tạo biên độ góc m
    # Initialize EMA
    ema = EMA(backbone, decay=Config.ema_decay) # Khởi tạo bộ trung bình động trọng số giúp tăng tính tổng quát hóa

    # ── Optimizer & Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.SGD( # Tối ưu hóa bằng thuật toán SGD với Momentum và giảm trọng số Weight Decay
        [{'params': backbone.parameters()}, {'params': header.parameters()}],
        lr=Config.base_lr, momentum=Config.momentum, weight_decay=Config.weight_decay
    )

    # Scheduler: CosineAnnealing after warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( # Điều chỉnh LR giảm dần theo đường cong Cosine sau thời gian Warmup
        optimizer, T_max=(Config.epochs - Config.warmup_epochs), eta_min=1e-5
    )

    # ── Loss & Scaler ────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss() # Sử dụng CrossEntropyLoss truyền thống kết hợp với ArcMargin Logits
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None # Bộ điều chỉnh tỷ lệ gradient để chống lỗi underflow của kiểu FP16

    # ── TensorBoard ──────────────────────────────────────────────────────────
    writer = SummaryWriter(Config.log_dir)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    best_lfw_acc = 0.0
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    resume_path = os.path.join(Config.checkpoint_dir, 'latest.pth')
    if os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        backbone.load_state_dict(ckpt['backbone'])
        header.load_state_dict(ckpt['header'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if 'ema' in ckpt:
            ema.load_state_dict(ckpt['ema'])
        start_epoch = ckpt['epoch'] + 1
        best_lfw_acc = ckpt.get('best_lfw_acc', 0.0)
        print(f"Resumed at epoch {start_epoch}, best LFW acc: {best_lfw_acc:.4f}")

    # ── Training Loop ────────────────────────────────────────────────────────
    global_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, Config.epochs):
        backbone.train()
        header.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_clean_acc = 0.0 
        start_time = time.time()
        
        pbar = tqdm(enumerate(train_loader), total=steps_per_epoch, desc=f"Epoch {epoch:02d}/{Config.epochs}")

        for i, (imgs, labels) in pbar:
            global_step += 1
            if global_step <= warmup_steps:
                # Linear LR Warmup: from warmup_init_lr to base_lr
                alpha = global_step / max(1, warmup_steps)
                cur_lr = Config.warmup_init_lr + (Config.base_lr - Config.warmup_init_lr) * alpha # Tăng dần LR từ giá trị nhỏ đến base_lr
                for pg in optimizer.param_groups:
                    pg['lr'] = cur_lr # Cập nhật tốc độ học mới cho optimizer
            
            # Margin Warmup (Independent of LR Warmup)
            if global_step <= margin_warmup_steps:
                alpha_m = global_step / max(1, margin_warmup_steps)
                cur_m = Config.m * alpha_m # Tăng dần biên độ góc m từ 0 để chống lỗi tràn số NaN ở đầu buổi train
                header.update_margin(cur_m) # Cập nhật biên độ m hiện tại vào ArcMarginProduct
            else:
                header.update_margin(Config.m) # Sau thời gian warmup, duy trì biên độ m tối đa cố định
                
            if global_step % 10 == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                cur_m = header.m
                writer.add_scalar('Train/LR', cur_lr, global_step)
                writer.add_scalar('Train/Margin', cur_m, global_step)

            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if scaler:
                with torch.amp.autocast('cuda'): # Kích hoạt chế độ tính toán dấu phẩy động hỗn hợp tự động (AMP FP16)
                    features = backbone(imgs) # Trích xuất vector đặc trưng 512 chiều từ ảnh khuôn mặt
                    thetas   = header(features, labels) # Áp dụng biên độ phạt góc ArcFace cho lớp nhãn thực tế
                    loss     = criterion(thetas, labels) # Tính toán giá trị loss phân loại
                scaler.scale(loss).backward() # Nhân tỷ lệ loss và thực hiện lan truyền ngược (backward pass)
                scaler.unscale_(optimizer) # Chuyển đạo hàm về tỷ lệ gốc trước khi thực hiện gradient clipping
                torch.nn.utils.clip_grad_norm_( # Giới hạn chuẩn gradient tối đa để tránh hiện tượng nổ đạo hàm (Gradient Explosion)
                    list(backbone.parameters()) + list(header.parameters()),
                    Config.max_grad_norm
                )
                scaler.step(optimizer) # Cập nhật các trọng số mô hình thông qua optimizer sử dụng đạo hàm đã điều chỉnh
                scaler.update() # Cập nhật hệ số phóng đại của scaler cho các batch tiếp theo
            else:
                features = backbone(imgs)
                thetas   = header(features, labels)
                loss     = criterion(thetas, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(header.parameters()),
                    Config.max_grad_norm
                )
                optimizer.step()
            
            # Update EMA weights
            ema.update() # Cập nhật trọng số trung bình động lũy thừa (EMA) dựa trên trọng số mô hình hiện tại

            # Tính toán Accuracy tập train
            with torch.no_grad():
                # Margin Accuracy (Logits with margin penalty)
                preds = torch.argmax(thetas, dim=1) # Tìm lớp dự đoán có xác suất cao nhất (đã cộng biên độ m)
                acc = (preds == labels).float().mean().item() # Tính độ chính xác có kèm biên phạt ArcFace
                
                # Clean Accuracy (Logits WITHOUT margin penalty)
                # Ép kiểu features về float() để tránh lỗi Half (FP16) vs Float (FP32) khi dùng AMP
                cosine = F.linear(F.normalize(features.float()), F.normalize(header.weight.float())) # Tính độ tương đồng cosine sạch giữa ảnh và lớp
                clean_preds = torch.argmax(cosine, dim=1) # Tìm lớp dự đoán sạch không bị phạt biên độ m
                clean_acc = (clean_preds == labels).float().mean().item() # Tính độ chính xác sạch (thường cao hơn MarginAcc)
            
            total_loss += loss.item()
            total_accuracy += acc
            total_clean_acc += clean_acc

            # Cập nhật thông tin lên thanh progress bar
            if i % 10 == 0:
                elapsed = time.time() - start_time
                avg_loss = total_loss / (i + 1)
                avg_acc  = total_accuracy / (i + 1)
                avg_clean_acc = total_clean_acc / (i + 1)
                cur_lr   = optimizer.param_groups[0]['lr']
                
                # Ước tính thời gian còn lại (ETA)
                eta_epoch = elapsed / (i + 1) * (steps_per_epoch - (i + 1))
                eta_str = str(timedelta(seconds=int(eta_epoch)))
                
                pbar.set_postfix({
                    "Loss": f"{avg_loss:.4f}",
                    "CleanAcc": f"{avg_clean_acc:.4f}",
                    "MarginAcc": f"{avg_acc:.4f}",
                    "LR": f"{cur_lr:.7f}",
                    "M": f"{header.m:.3f}",
                    "ETA": eta_str
                })
        if epoch >= Config.warmup_epochs:
            scheduler.step()

        avg_loss = total_loss / steps_per_epoch
        print(f"--- Epoch {epoch:02d} done | Avg Loss: {avg_loss:.4f} ---")
        writer.add_scalar('Epoch/Loss', avg_loss, epoch)

        # ── Validation (Sử dụng EMA weights để đạt kết quả cao nhất) ───────────
        ema.apply_shadow() # Tạm thời nạp trọng số EMA mượt mà vào mô hình để thực hiện kiểm thử
        epoch_lfw_acc = 0.0
        for name, data, issame in val_data_list:
            acc, std, thr = perform_val(backbone, device, data, issame) # Chạy kiểm thử trên tập dữ liệu tương ứng (.bin)
            print(f"--- [VAL] {name.upper()} | Acc: {acc:.4f} ± {std:.4f} | Threshold: {thr:.2f} ---")
            writer.add_scalar(f'Val/{name.upper()}_Acc', acc, epoch)
            if name == 'lfw':
                epoch_lfw_acc = acc

        # ── Save Checkpoints ─────────────────────────────────────────────────
        ckpt = {
            'epoch': epoch,
            'backbone': backbone.state_dict(),
            'ema': ema.state_dict(),
            'header': header.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_lfw_acc': max(best_lfw_acc, epoch_lfw_acc),
        }
        torch.save(ckpt, os.path.join(Config.checkpoint_dir, 'latest.pth'))
        torch.save(backbone.state_dict(), os.path.join(Config.checkpoint_dir, f'backbone_epoch_{epoch:02d}.pth'))

        if epoch_lfw_acc > best_lfw_acc:
            best_lfw_acc = epoch_lfw_acc
            torch.save(backbone.state_dict(), os.path.join(Config.checkpoint_dir, 'backbone_best.pth')) # Lưu riêng trọng số backbone tốt nhất
            print(f"--- New best LFW accuracy: {best_lfw_acc:.4f}! Saved backbone_best.pth ---")
        
        # Trả lại trọng số gốc để tiếp tục huấn luyện
        ema.restore() # Khôi phục lại trọng số huấn luyện ban đầu của mô hình để tiếp tục tối ưu hóa ở epoch sau

    writer.close()
    print(f"Training complete. Best LFW accuracy: {best_lfw_acc:.4f}")


if __name__ == "__main__":
    train()