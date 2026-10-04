import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json
import csv
import builtins
import io

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_backbone, ArcMarginProduct
from src.ema import EMA
from src.eval import perform_val
from src.utils import load_bin
from config_finetune import FinetuneConfig as Config
from dataset_finetune import get_finetune_dataloader

def make_print_logger(log_path):
    original_print = builtins.print
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    def logger_print(*args, **kwargs):
        # Print to console
        original_print(*args, **kwargs)
        
        # Check if printing to file/stream other than stdout/stderr
        file_arg = kwargs.get('file', None)
        if file_arg is not None and file_arg is not sys.stdout and file_arg is not sys.stderr:
            return
            
        # Format the message
        f = io.StringIO()
        kwargs_copy = kwargs.copy()
        kwargs_copy['file'] = f
        original_print(*args, **kwargs_copy)
        message = f.getvalue()
        
        with open(log_path, 'a', encoding='utf-8') as lf:
            lf.write(message)
            
    builtins.print = logger_print

def log_epoch_metrics(csv_path, epoch, loss, train_acc, train_clean_acc, lr, margin, val_accs):
    file_exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    headers = ['epoch', 'loss', 'train_acc', 'train_clean_acc', 'lr', 'margin']
    for name in val_accs.keys():
        col_name = f'val_{name}'
        if col_name not in headers:
            headers.append(col_name)
            
    row = {
        'epoch': epoch,
        'loss': loss,
        'train_acc': train_acc,
        'train_clean_acc': train_clean_acc,
        'lr': lr,
        'margin': margin
    }
    for name, acc in val_accs.items():
        row[f'val_{name}'] = acc
        
    if file_exists:
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                existing_headers = next(reader)
            # Use existing headers if they exist
            for h in headers:
                if h not in existing_headers:
                    existing_headers.append(h)
            headers = existing_headers
        except Exception:
            pass
            
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def get_parameter_groups(backbone, header, weight_decay, freeze_backbone=False):
    backbone_decay = []
    backbone_no_decay = []
    header_decay = []
    header_no_decay = []
    
    # Process backbone
    if not freeze_backbone:
        for m in backbone.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                for p in m.parameters():
                    if p.requires_grad:
                        backbone_no_decay.append(p)
            else:
                for name, p in m.named_parameters(recurse=False):
                    if not p.requires_grad:
                        continue
                    if 'bias' in name:
                        backbone_no_decay.append(p)
                    else:
                        backbone_decay.append(p)
                        
    # Process header
    for m in header.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            for p in m.parameters():
                if p.requires_grad:
                    header_no_decay.append(p)
        else:
            for name, p in m.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                if 'bias' in name:
                    header_no_decay.append(p)
                else:
                    header_decay.append(p)
                    
    # Chia nhỏ LR: Backbone dùng hệ số 0.1 (tinh chỉnh khẽ), Classifier Head dùng hệ số 1.0 (học mạnh)
    groups = []
    if not freeze_backbone:
        groups.append({'params': backbone_decay, 'weight_decay': weight_decay, 'lr_mult': 0.1})
        groups.append({'params': backbone_no_decay, 'weight_decay': 0.0, 'lr_mult': 0.1})
    groups.append({'params': header_decay, 'weight_decay': weight_decay, 'lr_mult': 1.0})
    groups.append({'params': header_no_decay, 'weight_decay': 0.0, 'lr_mult': 1.0})
    
    return groups

def train_finetune():
    # Khởi tạo thư mục và tệp ghi nhận kết quả (nhật ký)
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    log_file = os.path.join(Config.checkpoint_dir, 'finetune_log.txt')
    csv_file = os.path.join(Config.checkpoint_dir, 'finetune_metrics.csv')
    
    # Chỉ xoá log cũ nếu huấn luyện từ đầu và không có checkpoint cũ để chạy tiếp
    latest_ckpt_path = os.path.join(Config.checkpoint_dir, 'latest.pth')
    if not os.path.exists(latest_ckpt_path):
        if os.path.exists(log_file):
            try:
                os.remove(log_file)
            except Exception:
                pass
        if os.path.exists(csv_file):
            try:
                os.remove(csv_file)
            except Exception:
                pass
                
    make_print_logger(log_file)

    device = Config.device
    print(f"--- Bắt đầu chế độ FINE-TUNE trên {device} ---")
    
    # 1. Khởi tạo Dataloader
    dataset_path = Config.dataset_path
    if os.path.isdir(dataset_path):
        rec_path = os.path.join(dataset_path, 'train.rec')
    else:
        rec_path = dataset_path
        
    if not os.path.exists(rec_path):
        print(f"Không tìm thấy file {rec_path}.")
        return
        
    train_loader, num_classes = get_finetune_dataloader(
        rec_path,
        Config.batch_size,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
        subsample_ratio=Config.subsample_ratio
    )
    
    if num_classes == 0:
        print("Không tìm thấy class nào. Vui lòng kiểm tra lại dataset.")
        return
        
    Config.num_classes = num_classes
    
    # Load validation data
    val_targets = getattr(Config, 'val_targets', ['lfw', 'agedb_30', 'cfp_fp'])
    val_data_list = []
    for target in val_targets:
        bin_path = os.path.join(dataset_path, f'{target}.bin')
        if not os.path.exists(bin_path):
            bin_path = os.path.join(dataset_path, 'raw', f'{target}.bin')
            
        if os.path.exists(bin_path):
            print(f"Loading validation set: {target}")
            data, issame = load_bin(bin_path)
            val_data_list.append((target, data, issame))
    
    # 2. Khởi tạo Backbone
    backbone = get_backbone(
        name=Config.backbone_name, 
        embedding_size=Config.embedding_size,
        dropout=Config.dropout
    ).to(device)
    
    # 3. Khởi tạo Header mới (ArcMarginProduct)
    header = ArcMarginProduct(Config.embedding_size, Config.num_classes, s=Config.s, m=Config.m).to(device)
    
    # 4. Khởi tạo Optimizer, Scheduler
    if Config.freeze_backbone:
        print("Đang đóng băng (freeze) các trọng số của backbone...")
        for param in backbone.parameters():
            param.requires_grad = False

    # Phân tách weight decay và cấu hình learning rate riêng cho từng nhóm (Backbone 10x nhỏ hơn)
    param_groups = get_parameter_groups(backbone, header, Config.weight_decay, Config.freeze_backbone)
    for pg in param_groups:
        pg['lr'] = Config.base_lr * pg.get('lr_mult', 1.0)
        
    optimizer = torch.optim.SGD(
        param_groups,
        lr=Config.base_lr, momentum=Config.momentum
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(Config.epochs - Config.warmup_epochs), eta_min=1e-5
    )
    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # Khởi tạo các trạng thái huấn luyện
    start_epoch = 0
    best_lfw_acc = 0.0
    checkpoint_loaded = False

    # Nạp checkpoint nếu có sẵn
    if os.path.exists(latest_ckpt_path):
        print(f"Phát hiện checkpoint cũ tại {latest_ckpt_path}. Đang nạp để chạy tiếp...")
        try:
            ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
            start_epoch = ckpt['epoch'] + 1
            backbone.load_state_dict(ckpt['backbone'])
            header.load_state_dict(ckpt['header'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            best_lfw_acc = ckpt.get('best_lfw_acc', 0.0)
            
            # Khởi tạo EMA và nạp shadow weights từ checkpoint
            ema = EMA(backbone, decay=Config.ema_decay)
            ema.load_state_dict(ckpt['ema'])
            checkpoint_loaded = True
            
            print(f"Nạp checkpoint thành công! Tiếp tục từ Epoch {start_epoch}, LFW acc tốt nhất: {best_lfw_acc:.4f}")
        except Exception as e:
            print(f"Không thể nạp checkpoint ({e}). Sẽ nạp pre-trained hoặc train từ đầu.")
            
    # Nếu không nạp checkpoint thành công và cần nạp pre-trained
    if not checkpoint_loaded:
        if Config.load_pretrained:
            if os.path.exists(Config.pretrained_path):
                print(f"Loading pretrained weights từ {Config.pretrained_path}...")
                try:
                    pretrained_dict = torch.load(Config.pretrained_path, map_location=device, weights_only=False)
                    if 'backbone' in pretrained_dict:
                        pretrained_dict = pretrained_dict['backbone']
                    backbone.load_state_dict(pretrained_dict)
                except Exception as e:
                    print(f"Lỗi nạp pretrained ({e}). Sẽ train từ đầu.")
            else:
                print(f"CẢNH BÁO: Không tìm thấy file pretrained tại {Config.pretrained_path}!")
        else:
            print("Huấn luyện từ đầu (from scratch) trên 2/10 dữ liệu...")
            
        # Khởi tạo EMA sau khi backbone đã nạp xong trọng số pretrained hoặc ngẫu nhiên
        ema = EMA(backbone, decay=Config.ema_decay)

    # 5. Vòng lặp Training
    steps_per_epoch = len(train_loader)
    warmup_steps = Config.warmup_epochs * steps_per_epoch
    margin_warmup_steps = Config.margin_warmup_epochs * steps_per_epoch
    global_step = start_epoch * steps_per_epoch
    
    for epoch in range(start_epoch, Config.epochs):
        backbone.train()
        header.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_clean_acc = 0.0
        
        pbar = tqdm(enumerate(train_loader), total=steps_per_epoch, desc=f"Epoch {epoch:02d}/{Config.epochs}")
        
        optimizer.zero_grad(set_to_none=True)
        
        for i, (imgs, labels) in pbar:
            global_step += 1
            if global_step <= warmup_steps:
                alpha = global_step / max(1, warmup_steps)
                cur_lr = Config.warmup_init_lr + (Config.base_lr - Config.warmup_init_lr) * alpha
                for pg in optimizer.param_groups:
                    pg['lr'] = cur_lr * pg.get('lr_mult', 1.0)
            
            if global_step <= margin_warmup_steps:
                alpha_m = global_step / max(1, margin_warmup_steps)
                cur_m = Config.m * alpha_m
                header.update_margin(cur_m)
            else:
                header.update_margin(Config.m)
                
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Tích lũy gradient
            is_update_step = ((i + 1) % Config.grad_accumulate_steps == 0) or (i + 1 == steps_per_epoch)
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    features = backbone(imgs)
                    thetas = header(features, labels)
                    loss = criterion(thetas, labels)
                    loss_scaled = loss / Config.grad_accumulate_steps
                
                scaler.scale(loss_scaled).backward()
                
                if is_update_step:
                    scaler.unscale_(optimizer)
                    # Chỉ clip grad cho những param requires_grad
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in backbone.parameters() if p.requires_grad) if not Config.freeze_backbone else [],
                        Config.max_grad_norm
                    )
                    torch.nn.utils.clip_grad_norm_(header.parameters(), Config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                features = backbone(imgs)
                thetas = header(features, labels)
                loss = criterion(thetas, labels)
                loss_scaled = loss / Config.grad_accumulate_steps
                
                loss_scaled.backward()
                
                if is_update_step:
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in backbone.parameters() if p.requires_grad) if not Config.freeze_backbone else [],
                        Config.max_grad_norm
                    )
                    torch.nn.utils.clip_grad_norm_(header.parameters(), Config.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                
            # Cập nhật EMA chỉ ở các bước cập nhật trọng số
            if not Config.freeze_backbone and is_update_step:
                ema.update()
                
            with torch.no_grad():
                preds = torch.argmax(thetas, dim=1)
                acc = (preds == labels).float().mean().item()
                
                cosine = F.linear(F.normalize(features.float()), F.normalize(header.weight.float()))
                clean_preds = torch.argmax(cosine, dim=1)
                clean_acc = (clean_preds == labels).float().mean().item()
                
            total_loss += loss.item()
            total_accuracy += acc
            total_clean_acc += clean_acc
            
            if i % 5 == 0:
                avg_loss = total_loss / (i + 1)
                avg_acc = total_accuracy / (i + 1)
                avg_clean_acc = total_clean_acc / (i + 1)
                pbar.set_postfix({
                    "Loss": f"{avg_loss:.4f}",
                    "CleanAcc": f"{avg_clean_acc:.4f}",
                    "MarginAcc": f"{avg_acc:.4f}",
                    "LR": f"{optimizer.param_groups[0]['lr']:.6f}",
                    "M": f"{header.m:.3f}"
                })
                
        if epoch >= Config.warmup_epochs:
            scheduler.step()
            
        avg_loss_epoch = total_loss / steps_per_epoch
        avg_acc_epoch = total_accuracy / steps_per_epoch
        avg_clean_acc_epoch = total_clean_acc / steps_per_epoch
        
        print(f"--- Epoch {epoch:02d} done | Avg Loss: {avg_loss_epoch:.4f} ---")
        
        # Validation & Checkpoints
        if not Config.freeze_backbone:
            ema.apply_shadow()
            
        val_accs = {}
        epoch_lfw_acc = 0.0
        for name, data, issame in val_data_list:
            acc, std, thr = perform_val(backbone, device, data, issame)
            print(f"--- [VAL] {name.upper()} | Acc: {acc:.4f} ± {std:.4f} | Threshold: {thr:.2f} ---")
            val_accs[name] = acc
            if name == 'lfw':
                epoch_lfw_acc = acc
                
        # Lưu checkpoint latest để có thể resume
        ckpt = {
            'epoch': epoch,
            'backbone': backbone.state_dict(),
            'ema': ema.state_dict(),
            'header': header.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_lfw_acc': max(best_lfw_acc, epoch_lfw_acc),
        }
        torch.save(ckpt, latest_ckpt_path)
        
        # Lưu checkpoint tốt nhất dựa trên LFW accuracy
        if epoch_lfw_acc > best_lfw_acc:
            best_lfw_acc = epoch_lfw_acc
            torch.save(backbone.state_dict(), os.path.join(Config.checkpoint_dir, 'finetuned_best.pth'))
            print(f"--- New best LFW accuracy: {best_lfw_acc:.4f}! Saved finetuned_best.pth ---")
            
        # Ghi các metrics của epoch này ra CSV
        log_epoch_metrics(
            csv_file,
            epoch,
            avg_loss_epoch,
            avg_acc_epoch,
            avg_clean_acc_epoch,
            optimizer.param_groups[0]['lr'],
            header.m,
            val_accs
        )
        
        if not Config.freeze_backbone:
            ema.restore()
            
    print("Fine-tuning hoàn tất! Model tốt nhất đã lưu tại: ", os.path.join(Config.checkpoint_dir, 'finetuned_best.pth'))

if __name__ == "__main__":
    train_finetune()
