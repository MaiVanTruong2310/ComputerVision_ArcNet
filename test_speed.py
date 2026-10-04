import time
import torch
import sys
import os

# Thêm src vào path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from dataset import get_dataloader
from model import get_backbone, ArcMarginProduct
# Tự động kiểm tra và import theo phiên bản PyTorch để tránh lỗi tương thích
if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
    from torch.amp import autocast, GradScaler
    use_amp_api = True
else:
    from torch.cuda.amp import autocast, GradScaler
    use_amp_api = False

def test_dataloader():
    print("--- TESTING DATALOADER SPEED ---")
    loader = get_dataloader(Config.dataset_path, Config.batch_size, Config.num_workers, Config.pin_memory)
    start_time = time.time()
    count = 0
    for imgs, labels in loader:
        count += 1
        if count >= 20: # Test 20 batches
            break
    
    elapsed = time.time() - start_time
    print(f"DataLoader: {count} batches took {elapsed:.2f}s ({count/elapsed:.2f} it/s)")

def test_gpu():
    print("--- TESTING GPU SPEED ---")
    device = torch.device('cuda')
    backbone = get_backbone(Config.backbone_name, Config.embedding_size).to(device)
    header = ArcMarginProduct(Config.embedding_size, Config.num_classes).to(device)
    optimizer = torch.optim.SGD([{'params': backbone.parameters()}, {'params': header.parameters()}], lr=0.1)
    scaler = GradScaler('cuda') if use_amp_api else GradScaler()
    criterion = torch.nn.CrossEntropyLoss()
    
    imgs = torch.randn(Config.batch_size, 3, 112, 112).to(device)
    labels = torch.randint(0, Config.num_classes, (Config.batch_size,)).to(device)
    
    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        with autocast('cuda') if use_amp_api else autocast():
            feat = backbone(imgs)
            out = header(feat, labels)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.cuda.synchronize()
    start_time = time.time()
    count = 50
    for _ in range(count):
        optimizer.zero_grad()
        with autocast('cuda') if use_amp_api else autocast():
            feat = backbone(imgs)
            out = header(feat, labels)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    print(f"GPU Model: {count} batches took {elapsed:.2f}s ({count/elapsed:.2f} it/s)")

if __name__ == '__main__':
    test_dataloader()
    test_gpu()
