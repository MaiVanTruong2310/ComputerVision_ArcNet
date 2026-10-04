import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import sys

# Thêm thư mục src vào path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import io

# Đảm bảo in ra được tiếng Việt trên Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from model import get_backbone
from eval import perform_val
from utils import load_bin

def get_epoch_from_filename(filename):
    """Trích xuất số epoch từ filename backbone_epoch_X.pth"""
    match = re.search(r'epoch_(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1

def evaluate_all_checkpoints(checkpoint_dir='checkpoints', dataset_path='dataset/raw/lfw.bin'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Đang sử dụng thiết bị: {device} ---")

    if not os.path.exists(dataset_path):
        print(f"LỖI: Không tìm thấy file dữ liệu tại {dataset_path}")
        return

    print(f"--- Đang tải tập dữ liệu: {dataset_path} ---")
    data, issame = load_bin(dataset_path)

    # Tìm tất cả file checkpoint
    if os.path.isfile(checkpoint_dir):
        ckpt_files = [os.path.basename(checkpoint_dir)]
        checkpoint_dir = os.path.dirname(checkpoint_dir)
    else:
        if not os.path.exists(checkpoint_dir):
            print(f"LỖI: Không tìm thấy thư mục/file checkpoint tại {checkpoint_dir}")
            return
        ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        try:
            ckpt_files.sort(key=get_epoch_from_filename)
        except Exception:
            ckpt_files.sort()

    if not ckpt_files:
        print("Không tìm thấy checkpoint nào để đánh giá.")
        return

    results = []
    best_acc = -1
    best_roc_data = None
    best_epoch = -1

    model = get_backbone('iresnet50', 512).to(device)

    print(f"--- Bắt đầu đánh giá {len(ckpt_files)} checkpoints ---")
    for ckpt_file in tqdm(ckpt_files):
        epoch = get_epoch_from_filename(ckpt_file)
        ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
        
        # Load weights
        try:
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Lỗi khi load {ckpt_file}: {e}")
            continue

        acc, std, thr, eer, tpr, fpr = perform_val(model, device, data, issame, return_full=True)
        
        results.append({
            'epoch': epoch,
            'accuracy': acc,
            'std': std,
            'threshold': thr,
            'eer': eer
        })

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            best_roc_data = (tpr, fpr)

    # Lưu kết quả vào CSV
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv('results/evaluation_metrics.csv', index=False)
    print("--- Đã lưu kết quả vào results/evaluation_metrics.csv ---")

    # Vẽ biểu đồ Accuracy vs Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['accuracy'], marker='o', linestyle='-', color='b', label='LFW Accuracy')
    plt.fill_between(df['epoch'], df['accuracy'] - df['std'], df['accuracy'] + df['std'], alpha=0.2, color='b')
    plt.title('Độ chính xác theo Epoch (LFW)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig('results/accuracy_vs_epoch.png')
    plt.close()

    # Vẽ biểu đồ ROC Curve cho model tốt nhất
    if best_roc_data:
        tpr, fpr = best_roc_data
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (Best Epoch {best_epoch}, Acc={best_acc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('results/roc_curve_best.png')
        plt.close()

    print(f"--- Đã vẽ xong biểu đồ trong thư mục results/ ---")
    print(f"--- Kết quả tốt nhất tại Epoch {best_epoch}: Accuracy = {best_acc:.4f} ---")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Đánh giá mô hình ArcFace")
    parser.add_argument('--checkpoint', type=str, default='check/tune/model/finetuned_best.pth', 
                        help='Đường dẫn tới thư mục checkpoints hoặc file .pth cụ thể (Mặc định: check/tune/model/finetuned_best.pth)')
    parser.add_argument('--dataset', type=str, default='dataset/raw/lfw.bin', 
                        help='Đường dẫn tới file lfw.bin (Mặc định: dataset/raw/lfw.bin)')
    args = parser.parse_args()
    
    evaluate_all_checkpoints(args.checkpoint, args.dataset)
