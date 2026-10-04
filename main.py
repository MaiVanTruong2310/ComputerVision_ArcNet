import argparse
import os
import sys
import io

# Đảm bảo in ra được tiếng Việt trên Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Thêm thư mục src vào path để có thể import các submodule
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from train import train
from eval import perform_val
from model import get_backbone
from utils import load_bin
import torch

def main():
    parser = argparse.ArgumentParser(description='ArcFace Training and Evaluation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='Chế độ: train (huấn luyện) hoặc eval (đánh giá)')
    parser.add_argument('--dataset', type=str, default=Config.dataset_path,
                        help='Đường dẫn đến thư mục dữ liệu')
    parser.add_argument('--backbone', type=str, default=None,
                        help='Đường dẫn đến file checkpoint (.pth) để eval')
    parser.add_argument('--target', type=str, default='lfw',
                        help='Tên tập dữ liệu evaluation (ví dụ: lfw)')
    
    args = parser.parse_args()

    if args.mode == 'train':
        print("--- Bắt đầu chế độ TRAINING ---")
        train()
    
    elif args.mode == 'eval':
        print(f"--- Bắt đầu chế độ EVALUATION trên {args.target} ---")
        if args.backbone is None:
            print("LỖI: Vui lòng cung cấp path đến file checkpoint bằng --backbone")
            return
            
        device = Config.device
        model = get_backbone(
            name=Config.backbone_name, 
            embedding_size=Config.embedding_size,
            dropout=0 # Không dùng dropout khi eval
        ).to(device)
        
        # Load weights
        state_dict = torch.load(args.backbone, map_location=device, weights_only=False)
        if 'backbone' in state_dict:
            state_dict = state_dict['backbone']
        model.load_state_dict(state_dict)
        
        bin_path = os.path.join(args.dataset, f'{args.target}.bin')
        if not os.path.exists(bin_path):
            bin_path = os.path.join(args.dataset, 'raw', f'{args.target}.bin')
            
        if not os.path.exists(bin_path):
            print(f"LỖI: Không tìm thấy file {args.target}.bin")
            return
            
        data_list, issame_list = load_bin(bin_path)
        acc, std, threshold = perform_val(model, device, data_list, issame_list)
        
        print(f"KẾT QUẢ ĐÁNH GIÁ [{args.target.upper()}]:")
        print(f"Accuracy: {acc:.4f} +- {std:.4f}")
        print(f"Best Threshold: {threshold:.2f}")

if __name__ == "__main__":
    main()
