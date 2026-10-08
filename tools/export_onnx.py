import argparse
import os
import sys
import torch
import io

# Đảm bảo in ra được tiếng Việt trên Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Thêm thư mục gốc và src vào sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(ROOT_DIR, 'src'))
sys.path.append(ROOT_DIR)

from model import get_backbone
from config import Config

def export_to_onnx(checkpoint_path=None, output_path=None, opset_version=11):
    # Tự động tìm checkpoint nếu không chỉ định
    if checkpoint_path is None:
        candidates = [
            'check/tune/model/finetuned_best.pth',
            'check/train/model/backbone_best.pth',
            'fine_tune/checkpoints/latest.pth',
            'checkpoints/backbone_epoch_16.pth',
            'checkpoints/latest.pth',
        ]
        for c in candidates:
            if os.path.exists(c):
                checkpoint_path = c
                break
                
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"LỖI: Không tìm thấy file checkpoint. Vui lòng chỉ định đường dẫn hợp lệ với flag --checkpoint.")
        return

    if output_path is None:
        base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        output_path = os.path.join('outputs', f"{base_name}.onnx")
        
    print(f"--- Đang tải weights từ: {checkpoint_path} ---")
    model = get_backbone(name=Config.backbone_name, embedding_size=Config.embedding_size, dropout=0.0)
    
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'backbone' in ckpt:
        state_dict = ckpt['backbone']
    else:
        state_dict = ckpt
        
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, 112, 112, device='cpu')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"--- Đang xuất sang định dạng ONNX (Opset {opset_version})... ---")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Hoàn tất! File ONNX đã được lưu tại: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch ArcFace model to ONNX")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Đường dẫn tới file checkpoint .pth')
    parser.add_argument('--output', type=str, default=None,
                        help='Đường dẫn file .onnx đầu ra')
    parser.add_argument('--opset', type=int, default=11,
                        help='Phiên bản ONNX opset (mặc định: 11)')
    args = parser.parse_args()
    
    export_to_onnx(checkpoint_path=args.checkpoint, output_path=args.output, opset_version=args.opset)

if __name__ == '__main__':
    main()
