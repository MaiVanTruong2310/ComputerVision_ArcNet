import torch
import os
from src.model import get_backbone

def export_to_onnx():
    checkpoint_path = 'check/tune/model/finetuned_best.pth'
    onnx_path = 'outputs/finetuned_best.onnx'
    
    print(f"Loading weights from {checkpoint_path}...")
    
    # Khởi tạo model với cùng cấu hình
    model = get_backbone(name='iresnet50', embedding_size=512, dropout=0.0)
    
    # Load weights
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'backbone' in ckpt:
        state_dict = ckpt['backbone']
    else:
        state_dict = ckpt
        
    model.load_state_dict(state_dict)
    model.eval() # Bắt buộc phải chuyển sang chế độ eval trước khi export
    
    # Tạo dummy input (1 ảnh, 3 kênh màu, kích thước 112x112)
    dummy_input = torch.randn(1, 3, 112, 112, device='cpu')
    
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    print(f"Exporting model to ONNX format...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,        # Lưu các tham số huấn luyện vào trong file model
        opset_version=11,          # Phiên bản ONNX chuẩn tương thích rộng rãi
        do_constant_folding=True,  # Tối ưu hóa hằng số cho model nhỏ và nhanh hơn
        input_names=['input'],     # Đặt tên cho biến đầu vào
        output_names=['output'],   # Đặt tên cho biến đầu ra
        dynamic_axes={             # Khai báo kích thước batch size có thể thay đổi linh hoạt
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Hoàn tất! File ONNX đã được lưu tại: {onnx_path}")

if __name__ == '__main__':
    export_to_onnx()
