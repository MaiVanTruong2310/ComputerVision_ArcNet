# ==============================================================================
# Tên file: ema.py
# Công dụng: Lớp EMA (Exponential Moving Average) tính toán trung bình động lũy thừa
#            trọng số của mô hình trong quá trình huấn luyện với decay rate hệ số mượt.
#            Cung cấp phương thức áp dụng trọng số EMA tạm thời khi chạy validation/kiểm thử
#            và khôi phục lại trọng số gốc để tiếp tục huấn luyện.
# ==============================================================================
import torch

class EMA:
    """
    Exponential Moving Average of model weights.
    Standard technique in SOTA Face Recognition for stability and performance.
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in self.model.state_dict().items():
            self.shadow[name] = param.data.clone()

    def update(self):
        with torch.no_grad():
            for name, param in self.model.state_dict().items():
                if param.dtype.is_floating_point:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
                else:
                    self.shadow[name].copy_(param.data)

    def apply_shadow(self):
        """Replace model weights with shadow weights (for validation/saving)"""
        for name, param in self.model.state_dict().items():
            self.backup[name] = param.data.clone()
            param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original weights (to continue training)"""
        for name, param in self.model.state_dict().items():
            param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        for name, value in state_dict.items():
            self.shadow[name].copy_(value)
