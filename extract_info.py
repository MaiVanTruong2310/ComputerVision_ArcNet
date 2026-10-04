import os
import torch
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from src.model import get_backbone

# 1. Trích xuất thông tin Model
device = torch.device('cpu')
model = get_backbone(name='iresnet50', embedding_size=512, dropout=0.0)
ckpt = torch.load('check/tune/model/finetuned_best.pth', map_location='cpu', weights_only=False)

# Chỉnh sửa nếu ckpt là dictionary
if 'backbone' in ckpt:
    state_dict = ckpt['backbone']
else:
    state_dict = ckpt

model.load_state_dict(state_dict)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=== MODEL PARAMETERS ===")
print(f"Architecture: IResNet50")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Embedding size: 512")
print("Main layers details:")
for name, module in model.named_children():
    params = sum(p.numel() for p in module.parameters())
    print(f" - {name}: {params:,} parameters")

# 2. Đọc TensorBoard Logs
print("\n=== READ TENSORBOARD LOGS ===")
log_dir = 'logs'
event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if 'events.out.tfevents' in f]

# Lấy file log có dung lượng lớn nhất (thường là file train lâu nhất)
event_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
best_log = event_files[0]
print(f"Read main log file: {best_log}")

ea = EventAccumulator(best_log)
ea.Reload()

# Trích xuất dữ liệu
loss_data = []
if 'Epoch/Loss' in ea.scalars.Keys():
    loss_events = ea.scalars.Items('Epoch/Loss')
    loss_data = [(e.step, e.value) for e in loss_events]

lfw_data = []
if 'Val/LFW_Acc' in ea.scalars.Keys():
    lfw_events = ea.scalars.Items('Val/LFW_Acc')
    lfw_data = [(e.step, e.value) for e in lfw_events]

# 3. Vẽ biểu đồ
if loss_data and lfw_data:
    fig, ax1 = plt.subplots(figsize=(10, 5))

    epochs_loss = [x[0] for x in loss_data]
    vals_loss = [x[1] for x in loss_data]
    
    epochs_lfw = [x[0] for x in lfw_data]
    vals_lfw = [x[1] for x in lfw_data]

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs_loss, vals_loss, color=color, marker='o', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('LFW Accuracy', color=color)  
    ax2.plot(epochs_lfw, vals_lfw, color=color, marker='s', label='LFW Acc')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0.0, 1.0])

    fig.tight_layout()  
    
    # Save the plot to the conversation brain directory
    out_dir = r"C:\Temp\Facenet\information"
    out_path = os.path.join(out_dir, "training_curve.png")
    plt.title("Loss and LFW Accuracy over Epochs")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    print(f"Saved plot at: {out_path}")
else:
    print("Could not find enough Loss and LFW_Acc data in logs to plot.")
