import os
import sys
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Sử dụng lại module recordio từ src theo yêu cầu
from src.recordio import IndexedRecordIO, unpack

class FinetuneRecDataset(Dataset):
    def __init__(self, rec_path, transform=None, subsample_ratio=1.0):
        super(FinetuneRecDataset, self).__init__()
        self.transform = transform
        self.rec_path = rec_path
        self.idx_path = rec_path.replace('.rec', '.idx')
        self.lst_path = rec_path.replace('.rec', '.lst')
        
        if not os.path.exists(self.rec_path):
            raise FileNotFoundError(f"Không tìm thấy file {self.rec_path}")
        if not os.path.exists(self.idx_path):
            raise FileNotFoundError(f"Không tìm thấy file {self.idx_path}")
            
        self.imgrec = IndexedRecordIO(self.idx_path, self.rec_path, 'r')
        
        # Đọc nhãn
        self.labels = {}
        if os.path.exists(self.lst_path):
            print(f"Loading labels from {self.lst_path}...")
            with open(self.lst_path, 'r') as f:
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        label = int(parts[2])
                        self.labels[i + 1] = label
                        
            # Tách biệt: Thực hiện rút gọn dữ liệu ảo nếu subsample_ratio < 1.0
            if subsample_ratio < 1.0:
                print(f"Subsampling dataset virtually to {subsample_ratio * 100}%...")
                # Lấy danh sách các class (nhãn) độc nhất
                unique_labels = sorted(list(set(self.labels.values())))
                
                # Chọn ngẫu nhiên nhưng cố định các class (seed 42)
                np.random.seed(42)
                num_classes_to_keep = max(1, int(len(unique_labels) * subsample_ratio))
                classes_to_keep = set(np.random.choice(unique_labels, size=num_classes_to_keep, replace=False))
                
                # Tạo bản đồ ánh xạ lại nhãn từ 0 đến N-1
                label_map = {old_lbl: new_lbl for new_lbl, old_lbl in enumerate(sorted(list(classes_to_keep)))}
                
                # Lọc nhãn và ảnh
                filtered_labels = {}
                for idx, lbl in self.labels.items():
                    if lbl in classes_to_keep:
                        filtered_labels[idx] = label_map[lbl]
                
                self.labels = filtered_labels
                self.imgidx = np.array(sorted(self.labels.keys()))
                print(f"Subsampled to {len(classes_to_keep)} classes and {len(self.imgidx)} images.")
            else:
                self.imgidx = np.array(sorted(self.labels.keys()))
                print(f"Loaded {len(self.imgidx)} images from .lst file")
        else:
            keys = sorted(self.imgrec.keys)
            first_key = next((k for k in keys if k > 0), None)
            if first_key is None:
                raise ValueError("Dataset trống.")
            s = self.imgrec.read_idx(first_key)
            header, _ = unpack(s)
            if header.flag > 0:
                self.header0 = (int(header.label[0]), int(header.label[1]))
                self.imgidx = np.array(range(self.header0[0], self.header0[1]))
            else:
                self.imgidx = np.array([k for k in keys if k > 0])

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img_bytes = unpack(s)
        
        if self.labels:
            label = self.labels[idx]
        else:
            label = header.label
            if not isinstance(label, (float, int)):
                label = label[0]
                
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
        sample = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if sample is None:
            sample = np.zeros((112, 112, 3), dtype=np.uint8)
        else:
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        
        sample = Image.fromarray(sample)
        
        if self.transform is not None:
            sample = self.transform(sample)
        else:
            sample = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])(sample)
            
        return sample, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.imgidx)

def get_finetune_dataloader(rec_path, batch_size, num_workers=4, pin_memory=False, subsample_ratio=1.0):
    # Data Augmentation cơ bản cho Fine-tune
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    dataset = FinetuneRecDataset(rec_path, transform=transform, subsample_ratio=subsample_ratio)
    
    # Tính số class tự động
    if dataset.labels:
        num_classes = max(dataset.labels.values()) + 1
    else:
        print("Đang quét để tìm số classes từ header...")
        max_label = 0
        for i in range(len(dataset)):
            idx = dataset.imgidx[i]
            s = dataset.imgrec.read_idx(idx)
            header, _ = unpack(s)
            label = header.label
            if not isinstance(label, (float, int)):
                label = label[0]
            if label > max_label:
                max_label = int(label)
        num_classes = max_label + 1
        print(f"Đã quét xong: tìm thấy {num_classes} classes.")
        
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    return dataloader, num_classes
