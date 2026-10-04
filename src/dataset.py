# ==============================================================================
# Tên file: dataset.py
# Công dụng: Lớp nạp dữ liệu FaceDataset kế thừa từ torch.utils.data.Dataset.
#            Đọc nhãn từ file .lst và ảnh nhị phân từ file .rec (thông qua recordio.py),
#            tiến hành giải mã ảnh JPEG/PNG bằng OpenCV và chuyển đổi hệ màu BGR->RGB
#            cùng các phép biến đổi (transform) để cấp dữ liệu dạng Tensor cho DataLoader.
# ==============================================================================
import os
import torch
import numpy as np
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Sử dụng bộ đọc recordio tự viết
from recordio import IndexedRecordIO, unpack

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super(FaceDataset, self).__init__()
        self.transform = transform
        self.root_dir = root_dir
        
        # Tìm file .rec
        paths_to_check = [
            os.path.join(root_dir, 'train.rec'),
            os.path.join(root_dir, 'raw', 'train.rec')
        ]
        
        path_imgrec = None
        for p in paths_to_check:
            if os.path.exists(p):
                path_imgrec = p
                break
        
        if path_imgrec is None:
            raise FileNotFoundError(f"Không tìm thấy file train.rec trong {root_dir}")

        path_imgidx = path_imgrec.replace('.rec', '.idx')
        path_imglst = path_imgrec.replace('.rec', '.lst')
        
        # Khởi tạo bộ đọc
        self.imgrec = IndexedRecordIO(path_imgidx, path_imgrec, 'r')
        
        # LOGIC MỚI: Đọc nhãn từ file .lst nếu có
        self.labels = {}
        if os.path.exists(path_imglst):
            print(f"Loading labels from {path_imglst}...")
            with open(path_imglst, 'r') as f:
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        # Index (1-based) tương ứng với dòng (0-based) trong im2rec
                        # Chúng ta giả định dòng thứ i trong .lst khớp với Key i+1 trong .rec
                        label = int(parts[2])
                        self.labels[i + 1] = label
            
            self.imgidx = np.array(sorted(self.labels.keys()))
            print(f"Loaded {len(self.imgidx)} images from .lst file")
        else:
            # Nếu không có .lst, quay lại logic đọc từ Header (nhưng bộ CASIA này sẽ lỗi nếu không có .lst)
            keys = sorted(self.imgrec.keys)
            first_key = None
            for k in keys:
                if k > 0:
                    first_key = k
                    break
            
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
        
        # Ưu tiên lấy label từ .lst đã nạp
        if self.labels:
            label = self.labels[idx]
        else:
            # Fallback lấy từ header
            label = header.label
            if not isinstance(label, (float, int)):
                label = label[0]
            
        # Giải mã ảnh
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
        sample = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if sample is None:
            # Trường hợp ảnh lỗi, trả về ảnh đen
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

def get_dataloader(root_dir, batch_size, num_workers=4, pin_memory=False):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), # Đảo ngược ảnh tăng độ đa dạng dataset
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    dataset = FaceDataset(root_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0)
    )
    return dataloader
