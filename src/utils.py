# ==============================================================================
# Tên file: utils.py
# Công dụng: Các hàm tiện ích hỗ trợ nạp dữ liệu kiểm thử.
#            Hàm load_bin dùng để giải mã tệp tin benchmark đóng gói pickle (.bin)
#            như LFW, AgeDB-30, CFP-FP thành danh sách ảnh Tensor đã chuẩn hóa
#            và danh sách nhãn so khớp cặp (issame_list).
# ==============================================================================
import pickle
import io
import torch
import numpy as np
from PIL import Image

def load_bin(path, image_size=(112, 112)):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')
    except:
        # Hỗ trợ nếu file bị lỗi encoding
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)

    data_list = []
    for i in range(len(bins)):
        _bin = bins[i]
        img = Image.open(io.BytesIO(_bin)).convert('RGB')
        img = img.resize(image_size)
        img = np.array(img).transpose(2, 0, 1) # HWC -> CHW
        img = (img - 127.5) / 128.0 # Normalize chuẩn InsightFace
        data_list.append(torch.tensor(img, dtype=torch.float32))
    
    return data_list, issame_list