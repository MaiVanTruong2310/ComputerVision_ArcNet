import os
import sys
import random
import cv2
import numpy as np

# Thêm thư mục src vào path để import recordio
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from recordio import IndexedRecordIO, unpack

def extract_random_images(rec_path, output_dir, num_images=10):
    idx_path = rec_path.replace('.rec', '.idx')
    
    if not os.path.exists(rec_path):
        print(f"LỖI: Không tìm thấy file .rec tại {rec_path}")
        return
    if not os.path.exists(idx_path):
        print(f"LỖI: Không tìm thấy file .idx tại {idx_path}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Khởi tạo bộ đọc
    print("Đang nạp file index...")
    imgrec = IndexedRecordIO(idx_path, rec_path, 'r')
    
    # Lấy các keys hợp lệ (Key > 0 vì key 0 thường là metadata)
    keys = [k for k in imgrec.keys if k > 0]
    if len(keys) == 0:
        print("LỖI: File dataset trống.")
        return
        
    print(f"Tổng số ảnh trong dataset: {len(keys)}")
    
    # Chọn ngẫu nhiên 10 keys
    selected_keys = random.sample(keys, min(num_images, len(keys)))
    print(f"Đã chọn ngẫu nhiên các keys: {selected_keys}")
    
    for i, key in enumerate(selected_keys):
        s = imgrec.read_idx(key)
        header, img_bytes = unpack(s)
        
        # Lấy nhãn
        label = header.label
        if not isinstance(label, (float, int)):
            label = label[0]
            
        # Giải mã ảnh bằng OpenCV
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        
        if img is not None:
            out_path = os.path.join(output_dir, f"sample_{i+1}_key_{key}_label_{int(label)}.jpg")
            cv2.imwrite(out_path, img)
            print(f"Đã trích xuất: {out_path}")
        else:
            print(f"LỖI: Không thể giải mã ảnh cho key {key}")
            
    print(f"\nHoàn tất! Các ảnh đã được lưu vào thư mục: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    rec_path = 'dataset/raw/train.rec'
    if not os.path.exists(rec_path):
        rec_path = 'dataset/train.rec'
        
    output_dir = 'extracted_samples'
    extract_random_images(rec_path, output_dir, 10)
