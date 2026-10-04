
# ==============================================================================
# Tên file: inspect_dataset.py
# Công dụng: Công cụ chẩn đoán và kiểm tra dữ liệu đầu vào (Dataset Diagnostics Tool).
#            Quét các tệp tin train.rec, train.idx, train.lst để đếm số lượng ảnh,
#            số lượng người (classes) thực tế và đưa ra gợi ý cấu hình num_classes phù hợp.
# ==============================================================================
import os
import argparse
import sys
import io
import numpy as np

# Đảm bảo in ra được tiếng Việt trên Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Thêm thư mục src vào path để có thể import các submodule
sys.path.append(os.path.dirname(__file__))

from recordio import IndexedRecordIO, unpack

def inspect_rec(path):
    path_idx = path.replace('.rec', '.idx')
    path_lst = path.replace('.rec', '.lst')
    
    if not os.path.exists(path_idx):
        print(f"LỖI: Không tìm thấy file index {path_idx}")
        return

    # Sử dụng IndexedRecordIO thuần Python
    imgrec = IndexedRecordIO(path_idx, path, 'r')
    
    print("\n--- Thông tin Dataset ArcFace (Pure Python Reader) ---")
    print(f"File REC: {path}")
    
    num_classes = 0
    num_images = 0
    
    # Ưu tiên kiểm tra file .lst
    if os.path.exists(path_lst):
        print(f"Phát hiện file danh sách nhãn: {path_lst}")
        unique_labels = set()
        count = 0
        with open(path_lst, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    unique_labels.add(int(parts[2]))
                    count += 1
        num_classes = len(unique_labels)
        num_images = count
        print(f"-> Phân tích từ .lst: Có {num_images} ảnh thuộc {num_classes} Class (người).")
    else:
        # Dự phòng: đọc từ Record 1
        keys = sorted(imgrec.keys)
        first_key = next((k for k in keys if k > 0), None)
        
        if first_key:
            s = imgrec.read_idx(first_key)
            header, _ = unpack(s)
            if isinstance(header.label, np.ndarray) and header.flag > 0:
                id_start = int(header.label[0])
                id_end = int(header.label[1])
                num_images = id_end - id_start
                num_classes = id_start - 1
                print(f"-> Phân tích từ Metadata: Có {num_images} ảnh thuộc {num_classes} Class.")
            else:
                print("-> Không tìm thấy metadata nhãn. Đang ước tính từ labels trong records...")
                # Quét 2000 records
                max_label = 0
                for k in keys[1:2001]:
                    s = imgrec.read_idx(k)
                    if not s: continue
                    h, _ = unpack(s)
                    l = h.label
                    if not isinstance(l, (float, int)): l = l[0]
                    if l > max_label: max_label = l
                num_classes = int(max_label) + 1
                num_images = len(keys)
                print(f"-> Ước tính sơ bộ: {num_images} records, {num_classes} Class.")

    print("--------------------------------------------------")
    if num_classes > 1:
        print(f"GỢI Ý: Hãy thiết lập num_classes = {num_classes} trong src/train.py")
    elif num_classes == 1:print(f"CẢNH BÁO: Chỉ tìm thấy 1 Class. Kiểm tra lại nạp nhãn từ file .lst!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None, help='Path to .rec file')
    args = parser.parse_args()
    
    target_path = args.path
    if target_path is None:
        potential_paths = ['dataset/train.rec', 'dataset/raw/train.rec']
        target_path = next((p for p in potential_paths if os.path.exists(p)), None)
                
    if target_path and os.path.exists(target_path):
        inspect_rec(target_path)
    else:
        print(f"LỖI: Không tìm thấy file .rec.")