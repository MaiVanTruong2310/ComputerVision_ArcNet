import os
import sys
import torch
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image

# Thêm thư mục src vào path (sử dụng abspath để chạy được từ mọi vị trí)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, 'src'))

from model import get_backbone
from utils import load_bin
from config import Config
import io

# Đảm bảo in ra được tiếng Việt trên Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

# Các ngưỡng đánh giá bảo mật
SECURITY_THRESHOLDS = {
    "Low Security (LFW Standard)": 1.60,
    "Medium Security (PassNote Default)": 1.40,
    "High Security (Strict Guard)": 1.20
}

def load_and_preprocess_custom_image(path):
    """Đọc ảnh từ đĩa, chuyển sang RGB, resize về 112x112 và chuẩn hóa"""
    try:
        # Đọc ảnh bằng OpenCV tiếng Việt
        img_bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Resize về 112x112 chuẩn ArcFace
        img_resized = cv2.resize(img_rgb, (112, 112), interpolation=cv2.INTER_AREA)
        
        # Chuẩn hóa InsightFace
        img_data = img_resized.astype(np.float32)
        img_data = (img_data - 127.5) / 128.0
        img_data = img_data.transpose(2, 0, 1) # HWC -> CHW
        
        return torch.tensor(img_data, dtype=torch.float32)
    except Exception as e:
        print(f"Lỗi khi đọc file {os.path.basename(path)}: {e}")
        return None

class LookalikeTwinsTester:
    def __init__(self, checkpoint_path=None, dataset_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Đảm bảo đường dẫn tuyệt đối chính xác
        if checkpoint_path is None:
            checkpoint_path = os.path.join(ROOT_DIR, 'check', 'tune', 'model', 'finetuned_best.pth')
        if dataset_path is None:
            dataset_path = os.path.join(ROOT_DIR, 'dataset', 'raw', 'lfw.bin')
            
        self.results_dir = os.path.join(ROOT_DIR, 'robustness_testing', 'results')
        self.custom_dir = os.path.join(ROOT_DIR, 'robustness_testing', 'custom_test_images')
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.custom_dir, exist_ok=True)
        
        # Load Model
        print(f"--- Đang tải mô hình từ: {checkpoint_path} ---")
        self.model = get_backbone(Config.backbone_name, Config.embedding_size).to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Load LFW
        self.dataset_path = dataset_path
        self.data_list = None
        self.issame = None

    def load_lfw(self):
        if self.data_list is None and os.path.exists(self.dataset_path):
            print(f"--- Đang tải LFW từ {self.dataset_path} ---")
            self.data_list, self.issame = load_bin(self.dataset_path)
            self.issame = np.array(self.issame)

    def extract_lfw_embeddings(self):
        self.load_lfw()
        if self.data_list is None:
            return None, None
            
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(self.data_list), Config.val_batch_size):
                batch = torch.stack(self.data_list[i:i + Config.val_batch_size]).to(self.device)
                embed = self.model(batch)
                embed = torch.nn.functional.normalize(embed)
                embeddings.append(embed.cpu().numpy())
                
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings[0::2], embeddings[1::2]

    def test_lfw_hard_negatives(self):
        """Tìm 20 cặp khác người (Different) dễ nhận nhầm nhất (Hard Negatives) trong LFW"""
        print("\n=== Đang tìm các cặp người giống nhau dễ nhận nhầm nhất (LFW Hard Negatives) ===")
        emb1, emb2 = self.extract_lfw_embeddings()
        if emb1 is None:
            print("Không tìm thấy LFW dataset. Bỏ qua bước này.")
            return
            
        diff = np.subtract(emb1, emb2)
        dist = np.sum(np.square(diff), 1) # Squared L2 distance
        
        # Tính Cosine similarity
        cos_sim = np.sum(emb1 * emb2, axis=1)
        
        # Lọc ra các cặp khác người (issame == False)
        diff_indices = np.where(self.issame == False)[0]
        
        # Sắp xếp các cặp khác người theo khoảng cách tăng dần (tức là giống nhau nhất)
        sorted_diff_indices = diff_indices[np.argsort(dist[diff_indices])]
        
        hard_negatives = []
        for idx in sorted_diff_indices[:20]: # Top 20 cặp giống nhau nhất
            hard_negatives.append({
                'Pair_Index': int(idx),
                'Euclidean_Dist': float(dist[idx]),
                'Cosine_Similarity': float(cos_sim[idx]),
                # Ở ngưỡng tiêu chuẩn 1.60, có bị nhận nhầm thành SAME không?
                'False_Accept_Standard': dist[idx] < 1.60,
                'False_Accept_Medium': dist[idx] < 1.40,
                'False_Accept_Strict': dist[idx] < 1.20
            })
            
        df = pd.DataFrame(hard_negatives)
        df.to_csv(f'{self.results_dir}/lfw_hard_negatives.csv', index=False)
        
        # Lưu báo cáo text
        report_path = f"{self.results_dir}/lfw_hard_negatives_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Báo cáo Các cặp dễ nhận nhầm nhất trong LFW (Hard Negatives)\n\n")
            f.write("Báo cáo này liệt kê top 20 cặp hình ảnh của các cá nhân khác nhau có độ tương đồng cao nhất.\n")
            f.write("Giúp đánh giá khả năng chống giả mạo / nhận nhầm của mô hình ở các cấu hình ngưỡng bảo mật khác nhau.\n\n")
            f.write(df.to_markdown(index=False) + "\n")
            
        print(f"  > Đã trích xuất xong 20 cặp. Báo cáo lưu tại: {report_path}")

    def test_custom_images(self):
        """So sánh chéo toàn bộ ảnh đặt trong thư mục custom_test_images"""
        print("\n=== Đang chạy so sánh chéo ảnh tự chọn (custom_test_images/) ===")
        # Quét thư mục
        supported_exts = ('.png', '.jpg', '.jpeg', '.webp')
        img_paths = [os.path.join(self.custom_dir, f) for f in os.listdir(self.custom_dir) 
                     if f.lower().endswith(supported_exts)]
                     
        if not img_paths:
            # Viết file hướng dẫn vào thư mục custom_test_images
            readme_path = f"{self.custom_dir}/README.txt"
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write("HUONG DAN SU DUNG THU MUC ANH TU CHON:\n")
                f.write("1. Copy cac file anh cua anh em, sinh doi, hoac nguoi than vao thu muc nay.\n")
                f.write("2. Nen cat (crop) can canh khuon mat tap trung giong nhu anh the/Passport.\n")
                f.write("3. Chay lai script nay de so sanh tung cap anh va xem model co nhan nham hoac phan biet duoc hay khong.\n")
            print(f"  > Thư mục trống. Đã tạo hướng dẫn sử dụng tại: {readme_path}")
            return
            
        print(f"  > Tìm thấy {len(img_paths)} ảnh tự chọn. Đang trích xuất embeddings...")
        
        custom_embeddings = {}
        for path in img_paths:
            filename = os.path.basename(path)
            tensor = load_and_preprocess_custom_image(path)
            if tensor is None:
                continue
                
            # Đưa qua model
            with torch.no_grad():
                batch = tensor.unsqueeze(0).to(self.device)
                embed = self.model(batch)
                embed = torch.nn.functional.normalize(embed)
                custom_embeddings[filename] = embed.cpu().numpy()[0]
                
        if len(custom_embeddings) < 2:
            print("  > Cần ít nhất 2 ảnh hợp lệ để so sánh chéo.")
            return
            
        # So sánh chéo cặp
        filenames = list(custom_embeddings.keys())
        results = []
        
        for i in range(len(filenames)):
            for j in range(i + 1, len(filenames)):
                fn1 = filenames[i]
                fn2 = filenames[j]
                
                emb1 = custom_embeddings[fn1]
                emb2 = custom_embeddings[fn2]
                
                # Tính khoảng cách
                diff = emb1 - emb2
                dist_l2 = np.sum(np.square(diff)) # Squared L2
                cos_sim = np.sum(emb1 * emb2)
                
                # Xác định kết quả ở các ngưỡng khác nhau
                decision_std = "SAME" if dist_l2 < 1.60 else "DIFF"
                decision_med = "SAME" if dist_l2 < 1.40 else "DIFF"
                decision_strict = "SAME" if dist_l2 < 1.20 else "DIFF"
                
                results.append({
                    'File_1': fn1,
                    'File_2': fn2,
                    'Euclidean_Dist': dist_l2,
                    'Cosine_Similarity': cos_sim,
                    'Standard_Thr_1.60': decision_std,
                    'Medium_Thr_1.40': decision_med,
                    'Strict_Thr_1.20': decision_strict
                })
                
        df = pd.DataFrame(results)
        df.to_csv(f'{self.results_dir}/custom_comparison_results.csv', index=False)
        
        # Báo cáo dạng Markdown
        report_path = f"{self.results_dir}/custom_comparison_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Báo cáo Đánh giá So sánh chéo Trực quan (Ảnh Tự Chọn)\n\n")
            f.write("Kết quả so sánh cặp ảnh gia đình / anh em sinh đôi tự chọn trên các ngưỡng bảo mật khác nhau:\n\n")
            
            # Giải thích các ngưỡng
            f.write("### 📌 Định nghĩa Ngưỡng Bảo mật:\n")
            for k, v in SECURITY_THRESHOLDS.items():
                f.write(f"- **{k}**: Khoảng cách L2 < `{v:.2f}` (Cosine Sim lớn hơn ~`{1 - (v/2):.2f}`)\n")
            f.write("\n")
            
            f.write("### 📊 Bảng so sánh chi tiết:\n")
            f.write(df.to_markdown(index=False) + "\n")
            
        print(f"  > Đã so sánh chéo {len(results)} cặp. Báo cáo kết quả lưu tại: {report_path}")

if __name__ == '__main__':
    tester = LookalikeTwinsTester()
    tester.test_lfw_hard_negatives()
    tester.test_custom_images()
