import os
import sys
import torch
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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

# Các hàm phụ trợ chuyển đổi Tensor <-> OpenCV Image
def tensor_to_cv2(tensor):
    """Chuyển tensor [-1.0, 1.0] CHW -> OpenCV RGB [0, 255] HWC"""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * 128.0 + 127.5).clip(0, 255).astype(np.uint8)
    return img

def cv2_to_tensor(img):
    """Chuyển OpenCV RGB [0, 255] HWC -> tensor [-1.0, 1.0] CHW"""
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = img.transpose(2, 0, 1)
    return torch.tensor(img, dtype=torch.float32)

# Các hàm tạo nhiễu/perturbation
def apply_brightness(img, beta):
    """Thay đổi độ sáng (beta > 0: sáng lên, beta < 0: tối đi)"""
    return cv2.convertScaleAbs(img, alpha=1.0, beta=beta)

def apply_blur(img, ksize):
    """Làm mờ ảnh bằng Gaussian Blur (ksize lẻ)"""
    if ksize <= 1:
        return img
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def apply_resolution(img, size):
    """Giảm độ phân giải xuống size x size rồi scale ngược lại 112x112"""
    if size >= 112:
        return img
    temp = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (112, 112), interpolation=cv2.INTER_LINEAR)

def apply_rotation(img, angle):
    """Xoay ảnh một góc angle độ xung quanh tâm"""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def apply_occlusion(img, occ_type):
    """Giả lập che khuất bằng vẽ hình hộp đen lên mắt hoặc miệng"""
    out = img.copy()
    if occ_type == 'glasses':
        # Che vùng mắt (Sunglasses): dòng từ 38 đến 60, cột từ 15 đến 97
        cv2.rectangle(out, (15, 38), (97, 60), (0, 0, 0), -1)
    elif occ_type == 'mask':
        # Che vùng miệng và cằm (Mask): dòng từ 65 đến 112
        cv2.rectangle(out, (10, 65), (102, 112), (200, 200, 200), -1)
    return out

# Hàm tính toán chỉ số đánh giá nhanh
def evaluate_metrics(embeddings1, embeddings2, issame, fixed_threshold=1.60):
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    
    # 1. Tính Accuracy tại ngưỡng cố định (1.60)
    preds = np.less(dist, fixed_threshold)
    acc_fixed = np.mean(preds == issame)
    
    # 2. Tính EER và TPR @ FAR
    # Sắp xếp khoảng cách
    thresholds = np.arange(0, 4, 0.01)
    tprs = np.zeros(len(thresholds))
    fprs = np.zeros(len(thresholds))
    for idx, thr in enumerate(thresholds):
        predict_issame = np.less(dist, thr)
        tp = np.sum(np.logical_and(predict_issame, issame))
        fp = np.sum(np.logical_and(predict_issame, np.logical_not(issame)))
        tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(issame)))
        fn = np.sum(np.logical_and(np.logical_not(predict_issame), issame))
        tprs[idx] = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
        fprs[idx] = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
        
    fnrs = 1 - tprs
    idx_eer = np.nanargmin(np.absolute((fprs - fnrs)))
    eer = (fprs[idx_eer] + fnrs[idx_eer]) / 2
    
    # Cosine Similarity
    # Normalize embeddings to measure cosine similarity directly
    norms1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
    cos_sim = np.sum((embeddings1 / norms1) * (embeddings2 / norms2), axis=1)
    
    mean_sim_same = np.mean(cos_sim[issame == True])
    mean_sim_diff = np.mean(cos_sim[issame == False])
    
    return acc_fixed, eer, mean_sim_same, mean_sim_diff

class StressTestRunner:
    def __init__(self, checkpoint_path=None, dataset_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Đảm bảo đường dẫn tuyệt đối chính xác
        if checkpoint_path is None:
            checkpoint_path = os.path.join(ROOT_DIR, 'check', 'tune', 'model', 'finetuned_best.pth')
        if dataset_path is None:
            dataset_path = os.path.join(ROOT_DIR, 'dataset', 'raw', 'lfw.bin')
            
        self.results_dir = os.path.join(ROOT_DIR, 'robustness_testing', 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load Model
        print(f"--- Đang tải mô hình từ: {checkpoint_path} ---")
        self.model = get_backbone(Config.backbone_name, Config.embedding_size).to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Load LFW
        print(f"--- Đang tải tập dữ liệu LFW từ: {dataset_path} ---")
        full_data_list, full_issame = load_bin(dataset_path)
        
        # Subsample to 500 pairs (1000 images) for stress test speed
        num_pairs_to_test = 500
        self.issame = np.array(full_issame[:num_pairs_to_test])
        self.data_list = full_data_list[:num_pairs_to_test * 2]
        print(f"--- Đã subsample tập LFW thành {num_pairs_to_test} cặp ({len(self.data_list)} ảnh) để chạy stress test nhanh ---")
        
    def run_inference(self, perturbed_data):
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(perturbed_data), Config.val_batch_size):
                batch = torch.stack(perturbed_data[i:i + Config.val_batch_size]).to(self.device)
                
                # Để đánh giá khắt khe, chúng ta KO dùng lật ảnh (flip) nhằm đo đúng độ nhạy của hướng ảnh gốc
                embed = self.model(batch)
                embed = torch.nn.functional.normalize(embed)
                embeddings.append(embed.cpu().numpy())
                
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings[0::2], embeddings[1::2]

    def test_brightness(self):
        print("\n=== Đang chạy kiểm thử ánh sáng (Brightness/Lighting) ===")
        # Các mức tăng/giảm độ sáng
        levels = [-100, -75, -50, -25, 0, 25, 50, 75, 100]
        results = []
        
        for lvl in levels:
            perturbed = []
            for t in self.data_list:
                img_cv = tensor_to_cv2(t)
                perturbed_img = apply_brightness(img_cv, lvl)
                perturbed.append(cv2_to_tensor(perturbed_img))
                
            emb1, emb2 = self.run_inference(perturbed)
            acc, eer, sim_same, sim_diff = evaluate_metrics(emb1, emb2, self.issame)
            print(f"  > Mức độ sáng {lvl:4d} | Acc (Cố định): {acc:.4f} | EER: {eer:.4f} | Sim Same: {sim_same:.3f} | Sim Diff: {sim_diff:.3f}")
            results.append({'Level': lvl, 'Accuracy': acc, 'EER': eer, 'Sim_Same': sim_same, 'Sim_Diff': sim_diff})
            
        df = pd.DataFrame(results)
        df.to_csv(f'{self.results_dir}/brightness_results.csv', index=False)
        
        # Vẽ biểu đồ
        plt.figure(figsize=(10, 5))
        plt.plot(df['Level'], df['Accuracy'], marker='o', color='orange', label='Accuracy (fixed threshold 1.60)')
        plt.plot(df['Level'], 1 - df['EER'], marker='s', color='blue', label='1 - EER (intrinsic power)')
        plt.title('Độ chính xác theo Cường độ Ánh sáng')
        plt.xlabel('Độ sáng tăng thêm (Beta Offset)')
        plt.ylabel('Tỉ lệ chính xác')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{self.results_dir}/brightness_impact.png')
        plt.close()
        return df

    def test_blur(self):
        print("\n=== Đang chạy kiểm thử độ mờ nhòe (Blur/Motion) ===")
        levels = [0, 3, 5, 7, 9, 11, 15]
        results = []
        
        for lvl in levels:
            perturbed = []
            for t in self.data_list:
                img_cv = tensor_to_cv2(t)
                perturbed_img = apply_blur(img_cv, lvl)
                perturbed.append(cv2_to_tensor(perturbed_img))
                
            emb1, emb2 = self.run_inference(perturbed)
            acc, eer, sim_same, sim_diff = evaluate_metrics(emb1, emb2, self.issame)
            print(f"  > Kernel Blur {lvl:2d}x{lvl:2d} | Acc (Cố định): {acc:.4f} | EER: {eer:.4f} | Sim Same: {sim_same:.3f} | Sim Diff: {sim_diff:.3f}")
            results.append({'Level': lvl, 'Accuracy': acc, 'EER': eer, 'Sim_Same': sim_same, 'Sim_Diff': sim_diff})
            
        df = pd.DataFrame(results)
        df.to_csv(f'{self.results_dir}/blur_results.csv', index=False)
        
        # Vẽ biểu đồ
        plt.figure(figsize=(10, 5))
        plt.plot(df['Level'], df['Accuracy'], marker='o', color='red', label='Accuracy (fixed threshold 1.60)')
        plt.plot(df['Level'], 1 - df['EER'], marker='s', color='blue', label='1 - EER (intrinsic power)')
        plt.title('Độ chính xác theo Độ mờ (Gaussian Blur)')
        plt.xlabel('Kích thước Kernel Blur')
        plt.ylabel('Tỉ lệ chính xác')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{self.results_dir}/blur_impact.png')
        plt.close()
        return df

    def test_resolution(self):
        print("\n=== Đang chạy kiểm thử độ phân giải (Resolution degradation) ===")
        levels = [112, 84, 56, 42, 28, 14]
        results = []
        
        for lvl in levels:
            perturbed = []
            for t in self.data_list:
                img_cv = tensor_to_cv2(t)
                perturbed_img = apply_resolution(img_cv, lvl)
                perturbed.append(cv2_to_tensor(perturbed_img))
                
            emb1, emb2 = self.run_inference(perturbed)
            acc, eer, sim_same, sim_diff = evaluate_metrics(emb1, emb2, self.issame)
            print(f"  > Kích thước {lvl:3d}x{lvl:3d} | Acc (Cố định): {acc:.4f} | EER: {eer:.4f} | Sim Same: {sim_same:.3f} | Sim Diff: {sim_diff:.3f}")
            results.append({'Level': lvl, 'Accuracy': acc, 'EER': eer, 'Sim_Same': sim_same, 'Sim_Diff': sim_diff})
            
        df = pd.DataFrame(results)
        df.to_csv(f'{self.results_dir}/resolution_results.csv', index=False)
        
        # Vẽ biểu đồ
        plt.figure(figsize=(10, 5))
        plt.plot(df['Level'], df['Accuracy'], marker='o', color='green', label='Accuracy (fixed threshold 1.60)')
        plt.plot(df['Level'], 1 - df['EER'], marker='s', color='blue', label='1 - EER (intrinsic power)')
        # Đảo ngược trục X để thể hiện chất lượng giảm dần từ trái qua phải
        plt.gca().invert_xaxis()
        plt.title('Độ chính xác theo Độ phân giải hình ảnh')
        plt.xlabel('Kích thước ảnh sau giảm độ phân giải (pixels)')
        plt.ylabel('Tỉ lệ chính xác')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{self.results_dir}/resolution_impact.png')
        plt.close()
        return df

    def test_rotation(self):
        print("\n=== Đang chạy kiểm thử độ xoay góc ảnh (Rotation sensitivity) ===")
        levels = [0, 5, 10, 15, 20, 30, 45]
        results = []
        
        for lvl in levels:
            perturbed = []
            for t in self.data_list:
                img_cv = tensor_to_cv2(t)
                perturbed_img = apply_rotation(img_cv, lvl)
                perturbed.append(cv2_to_tensor(perturbed_img))
                
            emb1, emb2 = self.run_inference(perturbed)
            acc, eer, sim_same, sim_diff = evaluate_metrics(emb1, emb2, self.issame)
            print(f"  > Góc xoay {lvl:2d}° | Acc (Cố định): {acc:.4f} | EER: {eer:.4f} | Sim Same: {sim_same:.3f} | Sim Diff: {sim_diff:.3f}")
            results.append({'Level': lvl, 'Accuracy': acc, 'EER': eer, 'Sim_Same': sim_same, 'Sim_Diff': sim_diff})
            
        df = pd.DataFrame(results)
        df.to_csv(f'{self.results_dir}/rotation_results.csv', index=False)
        
        # Vẽ biểu đồ
        plt.figure(figsize=(10, 5))
        plt.plot(df['Level'], df['Accuracy'], marker='o', color='purple', label='Accuracy (fixed threshold 1.60)')
        plt.plot(df['Level'], 1 - df['EER'], marker='s', color='blue', label='1 - EER (intrinsic power)')
        plt.title('Độ chính xác theo Góc nghiêng xoay khuôn mặt')
        plt.xlabel('Góc nghiêng xoay (Độ)')
        plt.ylabel('Tỉ lệ chính xác')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{self.results_dir}/rotation_impact.png')
        plt.close()
        return df

    def test_occlusion(self):
        print("\n=== Đang chạy kiểm thử độ che khuất (Occlusion/Masks) ===")
        # Không che khuất, che mắt (kính mát), che miệng (khẩu trang)
        types = ['clean', 'glasses', 'mask']
        results = []
        
        for t_type in types:
            perturbed = []
            for t in self.data_list:
                img_cv = tensor_to_cv2(t)
                if t_type == 'clean':
                    perturbed_img = img_cv
                else:
                    perturbed_img = apply_occlusion(img_cv, t_type)
                perturbed.append(cv2_to_tensor(perturbed_img))
                
            emb1, emb2 = self.run_inference(perturbed)
            acc, eer, sim_same, sim_diff = evaluate_metrics(emb1, emb2, self.issame)
            print(f"  > Loại che khuất: {t_type:8s} | Acc (Cố định): {acc:.4f} | EER: {eer:.4f} | Sim Same: {sim_same:.3f} | Sim Diff: {sim_diff:.3f}")
            results.append({'Occlusion': t_type, 'Accuracy': acc, 'EER': eer, 'Sim_Same': sim_same, 'Sim_Diff': sim_diff})
            
        df = pd.DataFrame(results)
        df.to_csv(f'{self.results_dir}/occlusion_results.csv', index=False)
        
        # Vẽ biểu đồ cột
        plt.figure(figsize=(8, 5))
        sns.barplot(x='Occlusion', y='Accuracy', data=df, hue='Occlusion', palette='muted', legend=False)
        plt.ylim(0.5, 1.05)
        plt.title('Độ chính xác dưới các trạng thái Che khuất')
        plt.ylabel('Tỉ lệ chính xác')
        plt.grid(axis='y')
        plt.savefig(f'{self.results_dir}/occlusion_impact.png')
        plt.close()
        return df

    def generate_markdown_report(self, brightness_df, blur_df, res_df, rot_df, occ_df):
        report_path = f"{self.results_dir}/perturbation_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Báo cáo Chi tiết Kiểm thử Căng thẳng Nhiễu môi trường\n\n")
            f.write("Báo cáo này lưu trữ chi tiết khả năng nhận diện của mô hình ArcFace `backbone_best.pth` dưới các tác động ngoại cảnh bất lợi.\n\n")
            
            f.write("## 1. Kiểm thử Ánh sáng (Brightness)\n")
            f.write(brightness_df.to_markdown(index=False) + "\n\n")
            
            f.write("## 2. Kiểm thử Mờ nhòe (Blur)\n")
            f.write(blur_df.to_markdown(index=False) + "\n\n")
            
            f.write("## 3. Kiểm thử Giảm Độ Phân Giải (Resolution)\n")
            f.write(res_df.to_markdown(index=False) + "\n\n")
            
            f.write("## 4. Kiểm thử Góc Xoay Nghiêng (Rotation)\n")
            f.write(rot_df.to_markdown(index=False) + "\n\n")
            
            f.write("## 5. Kiểm thử Che khuất (Occlusion)\n")
            f.write(occ_df.to_markdown(index=False) + "\n\n")
            
        print(f"\n--- Đã xuất báo cáo tổng quan ra file: {report_path} ---")

if __name__ == '__main__':
    runner = StressTestRunner()
    b_df = runner.test_brightness()
    bl_df = runner.test_blur()
    r_df = runner.test_resolution()
    rot_df = runner.test_rotation()
    o_df = runner.test_occlusion()
    runner.generate_markdown_report(b_df, bl_df, r_df, rot_df, o_df)
