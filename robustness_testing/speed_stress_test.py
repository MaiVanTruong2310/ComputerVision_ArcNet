import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import threading
from queue import Queue

# Thêm thư mục src vào path (sử dụng abspath để chạy được từ mọi vị trí)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, 'src'))

from model import get_backbone
from config import Config
import io

# Đảm bảo in ra được tiếng Việt trên Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

class SpeedTester:
    def __init__(self, checkpoint_path=None):
        # Đảm bảo đường dẫn tuyệt đối chính xác
        if checkpoint_path is None:
            self.checkpoint_path = os.path.join(ROOT_DIR, 'checkpoints', 'backbone_best.pth')
        else:
            self.checkpoint_path = checkpoint_path
            
        self.results_dir = os.path.join(ROOT_DIR, 'robustness_testing', 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Kiểm tra CUDA
        self.has_cuda = torch.cuda.is_available()
        self.devices_to_test = ['cpu']
        if self.has_cuda:
            self.devices_to_test.append('cuda')
            
        print(f"Thiết bị khả dụng để kiểm thử tốc độ: {self.devices_to_test}")

    def load_model_on_device(self, device_name):
        device = torch.device(device_name)
        model = get_backbone(Config.backbone_name, Config.embedding_size).to(device)
        try:
            state_dict = torch.load(self.checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Lỗi nạp mô hình trên {device_name}: {e}")
        model.eval()
        return model

    def run_latency_test(self, device_name, batch_size=1, num_runs=200):
        device = torch.device(device_name)
        model = self.load_model_on_device(device_name)
        
        # Chuẩn bị dummy input chuẩn ArcFace (112x112, 3 kênh màu)
        dummy_input = torch.randn(batch_size, 3, 112, 112).to(device)
        
        # 1. Đo Cold Start (Lần chạy đầu tiên)
        t_start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        if device_name == 'cuda':
            torch.cuda.synchronize()
        cold_start_time = (time.perf_counter() - t_start) * 1000  # ms
        
        # 2. Đo Warm Start (Các lần chạy sau)
        latencies = []
        for _ in range(num_runs):
            t_start = time.perf_counter()
            with torch.no_grad():
                _ = model(dummy_input)
            if device_name == 'cuda':
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t_start) * 1000) # ms
            
        # Thống kê
        latencies = np.array(latencies)
        mean_lat = np.mean(latencies)
        median_lat = np.median(latencies)
        p95_lat = np.percentile(latencies, 95)
        p99_lat = np.percentile(latencies, 99)
        fps = (batch_size * 1000) / mean_lat
        
        return {
            'Device': device_name,
            'Batch_Size': batch_size,
            'Cold_Start_ms': cold_start_time,
            'Mean_ms': mean_lat,
            'Median_ms': median_lat,
            'P95_ms': p95_lat,
            'P99_ms': p99_lat,
            'FPS': fps
        }

    def run_concurrency_stress_test(self, device_name, num_threads=4, total_requests=100):
        """Mô phỏng đa luồng gửi yêu cầu đồng thời để ép tải mô hình"""
        device = torch.device(device_name)
        model = self.load_model_on_device(device_name)
        
        input_queue = Queue()
        output_queue = Queue()
        
        # Nạp các request vào queue
        for i in range(total_requests):
            input_queue.put(torch.randn(1, 3, 112, 112).to(device))
            
        latencies = []
        lock = threading.Lock()
        
        def worker():
            while not input_queue.empty():
                try:
                    req_input = input_queue.get_nowait()
                except:
                    break
                
                t_start = time.perf_counter()
                with torch.no_grad():
                    _ = model(req_input)
                if device_name == 'cuda':
                    torch.cuda.synchronize()
                dur = (time.perf_counter() - t_start) * 1000
                
                with lock:
                    latencies.append(dur)
                input_queue.task_done()
                
        # Khởi chạy luồng
        threads = []
        start_time = time.perf_counter()
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)
            
        for t in threads:
            t.join()
            
        total_time = (time.perf_counter() - start_time) * 1000
        latencies = np.array(latencies)
        
        # Tính toán mức tiêu thụ bộ nhớ hiện tại
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024)
        
        vram_mb = 0
        if device_name == 'cuda':
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
        return {
            'Device': device_name,
            'Threads': num_threads,
            'Total_Time_ms': total_time,
            'Throughput_RPS': total_requests / (total_time / 1000),
            'Mean_Latency_ms': np.mean(latencies),
            'P95_Latency_ms': np.percentile(latencies, 95),
            'RAM_Usage_MB': ram_mb,
            'VRAM_Usage_MB': vram_mb,
            'Raw_Latencies': latencies
        }

    def execute_all_speed_tests(self):
        # 1. Kiểm thử Latency theo Batch Size
        batch_sizes = [1, 2, 4, 8, 16]
        latency_results = []
        
        print("\n=== Đang chạy kiểm thử Latency & FPS theo Batch Size ===")
        for dev in self.devices_to_test:
            for bs in batch_sizes:
                res = self.run_latency_test(dev, bs)
                print(f"  > [{dev.upper()}] Batch Size {bs:2d} | Latency Mean: {res['Mean_ms']:.2f}ms | Throughput: {res['FPS']:.1f} FPS")
                latency_results.append(res)
                
        df_latency = pd.DataFrame(latency_results)
        df_latency.to_csv(f'{self.results_dir}/batch_speed_results.csv', index=False)

        # Vẽ biểu đồ so sánh FPS giữa CPU và GPU
        plt.figure(figsize=(10, 5))
        for dev in self.devices_to_test:
            dev_df = df_latency[df_latency['Device'] == dev]
            plt.plot(dev_df['Batch_Size'], dev_df['FPS'], marker='o', label=f'{dev.upper()} FPS')
        plt.title('Thông lượng xử lý (FPS) theo Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('FPS (Khung hình / Giây)')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{self.results_dir}/fps_vs_batch_size.png')
        plt.close()

        # 2. Kiểm thử Ép tải Đa luồng (Concurrency Stress Test)
        print("\n=== Đang chạy kiểm thử Ép tải Đa luồng (Concurrency Stress) ===")
        concurrency_results = []
        raw_latencies_dict = {}
        
        for dev in self.devices_to_test:
            # Test với 1, 2, 4, 8 luồng song song
            for th in [1, 2, 4, 8]:
                res = self.run_concurrency_stress_test(dev, num_threads=th)
                print(f"  > [{dev.upper()}] Luồng song song: {th} | RPS: {res['Throughput_RPS']:.1f} | Mean Latency: {res['Mean_Latency_ms']:.2f}ms | RAM: {res['RAM_Usage_MB']:.1f}MB")
                
                # Lưu raw latencies cho luồng tối đa (8 luồng)
                if th == 8:
                    raw_latencies_dict[dev] = res['Raw_Latencies']
                    
                # Loại bỏ danh sách raw latency ra khỏi kết quả bảng biểu trước khi lưu
                summary_res = {k: v for k, v in res.items() if k != 'Raw_Latencies'}
                concurrency_results.append(summary_res)
                
        df_concurrency = pd.DataFrame(concurrency_results)
        df_concurrency.to_csv(f'{self.results_dir}/concurrency_stress_results.csv', index=False)

        # Vẽ biểu đồ phân phối độ trễ (Latency Distribution) dưới tải nặng (8 luồng song song)
        if raw_latencies_dict:
            plt.figure(figsize=(10, 6))
            for dev, lats in raw_latencies_dict.items():
                sns.histplot(lats, kde=True, label=f'{dev.upper()} (8 luồng song song)', stat='probability', alpha=0.6)
            plt.title('Biểu đồ phân bổ Độ trễ (Latency Distribution) dưới Tải Đa luồng')
            plt.xlabel('Độ trễ xử lý (ms)')
            plt.ylabel('Xác suất')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{self.results_dir}/latency_stress_distribution.png')
            plt.close()

        # 3. Tạo báo cáo Markdown tổng kết
        self.generate_speed_report(df_latency, df_concurrency)

    def generate_speed_report(self, df_latency, df_concurrency):
        report_path = f"{self.results_dir}/speed_stress_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Báo cáo Hiệu năng & Tốc độ Xử lý (Speed & Latency Stress Report)\n\n")
            f.write("Báo cáo đo đạc khả năng đáp ứng tốc độ và mức độ tiêu thụ bộ nhớ của mô hình ArcFace.\n\n")
            
            f.write("## 1. Thời gian phản hồi và FPS theo Batch Size\n")
            f.write("Bảng đo thời gian trễ của 1 lần suy diễn (Latency) và số lượng ảnh xử lý được trên giây (FPS).\n\n")
            f.write(df_latency.to_markdown(index=False) + "\n\n")
            
            f.write("## 2. Kiểm thử Ép tải Đa luồng đồng thời (Concurrency Test)\n")
            f.write("Mô phỏng khả năng xử lý của mô hình dưới áp lực nhiều yêu cầu gửi cùng lúc (ví dụ chạy trên máy chủ API hoặc các luồng camera song song).\n\n")
            f.write(df_concurrency.to_markdown(index=False) + "\n\n")
            
            # Gợi ý tối ưu hóa triển khai
            f.write("## 3. Đánh giá Khả năng triển khai thực tế trên Di động (PassNote)\n")
            # Trích xuất Latency CPU Batch Size 1
            cpu_bs1 = df_latency[(df_latency['Device'] == 'cpu') & (df_latency['Batch_Size'] == 1)].iloc[0]
            f.write(f"- **Độ trễ xử lý trên CPU (Single Frame)**: `{cpu_bs1['Mean_ms']:.2f} ms` (Tương đương `{cpu_bs1['FPS']:.1f} FPS`).\n")
            f.write(f"- **Khởi động lạnh (Cold Start)** trên CPU tốn `{cpu_bs1['Cold_Start_ms']:.2f} ms` để nạp các thư viện toán học CUDA/MKL lần đầu tiên.\n")
            
            if cpu_bs1['Mean_ms'] > 100:
                f.write("> [!WARNING]\n")
                f.write(f"> Độ trễ trên CPU hiện tại là `{cpu_bs1['Mean_ms']:.2f} ms`, khá cao đối với camera thời gian thực (>10FPS cần dưới 100ms).\n")
                f.write("> Khuyến nghị chuyển đổi sang định dạng **ONNX Runtime với OpenVINO/DirectML** hoặc **TFLite (float16/int8)** để tăng tốc độ trên thiết bị di động.\n")
            else:
                f.write("> [!NOTE]\n")
                f.write(f"> Độ trễ trên CPU đạt mức tốt (`{cpu_bs1['Mean_ms']:.2f} ms`), đáp ứng mượt mà trải nghiệm quét camera cục bộ.\n")
                
        print(f"\n--- Đã xuất báo cáo hiệu năng ra file: {report_path} ---")

if __name__ == '__main__':
    tester = SpeedTester()
    tester.execute_all_speed_tests()
