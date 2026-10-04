# ==============================================================================
# Tên file: recordio.py
# Công dụng: Bộ đọc file MXNet RecordIO (.rec, .idx) thuần Python (Pure Python Reader).
#            Thay thế hoàn toàn thư viện MXNet để đọc ảnh nhị phân và giải mã
#            header sinh trắc học của InsightFace trên các môi trường không cài được MXNet.
# ==============================================================================
import struct
import numpy as np
import os
from collections import namedtuple

# Định nghĩa Header của InsightFace/MXNet
IRHeader = namedtuple('IRHeader', ['flag', 'label', 'id', 'id2'])

class RecordIO:
    def __init__(self, path, mode='r'):
        self.path = path
        self.f = None 
        
    def _check_open(self):
        if self.f is None:
            self.f = open(self.path, 'rb')
            
    def close(self):
        if self.f:
            self.f.close()
            self.f = None
        
    def read(self):
        self._check_open()
        header = self.f.read(8)
        if not header or len(header) < 8:
            return None
        
        magic, length_flant = struct.unpack('<II', header)
        if (magic >> 16) != 0xced7:
            return None
            
        length = length_flant & 0x00ffffff
        data = self.f.read(length)
        
        # Bỏ qua phần padding (align 4 bytes)
        padding = (4 - (length % 4)) % 4
        if padding > 0:
            self.f.read(padding)
            
        return data

class IndexedRecordIO(RecordIO):
    """Bộ đọc file .rec có hỗ trợ seek qua file .idx"""
    def __init__(self, idx_path, rec_path, mode='r'):
        super().__init__(rec_path, mode)
        self.idx_path = idx_path
        self.idx = {}
        self.keys = []
        self._load_idx()
        
    def _load_idx(self):
        # Thử đọc dạng text (im2rec.py format)
        try:
            with open(self.idx_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            k, v = int(parts[0]), int(parts[1])
                            self.idx[k] = v
                        except ValueError:
                            continue
            if len(self.idx) > 0:
                self.keys = sorted(self.idx.keys())
                return
        except Exception:
            pass
            
        # Thử đọc dạng binary (16 bytes per record: key, offset)
        self.idx = {}
        if os.path.getsize(self.idx_path) % 16 == 0:
            with open(self.idx_path, 'rb') as f:
                while True:
                    data = f.read(16)
                    if not data or len(data) < 16: break
                    k, v = struct.unpack('<QQ', data)
                    self.idx[k] = v
            self.keys = sorted(self.idx.keys())

    def read_idx(self, key):
        if key not in self.idx:
            return None
        offset = self.idx[key]
        self._check_open()
        self.f.seek(offset)
        return self.read()

def unpack(s):
    """Giải mã record của InsightFace sang Header và Data"""
    # MXNet Header: fQQi (label, id1, id2, flag)
    # Sử dụng '=' để đảm bảo kích thước chuẩn 24 bytes (không bị padding của OS)
    header_size = 24
    if len(s) < header_size:
        return None, s
        
    label, id_val, id2, flag = struct.unpack('=fQQi', s[:header_size])
    payload = s[header_size:]
    
    if flag > 0:
        label_array = np.frombuffer(payload, dtype=np.float32, count=flag)
        data = payload[flag * 4:]
        return IRHeader(flag, label_array, id_val, id2), data
    else:
        return IRHeader(flag, label, id_val, id2), payload
