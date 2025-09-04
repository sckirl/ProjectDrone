import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import argparse
import os

class CoinCounter:
    def __init__(self, model_path='ProjectDrone/coinFall150.pt', conf_threshold=0.5):
        """
        Inisialisasi model YOLOv10 untuk penghitungan koin
        
        Args:
            model_path: Path ke model YOLOv10 (default: coinFall150.pt)
            conf_threshold: Threshold confidence untuk deteksi
        """
        # Periksa apakah model ada
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")
        
        # Load model YOLOv10
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Inisialisasi counter
        self.total_coins_counted = 0  # Total koin yang pernah dihitung
        self.current_coins = 0        # Koin yang terdeteksi di frame saat ini
        
        # Untuk melacak koin yang sudah dihitung
        self.counted_coins = set()    # Menyimpan ID koin yang sudah dihitung
        self.next_coin_id = 1         # ID untuk koin berikutnya
        
        # History untuk smoothing dan tracking
        self.detection_history = deque(maxlen=10)
        self.previous_positions = {}  # Posisi koin sebelumnya untuk tracking
        
        # Nama kelas berdasarkan data.yaml
        self.class_names = {0: 'COIN', 1: 'COIN_JATUH'}
        
        # Warna untuk bounding box
        self.colors = {
            'COIN': (0, 255, 0),        # Hijau untuk koin diam
            'COIN_JATUH': (0, 0, 255)   # Merah untuk koin jatuh
        }
        
        print(f"Model YOLOv10 loaded from {model_path}")
        print("Penghitung Koin Siap - Setiap koin terdeteksi akan ditambah +1")
    
    def assign_coin_ids(self, current_detections):
        """
        Memberikan ID pada koin yang terdeteksi dan melacak pergerakannya
        """
        current_coin_ids = []
        updated_positions = {}
        
        for detection in current_detections:
            x1, y1, x2, y2, confidence, class_id = detection
            cx = (x1 + x2) / 2  # Center x
            cy = (y1 + y2) / 2  # Center y
            
            # Cari koin terdekat dari frame sebelumnya
            closest_coin_id = None
            min_distance = float('inf')
            
            for coin_id, prev_pos in self.previous_positions.items():
                distance = np.sqrt((cx - prev_pos[0])**2 + (cy - prev_pos[1])**2)
                if distance < min_distance and distance < 50:  # Threshold jarak
                    min_distance = distance
                    closest_coin_id = coin_id
            
            if closest_coin_id is not None:
                # Gunakan ID yang sudah ada
                coin_id = closest_coin_id
                current_coin_ids.append(coin_id)
                updated_positions[coin_id] = (cx, cy)
            else:
                # Koin baru, beri ID baru dan tambahkan ke counter
                coin_id = self.next_coin_id
                self.next_coin_id += 1
                current_coin_ids.append(coin_id)
                updated_positions[coin_id] = (cx, cy)
                
                # Jika koin ini belum dihitung, tambahkan ke counter
                if coin_id not in self.counted_coins:
                    self.counted_coins.add(coin_id)
                    self.total_coins_counted += 1
                    print(f"✅ Koin terdeteksi! Total: {self.total_coins_counted}")
        
        # Update previous positions untuk frame berikutnya
        self.previous_positions = updated_positions
        self.current_coins = len(current_detections)
        
        return current_coin_ids
    
    def detect_and_count(self, frame):
        """
        Mendeteksi koin dalam frame dan menghitung jumlahnya
        
        Args:
            frame: Frame gambar dari kamera/video
            
        Returns:
            Frame dengan bounding box dan informasi penghitungan
        """
        # Lakukan inferensi
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        current_detections = []
        
        # Proses hasil deteksi
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    # Ambil informasi bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    
                    # Simpan deteksi
                    current_detections.append((x1, y1, x2, y2, confidence, class_id))
        
        # Beri ID pada koin dan hitung yang baru
        if current_detections:
            coin_ids = self.assign_coin_ids(current_detections)
        
        # Gambar bounding box dan informasi
        for i, detection in enumerate(current_detections):
            x1, y1, x2, y2, confidence, class_id = detection
            
            # Dapatkan nama kelas
            class_name = self.class_names.get(class_id, f"Class_{class_id}")
            
            # Gambar bounding box
            color = self.colors.get(class_name, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Tambahkan label dengan ID koin
            if i < len(coin_ids):
                label = f"ID:{coin_ids[i]} {class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Tambahkan informasi ke frame
        self.add_info_to_frame(frame)
        
        return frame
    
    def add_info_to_frame(self, frame):
        """Menambahkan informasi penghitungan ke frame"""
        # Background untuk teks
        y_start = 10
        cv2.rectangle(frame, (5, y_start), (350, y_start + 90), (0, 0, 0), -1)
        
        # Teks informasi
        cv2.putText(frame, f"KOIN SAAT INI: {self.current_coins}", (10, y_start + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"TOTAL DIHITUNG: {self.total_coins_counted}", (10, y_start + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Status
        status = "STATUS: Setiap koin terdeteksi = +1"
        cv2.putText(frame, status, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    def reset_counter(self):
        """Reset penghitung"""
        self.total_coins_counted = 0
        self.counted_coins.clear()
        self.previous_positions = {}
        self.next_coin_id = 1
        print("Penghitung telah direset ke 0")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Penghitung Koin dengan YOLOv10')
    parser.add_argument('--model', type=str, default='coinFall150.pt', help='Path ke model weights')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam)')
    args = parser.parse_args()
    
    # Inisialisasi detector
    try:
        coin_counter = CoinCounter(args.model, args.conf)
    except FileNotFoundError as e:
        print(e)
        print("Pastikan file model 'coinFall150.pt' ada di folder yang sama")
        return
    
    # Buka video source
    if args.source == '0':
        source = 0  # Webcam default
    else:
        source = args.source
    
    cap = cv2.VideoCapture(source)
    
    # Atur resolusi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("=" * 50)
    print("PENGHITUNG KOIN OTOMATIS")
    print("=" * 50)
    print("Cara kerja:")
    print("- Setiap koin yang terdeteksi (COIN atau COIN_JATUH) akan ditambah +1")
    print("- Sistem menggunakan tracking untuk menghindari penghitungan ganda")
    print("- Koin yang sama tidak akan dihitung dua kali")
    print("=" * 50)
    print("Kontrol:")
    print("Tekan 'q' untuk keluar")
    print("Tekan 'r' untuk reset penghitung")
    print("Tekan 's' untuk simpan screenshot")
    print("=" * 50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame")
            break
        
        # Flip frame untuk webcam
        if args.source == '0':
            frame = cv2.flip(frame, 1)
        
        # Deteksi dan hitung koin
        processed_frame = coin_counter.detect_and_count(frame)
        
        # Tampilkan frame
        cv2.imshow('Penghitung Koin Otomatis - coinFall150.pt', processed_frame)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            coin_counter.reset_counter()
        elif key == ord('s'):
            # Simpan screenshot
            filename = f"coin_count_{int(time.time())}.jpg"
            cv2.imwrite(filename, processed_frame)
            print(f"Screenshot disimpan sebagai {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nProgram dihentikan. Total koin yang dihitung: {coin_counter.total_coins_counted}")

if __name__ == "__main__":
    main()