import cv2
import os
from datetime import datetime

# rtsp_url = "rtsp://admin:Qwerty123@10.35.61.115/video"
rtsp_url = "rtsp://admin:Qwerty123@10.35.61.111/video"

# Buat output directory
output_dir = "captured_framesV2"  # Ganti dengan path yang diinginkan
# Contoh path absolut: output_dir = "/home/remote-user/AutoGate-G2/captured_frames"

# Buat directory jika belum ada
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Directory '{output_dir}' berhasil dibuat")
else:
    print(f"Directory '{output_dir}' sudah ada")

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Tidak dapat membuka URL RTSP")
    exit()

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Gagal membaca frame")
        break
    
    # Simpan frame setiap 30 frame
    if frame_count % 30 == 0:
        # Buat nama file dengan timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"frame_{frame_count}_{timestamp}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        cv2.imwrite(filepath, frame)
        saved_count += 1
        print(f"Frame {frame_count} tersimpan: {filepath}")
    
    frame_count += 1
    
    if frame_count > 30:  # Hentikan setelah 300 frame
        break

cap.release()
print(f"\nSelesai! Total {saved_count} frame tersimpan di directory: {output_dir}")