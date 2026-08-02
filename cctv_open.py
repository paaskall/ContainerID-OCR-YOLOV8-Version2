import cv2
import datetime
import os
import time
from pathlib import Path

class CCTV4HourRecorder:
    def __init__(self, rtsp_url, output_dir="rekaman_4jam", segment_duration=300):
        """
        Args:
            rtsp_url: URL RTSP stream
            output_dir: Direktori penyimpanan
            segment_duration: Durasi setiap segmen dalam detik (default: 300 = 5 menit)
        """
        self.rtsp_url = rtsp_url
        self.output_dir = output_dir
        self.segment_duration = segment_duration
        self.is_recording = False
        
        # Buat direktori dengan timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(output_dir, f"recording_{timestamp}")
        Path(self.session_dir).mkdir(parents=True, exist_ok=True)
        
    def _record_segment(self, output_file, duration):
        """Rekam satu segmen video"""
        cap = cv2.VideoCapture(self.rtsp_url)
        
        if not cap.isOpened():
            print(f"Error: Tidak dapat terhubung ke RTSP stream")
            return False
        
        # Dapatkan properti video
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        start_time = time.time()
        frame_count = 0
        last_progress = start_time
        
        while (time.time() - start_time) < duration and self.is_recording:
            ret, frame = cap.read()
            
            if not ret:
                print(f"Warning: Gagal membaca frame, mencoba reconnect...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(self.rtsp_url)
                continue
            
            # Tambahkan timestamp pada frame (opsional)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            out.write(frame)
            frame_count += 1
            
            # Progress setiap 30 detik
            if time.time() - last_progress >= 30:
                elapsed = time.time() - start_time
                remaining = duration - elapsed
                print(f"  Progress: {elapsed/60:.1f}/{duration/60:.1f} menit | "
                      f"Frame: {frame_count} | Sisa: {remaining/60:.1f} menit")
                last_progress = time.time()
        
        cap.release()
        out.release()
        
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  ✓ Segmen selesai: {os.path.basename(output_file)} "
              f"({file_size_mb:.1f} MB, {frame_count} frames)")
        return True
    
    def record_4_hours(self):
        """Rekam selama 4 jam"""
        self.is_recording = True
        start_time = datetime.datetime.now()
        end_time = start_time + datetime.timedelta(hours=4)
        
        # Hitung jumlah segmen
        total_segments = int(4 * 3600 / self.segment_duration)
        
        print("="*60)
        print(f"REKAMAN 4 JAM CCTV")
        print(f"Mulai: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Selesai: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Durasi per segmen: {self.segment_duration // 60} menit")
        print(f"Jumlah segmen: {total_segments}")
        print(f"Lokasi penyimpanan: {self.session_dir}")
        print("="*60)
        
        # Estimasi ukuran file
        estimated_size_mb = 4 * 1000  # Asumsi ~1GB per jam untuk 1080p
        print(f"Estimasi kebutuhan penyimpanan: ~{estimated_size_mb:.0f} MB ({estimated_size_mb/1024:.1f} GB)")
        print("="*60)
        
        segment_count = 0
        
        try:
            while self.is_recording and datetime.datetime.now() < end_time:
                # Buat nama file dengan timestamp
                timestamp = datetime.datetime.now().strftime("%H%M%S")
                output_file = os.path.join(self.session_dir, f"segment_{segment_count+1:03d}_{timestamp}.mp4")
                
                print(f"\n[Segmen {segment_count + 1}/{total_segments}] Merekam...")
                success = self._record_segment(output_file, self.segment_duration)
                
                if success:
                    segment_count += 1
                else:
                    print("  Gagal merekam segmen, mencoba lagi...")
                    time.sleep(5)
                
                # Hitung dan tampilkan progress keseluruhan
                elapsed = (datetime.datetime.now() - start_time).total_seconds()
                progress = (elapsed / (4 * 3600)) * 100
                remaining = end_time - datetime.datetime.now()
                
                print(f"\n📊 Progress Keseluruhan: {progress:.1f}% | "
                      f"Segmen: {segment_count}/{total_segments} | "
                      f"Sisa waktu: {str(remaining).split('.')[0]}")
                
        except KeyboardInterrupt:
            print("\n⚠️ Rekaman dihentikan oleh user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            self.stop_recording(segment_count, start_time)
    
    def stop_recording(self, segment_count, start_time):
        """Hentikan rekaman dan tampilkan ringkasan"""
        self.is_recording = False
        end_time = datetime.datetime.now()
        actual_duration = end_time - start_time
        
        print("\n" + "="*60)
        print(f"REKAMAN SELESAI")
        print(f"Durasi aktual: {str(actual_duration).split('.')[0]}")
        print(f"Segmen terekam: {segment_count}")
        print(f"Lokasi: {self.session_dir}")
        
        # Hitung total ukuran
        total_size = 0
        for file in Path(self.session_dir).glob("*.mp4"):
            total_size += file.stat().st_size
        total_size_mb = total_size / (1024 * 1024)
        print(f"Total ukuran: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
        print("="*60)

# Penggunaan
if __name__ == "__main__":
    rtsp_url = "rtsp://admin:Qwerty123@10.35.61.115/video"
    
    # Inisialisasi recorder untuk 4 jam
    recorder = CCTV4HourRecorder(
        rtsp_url, 
        output_dir="rekaman_cctv",
        segment_duration=300  # 5 menit per file (bisa diubah ke 600 untuk 10 menit)
    )
    
    # Mulai rekaman 4 jam
    recorder.record_4_hours()