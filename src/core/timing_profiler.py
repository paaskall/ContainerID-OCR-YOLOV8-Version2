"""
timing_profiler.py — Profiler waktu pemrosesan untuk skripsi

Mengukur dan menyimpan waktu eksekusi komponen utama AutoGate:
  - YOLO inference
  - Deteksi bbox (jumlah dan ukuran)
  - OCR per engine (Gemini, EasyOCR, CharYOLO)
  - Total waktu per frame

Output: CSV di folder yang ditentukan, satu baris per event OCR.

Kolom CSV:
  timestamp, session_id, camera_id, frame_id,
  detection_type, bbox_w, bbox_h, bbox_area,
  yolo_ms, ocr_engine, ocr_ms, ocr_text, ocr_success,
  total_frame_ms
"""

import os
import csv
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict


# ── Path default output ───────────────────────────────────────────────────────
_DEFAULT_CSV_DIR = "/home/remote-user/debug_ocr/g2/timing"


# ══════════════════════════════════════════════════════════════════════════════
# TimingRecord — satu record per OCR event
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TimingRecord:
    timestamp:       str
    session_id:      Optional[int]
    camera_id:       int
    frame_id:        int
    detection_type:  str
    bbox_w:          int
    bbox_h:          int
    bbox_area:       int
    yolo_ms:         float   # waktu YOLO predict untuk frame ini
    ocr_engine:      str     # "gemini" / "easyocr" / "char_yolo" / dll
    ocr_ms:          float   # waktu OCR untuk bbox ini
    ocr_text:        str
    ocr_success:     bool    # apakah terbaca (bukan TIDAK_TERBACA)
    total_frame_ms:  float   # total waktu process_frame()


# ══════════════════════════════════════════════════════════════════════════════
# TimingProfiler
# ══════════════════════════════════════════════════════════════════════════════

class TimingProfiler:
    """
    Singleton profiler — diakses via TimingProfiler.get_instance().

    Cara pakai:
        profiler = TimingProfiler.get_instance()

        # Ukur YOLO
        profiler.start("yolo")
        results = model.predict(...)
        yolo_ms = profiler.stop("yolo")

        # Ukur OCR
        profiler.start("ocr")
        text, engine = ocr.read(roi, label)
        ocr_ms = profiler.stop("ocr")

        # Simpan record
        profiler.record(
            camera_id=0, session_id=123,
            detection_type="Container_ID",
            bbox_w=400, bbox_h=80,
            yolo_ms=yolo_ms, ocr_engine=engine,
            ocr_ms=ocr_ms, ocr_text=text,
        )
    """

    _instance = None
    _lock      = threading.Lock()

    @classmethod
    def get_instance(cls, csv_dir: str = _DEFAULT_CSV_DIR) -> "TimingProfiler":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(csv_dir)
            return cls._instance

    def __init__(self, csv_dir: str = _DEFAULT_CSV_DIR):
        self.csv_dir  = csv_dir
        os.makedirs(csv_dir, exist_ok=True)

        # CSV file — satu file per hari
        self._csv_path   = None
        self._csv_file   = None
        self._csv_writer = None
        self._current_date = None
        self._file_lock  = threading.Lock()

        # Frame counter per kamera
        self._frame_counters: Dict[int, int] = defaultdict(int)
        self._fc_lock = threading.Lock()

        # Timer per thread (untuk nested timing)
        self._timers: Dict[str, float] = {}
        self._timer_lock = threading.Lock()

        # Statistik ringkasan in-memory
        self._stats: Dict[str, List[float]] = defaultdict(list)
        self._stats_lock = threading.Lock()

        # Flag aktif/nonaktif
        self.enabled = True

        print(f"[PROFILER] TimingProfiler aktif → CSV: {csv_dir}/")

    # ── Timer ─────────────────────────────────────────────────────────────────

    def start(self, key: str):
        """Mulai timer untuk key tertentu."""
        if not self.enabled:
            return
        with self._timer_lock:
            self._timers[key] = time.perf_counter()

    def stop(self, key: str) -> float:
        """
        Hentikan timer, kembalikan durasi dalam milidetik.
        Return 0.0 jika key tidak ditemukan.
        """
        if not self.enabled:
            return 0.0
        with self._timer_lock:
            t0 = self._timers.pop(key, None)
        if t0 is None:
            return 0.0
        ms = (time.perf_counter() - t0) * 1000.0
        with self._stats_lock:
            self._stats[key].append(ms)
        return ms

    # ── Frame counter ─────────────────────────────────────────────────────────

    def next_frame_id(self, camera_id: int) -> int:
        with self._fc_lock:
            self._frame_counters[camera_id] += 1
            return self._frame_counters[camera_id]

    # ── CSV writer ────────────────────────────────────────────────────────────

    def _get_writer(self):
        """Buka/rotate CSV file harian."""
        today = time.strftime("%Y-%m-%d")
        if today != self._current_date:
            if self._csv_file:
                self._csv_file.close()
            path = os.path.join(self.csv_dir, f"timing_{today}.csv")
            is_new = not os.path.exists(path)
            self._csv_file   = open(path, "a", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=[
                    "timestamp", "session_id", "camera_id", "frame_id",
                    "detection_type", "bbox_w", "bbox_h", "bbox_area",
                    "yolo_ms", "ocr_engine", "ocr_ms", "ocr_text",
                    "ocr_success", "total_frame_ms",
                ],
            )
            if is_new:
                self._csv_writer.writeheader()
            self._current_date = today
            self._csv_path     = path
        return self._csv_writer

    def record(
        self,
        camera_id:      int,
        frame_id:       int,
        detection_type: str,
        bbox_w:         int,
        bbox_h:         int,
        yolo_ms:        float,
        ocr_engine:     str,
        ocr_ms:         float,
        ocr_text:       str,
        session_id:     Optional[int] = None,
        total_frame_ms: float = 0.0,
    ):
        """Simpan satu record timing ke CSV."""
        if not self.enabled:
            return

        invalid = {"TIDAK_TERBACA", "ROI_INVALID", "ROI_TOO_SMALL", "", None}
        success = bool(ocr_text) and ocr_text not in invalid

        row = {
            "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id":     session_id or "",
            "camera_id":      camera_id,
            "frame_id":       frame_id,
            "detection_type": detection_type,
            "bbox_w":         bbox_w,
            "bbox_h":         bbox_h,
            "bbox_area":      bbox_w * bbox_h,
            "yolo_ms":        round(yolo_ms, 2),
            "ocr_engine":     ocr_engine,
            "ocr_ms":         round(ocr_ms, 2),
            "ocr_text":       ocr_text or "",
            "ocr_success":    1 if success else 0,
            "total_frame_ms": round(total_frame_ms, 2),
        }

        with self._file_lock:
            try:
                writer = self._get_writer()
                writer.writerow(row)
                self._csv_file.flush()
            except Exception as e:
                print(f"[PROFILER] CSV write error: {e}")

    # ── Statistik ringkasan ───────────────────────────────────────────────────

    def print_summary(self):
        """Cetak statistik min/avg/max ke terminal."""
        with self._stats_lock:
            if not self._stats:
                print("[PROFILER] Belum ada data timing.")
                return
            print("\n" + "="*55)
            print(" TIMING SUMMARY (ms)")
            print("="*55)
            print(f"{'Key':<20} {'N':>5} {'Min':>8} {'Avg':>8} {'Max':>8}")
            print("-"*55)
            for key, vals in sorted(self._stats.items()):
                n   = len(vals)
                mn  = min(vals)
                avg = sum(vals) / n
                mx  = max(vals)
                print(f"{key:<20} {n:>5} {mn:>8.1f} {avg:>8.1f} {mx:>8.1f}")
            print("="*55 + "\n")

    def get_summary_dict(self) -> Dict[str, Dict[str, float]]:
        """Return ringkasan sebagai dict untuk GUI."""
        with self._stats_lock:
            result = {}
            for key, vals in self._stats.items():
                if vals:
                    result[key] = {
                        "n":   len(vals),
                        "min": round(min(vals), 1),
                        "avg": round(sum(vals) / len(vals), 1),
                        "max": round(max(vals), 1),
                    }
            return result

    def reset_stats(self):
        """Reset statistik in-memory (tidak hapus CSV)."""
        with self._stats_lock:
            self._stats.clear()

    def close(self):
        """Tutup file CSV dengan bersih."""
        with self._file_lock:
            if self._csv_file:
                self._csv_file.close()
                self._csv_file = None
        print(f"[PROFILER] CSV disimpan di: {self._csv_path}")