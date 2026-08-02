# AutoGate System

Sistem deteksi otomatis Container ID dan Truck ID berbasis kamera CCTV, OCR, dan jembatan timbang dirancang untuk otomatisasi gerbang pelabuhan kontainer

---

## Daftar Isi

- [Fitur Utama](#fitur-utama)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Struktur Project](#struktur-project)
- [Penjelasan Modul](#penjelasan-modul)
- [Persyaratan Sistem](#persyaratan-sistem)
- [Instalasi & Setup](#instalasi--setup)
- [Konfigurasi](#konfigurasi)
- [Cara Menjalankan](#cara-menjalankan)
- [Alur Kerja Sistem](#alur-kerja-sistem)

---

## Fitur Utama

- **Deteksi multi-kamera** — mendukung 6 kamera RTSP secara bersamaan
- **OCR tiga engine** — Gemini AI (primer), EasyOCR (fallback), dan Char-YOLO (khusus kontainer vertikal)
- **Validasi ISO 6346** — validasi dan auto-repair Container ID sesuai standar internasional
- **Integrasi jembatan timbang** — session timbangan otomatis berbasis HTTP polling
- **Session isolation** — mencegah data dari beberapa kendaraan tercampur dalam satu sesi
- **Identity-based auto-finalize** — mendeteksi pergantian truk dari owner code kontainer
- **Database MySQL** — menyimpan semua deteksi, sesi timbang, dan trip kontainer
- **UI PyQt5** — tampilan real-time dengan info per kamera

---

## Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────────┐
│                        AUTOGATE SYSTEM                          │
├─────────────────┬───────────────────────┬───────────────────────┤
│   INPUT LAYER   │    PROCESSING LAYER   │     OUTPUT LAYER      │
│                 │                       │                       │
│  6x RTSP CCTV  ─┼─▶  CameraManager      │                       │
│                 │        │              │                       │
│                 │        ▼              │                       │
│                 │   FrameWorker         │                       │
│                 │   (per kamera,        │                       │
│                 │    QThread)           │                       │
│                 │        │              │                       │
│                 │        ▼              │                       │
│                 │  PlateRecognizer      │                       │
│                 │  (YOLO detect)        │                       │
│                 │        │              │                       │
│                 │        ▼              │   ┌─────────────────┐ │
│  Jembatan       │    OCRRouter ─────────┼──▶│  MySQL Database │ │
│  Timbang  ──────┼─▶  ┌──────────┐       │   │  - weigh_sess.  │ │
│  (HTTP poll)    │    │  Gemini  │       │   │  - detections   │ │
│                 │    │  EasyOCR │       │   │  - containers   │ │
│                 │    │  CharYOLO│       │   │  - trips        │ │
│                 │    └──────────┘       │   └─────────────────┘ │
│                 │        │              │                       │
│                 │        ▼              │   ┌─────────────────┐ │
│                 │  SessionAggregator    │   │   PyQt5 GUI     │ │
│                 │  (ISO 6346 validate)  ├──▶│  - CameraWidget │ │
│                 │        │              │   │  - Live frame   │ │
│                 │        ▼              │   │  - OCR result   │ │
│                 │   ScaleInterface      │   └─────────────────┘ │
└─────────────────┴───────────────────────┴───────────────────────┘
```

### Alur Data Per Frame

```
RTSP Frame
    │
    ▼
CameraManager.get_frame()
    │
    ▼
FrameWorker._process_one_frame()   ← berjalan di QThread
    │
    ▼
PlateRecognizer.process_frame()
    │
    ├── YOLO detect → bbox per objek
    │
    └── (jika session aktif) → OCRRouter.read(roi, label)
              │
              ├── Gemini API  (primer)
              │       │ gagal/quota?
              ├── EasyOCR     (fallback umum)
              │       │ label = ContainerID_Vertikal?
              └── CharYOLO    (fallback vertikal)
                      │
                      ▼
              SessionAggregator.add_observation()
                      │
                      ├── ISO 6346 validate + repair
                      ├── Noise filter (gate display, timestamp)
                      ├── Identity tracking (owner code)
                      └── DB: weigh_session_detections
```

---

## Struktur Project

```
AutoGate/
├── main.py                          # Entry point aplikasi
├── requirements.txt
│
├── config/
│   ├── config.py                    # Konfigurasi umum (kamera, path model)
│   ├── database_config.py           # Konfigurasi koneksi MySQL
│   ├── gemini_config.py             # Konfigurasi & wrapper Gemini API
│   └── easyocr_config.py            # Parameter EasyOCR per tipe deteksi
│
├── models/
│   └── best.pt                      # Model YOLO utama (deteksi bbox)
│
├── src/
│   ├── core/
│   │   ├── camera_manager.py        # Manajemen koneksi RTSP multi-kamera
│   │   ├── plate_recognizer.py      # Orkestrasi YOLO + OCR + session
│   │   ├── ocr_router.py            # Router pemilihan engine OCR
│   │   ├── ocr_gemini.py            # Engine OCR Gemini (circuit breaker)
│   │   ├── ocr_easyocr.py           # Engine OCR EasyOCR + preprocessing
│   │   ├── ocr_char_yolo.py         # Engine OCR Char-YOLO (vertikal)
│   │   ├── iso_6346.py              # Validasi & repair Container ID
│   │   ├── session_aggregator.py    # Agregasi hasil OCR per sesi timbang
│   │   └── database_handler.py      # Handler operasi MySQL
│   │
│   ├── hardware/
│   │   └── scale_interface.py       # Interface polling jembatan timbang
│   │
│   └── gui/
│       ├── main_window.py           # Jendela utama PyQt5
│       ├── camera_widget.py         # Widget per kamera (video + hasil OCR)
│       └── styles.py                # Stylesheet QSS global
│
└── debug_ocr_router/                # Folder debug ROI (auto-generate)
    └── YYYY-MM-DD/
        └── session_<id>/
            └── <detection_type>/
                ├── *_RAW.jpg
                ├── *_PRE.jpg
                └── *.txt
```

---

## Penjelasan Modul

### `main.py`
Entry point aplikasi. Menginisialisasi `QApplication`, membuat `MainWindow`, dan mengonfigurasi Gemini API key.

---

### `config/`

| File | Deskripsi |

| `config.py` | Nama aplikasi, versi, daftar kamera, path model YOLO, pengaturan GUI |
| `database_config.py` | Host, port, user, password MySQL, nama semua tabel |
| `gemini_config.py` | Inisialisasi Gemini SDK, prompt OCR per tipe deteksi, parsing hasil response |
| `easyocr_config.py` | Parameter reader EasyOCR, allowlist karakter per kelas, aturan post-process |

---

### `src/core/`

#### `camera_manager.py`
Mengelola 6 koneksi RTSP secara bersamaan. Setiap kamera berjalan di thread terpisah. Fitur: auto-reconnect saat koneksi putus, buffer frame dengan `Queue(maxsize=1)` agar selalu mendapat frame terbaru.

#### `plate_recognizer.py`
Orkestrasi utama deteksi. Menjalankan YOLO untuk menemukan bounding box, lalu memanggil `OCRRouter` untuk setiap ROI. Memiliki sistem dedup per-bbox dan limit OCR per session untuk menghemat kuota Gemini.

Kelas penting:
- `OCR_CONF_THRESH` — threshold confidence YOLO minimum sebelum OCR dipanggil
- `SESSION_OCR_LIMIT` — batas maksimal panggilan OCR per kelas per session
- `DEDUP_GRID` — grid ukuran piksel untuk deduplikasi bbox yang bergeser kecil

#### `ocr_router.py`
Memilih engine OCR terbaik dan melakukan crosscheck antar engine.

```
Alur pemilihan engine:
1. Gemini disabled?  → CharYOLO (vertikal) / EasyOCR (lainnya)
2. Gemini error?     → fallback sesuai tipe label
3. Gemini ISO-valid? → langsung return
4. Perlu crosscheck? → jalankan EasyOCR, bandingkan skor
```

Scoring mempertimbangkan validitas ISO 6346, panjang teks, dan pola regex.

#### `ocr_gemini.py`
Wrapper Gemini dengan **circuit breaker**: jika terkena rate limit (HTTP 429), engine dinonaktifkan selama 10 menit (`backoff_sec=600`) agar tidak spam API.

#### `ocr_easyocr.py`
Engine EasyOCR dengan preprocessing adaptif:
- **Deskew** — koreksi kemiringan teks (±20°)
- **Crop main text region** — membuang teks kecil di sekitar ROI utama
- **Multi-variant preprocessing** — original, gray+CLAHE, adaptive threshold, Otsu, dsb.
- **Charwise read** — khusus kontainer vertikal: threshold → deteksi karakter → OCR per karakter

#### `ocr_char_yolo.py`
Engine OCR berbasis model YOLO yang dilatih untuk mendeteksi karakter satu per satu. Digunakan khusus untuk `Container_ID_Vertikal` saat Gemini tidak tersedia.

#### `iso_6346.py`
Implementasi penuh standar ISO 6346 untuk Container ID:
- **`validate_iso()`** — validasi format dan check digit
- **`repair_iso()`** — auto-repair dengan strategi: fix category ID, confusable chars (`O↔0`, `I↔1`, `S↔5`), fix check digit, multi-edit prefix search

#### `session_aggregator.py`
Mengagregasi semua observasi OCR selama satu sesi timbang menjadi hasil terbaik.

Proteksi isolasi session:
- `start()` otomatis finalize session lama jika masih aktif
- `max_keepalive_sec` — batas absolut durasi session (default 120 detik)
- `min_gap_between_sessions_sec` — cooldown antar session mencegah bounce
- **Identity tracking** — deteksi pergantian truk dari owner code kontainer (misal `SPNU` → `TCKU`), langsung auto-finalize

#### `database_handler.py`
Handler MySQL dengan connection pool. Tabel utama:

| Tabel | Isi |
|-------|-----|
| `weigh_sessions` | Satu record per sesi timbang |
| `weigh_session_detections` | Semua deteksi OCR dalam sesi (termasuk data ISO) |
| `container_trips` | Trip kontainer yang valid ISO per sesi |
| `containers` | Master data kontainer |
| `trucks` / `vehicles` | Master data truk & plat |
| `weight_logs` | Log berat per sesi |
| `detection_logs` | Log semua deteksi YOLO |

---

### `src/hardware/`

#### `scale_interface.py`
Polling HTTP ke jembatan timbang setiap 0.3 detik. Menghasilkan dua event:
- `START` — berat ≥ 200 kg dan stabil (8 sample, toleransi ±30 kg)
- `END` — berat ≤ 80 kg dan stabil, minimal 2 detik setelah START

Event disimpan dalam **queue** (bukan satu slot) sehingga tidak ada event yang tertimpa. Saat HTTP error, buffer diisi `0.0` agar stabilitas dianggap hilang.

---

### `src/gui/`

#### `main_window.py`
Jendela utama PyQt5. YOLO+OCR berjalan di `FrameWorker` (QThread per kamera) agar UI tidak freeze. Main thread hanya menangani: update widget, poll skala, dan handle session event via `QTimer` 100ms.

#### `camera_widget.py`
Widget per kamera: video feed, badge status koneksi, FPS, dan panel hasil OCR. Memiliki dua state: **LIVE** (update setiap frame) dan **FINAL** (tampilkan hasil session, tidak berubah sampai session baru).

#### `styles.py`
Stylesheet QSS global dengan desain minimalis monokrom (hitam/putih/abu).

---

## Persyaratan Sistem

| Komponen | Minimum |
|----------|---------|
| OS | Ubuntu 20.04 / Windows 10 |
| Python | 3.9+ |
| GPU | NVIDIA GPU dengan CUDA (opsional, tapi sangat disarankan) |
| RAM | 8 GB |
| MySQL | 8.0+ |
| Kamera | 6x RTSP IP Camera |
| Jaringan | LAN ke kamera & jembatan timbang |

---

## Instalasi & Setup

### 1. Clone & buat virtual environment

```bash
git clone <repo-url>
cd AutoGate
python -m venv venv
source venv/bin/activate        # Linux/Mac
# atau
venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Isi `requirements.txt`:
```
ultralytics>=8.0.0
opencv-python>=4.5.0
PyQt5>=5.15.0
google-generativeai>=0.3.0
pyserial>=3.5
numpy>=1.21.0
Pillow>=8.3.0
mysql-connector-python>=8.0.0
sqlalchemy>=2.0.0
cryptography>=3.4.0
easyocr
```

> **Catatan GPU:** Untuk mengaktifkan GPU pada EasyOCR dan YOLO, pastikan CUDA Toolkit dan `torch` versi GPU sudah terinstall sebelum menjalankan pip install di atas.

### 3. Setup Database MySQL

```sql
CREATE DATABASE autogate_system;
CREATE USER 'admin'@'localhost' IDENTIFIED BY 'passwordku123';
GRANT ALL PRIVILEGES ON autogate_system.* TO 'admin'@'localhost';
FLUSH PRIVILEGES;
```

> Tabel akan dibuat otomatis saat aplikasi pertama kali dijalankan.

### 4. Letakkan model YOLO

Salin file model ke:
```
AutoGate/models/best.pt
```

Untuk Char-YOLO (opsional, untuk kontainer vertikal):
```
/home/remote-user/AutoGate/AutoGate - Char Model/YoloV8_ImG_Proses/best.pt
```

---

## Konfigurasi

### Kamera RTSP — `config/config.py`

```python
CAMERAS = [
    {"id": 0, "name": "Camera 1 - Entrance", "source": 0, "active": True},
    ...
]
```

URL RTSP aktual dikonfigurasi di `src/core/camera_manager.py`:

```python
self.camera_sources = {
    0: "rtsp://admin:PASSWORD@10.35.61.115/video",
    1: "rtsp://admin:PASSWORD@10.35.61.116/video",
    2: "rtsp://admin:PASSWORD@10.35.61.117/video",
    3: "rtsp://admin:PASSWORD@10.35.61.118/video",
    4: "rtsp://admin:PASSWORD@10.35.61.121/video",
    5: "rtsp://admin:PASSWORD@10.35.61.120/video",
}
```

### Database — `config/database_config.py`

```python
MYSQL_HOST     = "localhost"
MYSQL_PORT     = 3306
MYSQL_USER     = "admin"
MYSQL_PASSWORD = "passwordku123"   # ← ganti sesuai setup
MYSQL_DATABASE = "autogate_system"
```

### Jembatan Timbang — `src/gui/main_window.py`

```python
self.scale = ScaleInterface(
    url="http://<IP-TIMBANGAN>/path/ke/berat.txt",
    poll_interval=0.3,          # detik antar polling
    timeout=1.5,                # timeout HTTP request
    threshold_start_kg=200.0,   # berat minimum START session
    threshold_end_kg=80.0,      # berat maksimum END session
    stable_samples=8,           # jumlah sample untuk deteksi stabil
    stable_tolerance_kg=30.0,   # toleransi fluktuasi berat (kg)
)
```

> **Tips:** Pastikan `timeout < poll_interval` untuk mencegah penurunan polling rate saat HTTP timeout.

### Gemini API Key — `main.py`

```python
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
```

Dapatkan API key di: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

Model yang digunakan (otomatis fallback):
1. `gemini-2.5-flash` (prioritas utama)
2. `gemini-1.5-flash`
3. `gemini-1.5-pro`

---

## Cara Menjalankan

```bash
cd AutoGate
source venv/bin/activate
python main.py
```

Setelah aplikasi terbuka:
1. Klik **"START CAMERAS"** untuk memulai streaming semua kamera
2. OCR hanya aktif saat **session timbang sedang berlangsung** (timbangan mendeteksi berat ≥ 200 kg)
3. Hasil deteksi ditampilkan di setiap widget kamera
4. Saat sesi selesai (berat turun ≤ 80 kg), hasil terbaik ditampilkan dengan badge **FINAL**

---

## Alur Kerja Sistem

```
Truk masuk ke area timbang
        │
        ▼
ScaleInterface mendeteksi berat ≥ 200 kg (stabil)
        │
        ▼
Event START → SessionAggregator.start()
        │
        ▼
FrameWorker aktif memproses setiap kamera:
  YOLO deteksi bbox
        │
        ├── Container_ID (horizontal)
        │       └── OCRRouter → Gemini / EasyOCR
        ├── Container_ID_Vertikal
        │       └── OCRRouter → Gemini / CharYOLO
        ├── truck_id
        │       └── OCRRouter → Gemini / EasyOCR
        └── plate_number
                └── OCRRouter → Gemini / EasyOCR
        │
        ▼
SessionAggregator.add_observation()
  - Validasi ISO 6346
  - Filter noise
  - Tracking owner code
        │
        ▼
ScaleInterface mendeteksi berat ≤ 80 kg (stabil)
        │
        ▼
Event END → SessionAggregator.finalize()
  - Pilih hasil terbaik per tipe
  - Simpan ke weigh_sessions
  - Simpan ke container_trips
  - Log ke weight_logs
        │
        ▼
GUI menampilkan badge FINAL di semua camera widget
```

---

## Catatan Penting

- **Kuota Gemini:** Sistem memiliki circuit breaker otomatis. Jika kuota habis, Gemini dinonaktifkan sementara dan EasyOCR/CharYOLO mengambil alih.
- **Jangan edit `styles.py`** menggunakan script otomatis — file ini berisi hex warna CSS yang diawali `#` dan akan salah diproses sebagai komentar Python.
- **Debug ROI:** Setiap ROI yang diproses OCR disimpan di folder `debug_ocr_router/` untuk keperluan debugging dan peningkatan model.
- **ISO 6346:** Hanya Container ID yang lolos validasi ISO yang akan masuk ke tabel `containers` dan `container_trips` sebagai foreign key. Data raw OCR tetap tersimpan di `weigh_session_detections`.