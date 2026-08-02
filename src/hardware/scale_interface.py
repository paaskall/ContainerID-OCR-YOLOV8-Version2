import time
import threading
import requests
from dataclasses import dataclass
from typing import Optional, Deque, Tuple
from collections import deque

@dataclass
class ScaleEvent:
    kind: str
    weight_raw: str
    weight_kg: Optional[float]
    ts: float


class ScaleInterface:

    def __init__(
        self,
        url: str,
        poll_interval: float = 0.3,
        timeout: float = 1.5,
        threshold_start_kg: float = 200.0,
        threshold_end_kg: float = 80.0,
        stable_samples: int = 8,
        stable_tolerance_kg: float = 30.0,
        minimum_session_hold_sec: float = 2.0,
        offline_threshold_sec: float = 10.0,
    ):
        self.url                = url
        self.poll_interval      = poll_interval
        self.timeout            = timeout
        self.threshold_start_kg = threshold_start_kg
        self.threshold_end_kg   = threshold_end_kg
        self.stable_samples     = max(3, int(stable_samples))
        self.stable_tolerance_kg = float(stable_tolerance_kg)
        self.minimum_session_hold_sec = float(minimum_session_hold_sec)
        self.offline_threshold_sec    = float(offline_threshold_sec)

        if self.timeout >= self.poll_interval:
            print(
                f"[SCALE] WARNING: timeout ({self.timeout}s) >= poll_interval ({self.poll_interval}s). "
                f"Saat HTTP timeout, polling rate akan turun drastis. "
                f"Disarankan timeout < poll_interval, misal timeout=0.25 untuk poll_interval=0.3."
            )

        self._lock = threading.Lock()

        self._latest_raw: Optional[str]   = None
        self._latest_kg:  Optional[float] = None
        self._latest_ts:  float           = 0.0

        self._event_queue: deque = deque()

        self._buf: Deque[float] = deque(maxlen=self.stable_samples)

        self._armed:      bool  = True
        self._in_session: bool  = False
        self._session_started_at: float = 0.0

        self.last_error:  Optional[str] = None
        self.last_ok_ts:  float         = 0.0
        self._error_count: int          = 0

        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        """Hentikan polling thread."""
        self._stop.set()

    def rearm(self):
        """
        Reset state scale ke kondisi awal (armed=True, in_session=False).
        Dipanggil dari main_window jika session.start() gagal (DB error /
        min_gap belum terpenuhi) — agar scale tidak terjebak di state
        _in_session=True tanpa session aktif di aggregator.
        """
        with self._lock:
            self._armed      = True
            self._in_session = False
            self._session_started_at = 0.0
            self._buf.clear()   # clear buffer agar harus stabil ulang
            print("[SCALE] rearm() dipanggil — scale siap START lagi")

    def get_latest(self) -> Tuple[Optional[str], Optional[float], float, Optional[str], float]:
        """Return (raw, kg, ts, last_error, last_ok_ts)."""
        with self._lock:
            return (
                self._latest_raw,
                self._latest_kg,
                self._latest_ts,
                self.last_error,
                self.last_ok_ts,
            )

    def pop_event(self) -> Optional[ScaleEvent]:
        """Ambil satu event dari queue (FIFO)."""
        with self._lock:
            if self._event_queue:
                return self._event_queue.popleft()
            return None

    @property
    def is_online(self) -> bool:
        """True jika berhasil polling dalam offline_threshold_sec terakhir."""
        with self._lock:
            return (time.time() - self.last_ok_ts) < self.offline_threshold_sec

    @property
    def in_session(self) -> bool:
        """True jika sedang dalam weighing session (START sudah tapi END belum)."""
        with self._lock:
            return self._in_session

    def _parse_weight(self, raw: str) -> Optional[float]:
        raw = (raw or "").strip()
        if not raw:
            return None
        try:
            v = float(raw)
            return 0.0 if abs(v) < 1e-6 else v
        except Exception:
            return None

    def _is_stable(self) -> bool:
        if len(self._buf) < self.stable_samples:
            return False
        return (max(self._buf) - min(self._buf)) <= self.stable_tolerance_kg

    def _worker(self):
        last_print = 0.0

        while not self._stop.is_set():
            sleep_for = self.poll_interval
            try:
                ts_ms = int(time.time() * 1000)
                r     = requests.get(
                    f"{self.url}?_={ts_ms}",
                    timeout=self.timeout,
                    headers={"Connection": "close"},
                )
                r.raise_for_status()
                raw   = (r.text or "").strip()
                kg    = self._parse_weight(raw)
                now   = time.time()

                with self._lock:
                    self._latest_raw = raw
                    self._latest_kg  = kg
                    self._latest_ts  = now
                    self.last_error  = None
                    self.last_ok_ts  = now
                    self._error_count = 0

                    kg_val = float(kg) if kg is not None else 0.0
                    self._buf.append(kg_val)
                    stable = self._is_stable()

                    if self._armed and (not self._in_session):
                        if stable and kg_val >= self.threshold_start_kg:
                            self._event_queue.append(ScaleEvent(
                                kind="START",
                                weight_raw=raw,
                                weight_kg=kg_val,
                                ts=now,
                            ))
                            self._in_session        = True
                            self._armed             = False
                            self._session_started_at = now
                            print(f"[SCALE] START event | kg={kg_val:.0f}")

                    elif self._in_session:
                        held = now - self._session_started_at
                        if (
                            stable
                            and kg_val <= self.threshold_end_kg
                            and held >= self.minimum_session_hold_sec
                        ):
                            self._event_queue.append(ScaleEvent(
                                kind="END",
                                weight_raw=raw,
                                weight_kg=kg_val,
                                ts=now,
                            ))
                            self._in_session = False
                            self._armed      = True
                            print(f"[SCALE] END event | kg={kg_val:.0f} | held={held:.1f}s")

            except Exception as e:
                now = time.time()

                with self._lock:
                    self.last_error = str(e)
                    self._error_count += 1
                    error_count = self._error_count
                    # Tidak append 0 ke buffer — biarkan data lama
                    # agar setelah koneksi pulih tidak harus tunggu 8 sample baru
                    # Tapi jika sedang armed (belum session), kita clear buffer
                    # supaya tidak false-trigger START dari data lama
                    if not self._in_session:
                        self._buf.clear()

                sleep_for = min(self.poll_interval * min(error_count, 5), 8.0)

                if error_count == 1 or now - last_print > 30.0:
                    print(f"[SCALE] ERROR x{error_count}: {e}")
                    last_print = now

            time.sleep(sleep_for)
