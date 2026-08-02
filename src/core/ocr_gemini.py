import time
from typing import Tuple

from config.gemini_config import gemini_config

INVALID = {"TIDAK_TERBACA", "ERROR", "ROI_INVALID", "", None}

class GeminiOCREngine:

    def __init__(self, backoff_sec: float = 600.0):
        self.backoff_sec = float(backoff_sec)
        self.disabled_until = 0.0

    def is_disabled(self) -> bool:
        return time.time() < self.disabled_until

    def _disable(self, reason: str = ""):
        self.disabled_until = time.time() + self.backoff_sec
        print(f"[GEMINI] Disabled {self.backoff_sec:.0f}s. reason={reason}")

    def read(self, roi_bgr, label: str) -> Tuple[str, str]:
        """Return: (text, engine_name)"""
        if self.is_disabled():
            return "TIDAK_TERBACA", "gemini_disabled"

        try:
            txt = gemini_config.extract_text_from_image(roi_bgr, label)
            txt_norm = ("" if txt is None else str(txt)).strip().upper()

            if txt_norm in {"ERROR", "ROI_INVALID"}:
                return "TIDAK_TERBACA", "gemini_error"

            if not txt_norm:
                return "TIDAK_TERBACA", "gemini"

            if txt_norm == "TIDAK_TERBACA":
                return "TIDAK_TERBACA", "gemini"

            return txt_norm, "gemini"

        except Exception as e:
            msg = str(e).lower()

            if "429" in msg or "quota" in msg or "rate" in msg or "exceeded" in msg:
                self._disable(reason="rate_limit")
                return "TIDAK_TERBACA", "gemini_disabled"

            print(f"[GEMINI] error: {e}")
            return "TIDAK_TERBACA", "gemini_error"