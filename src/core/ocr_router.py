import re
from typing import Tuple, Optional

from src.core.ocr_gemini import GeminiOCREngine
from src.core.timing_profiler import TimingProfiler
from src.core.ocr_easyocr import EasyOCREngine
from src.core.ocr_char_yolo import CharYOLOEngine

try:
    from src.core.iso_6346 import validate_iso, repair_iso
    _HAS_ISO = True
except Exception:
    validate_iso = None
    repair_iso = None
    _HAS_ISO = False

INVALID = {"ERROR", "ROI_INVALID", "", None}
TIDAK_TERBACA = "TIDAK_TERBACA"

_CONTAINER_RE = re.compile(r"^[A-Z]{4}\d{7}$")

_CHAR_YOLO_LABELS = {"Container_ID_Vertikal", "container_id_vertikal"}

_MIN_LEN = {
    "container": 6,
    "truck":     3,
    "plate":     4,
}

def _is_valid_text(txt: Optional[str]) -> bool:
    return bool(txt) and txt not in INVALID and txt != TIDAK_TERBACA


def _label_category(label: str) -> str:
    lab = (label or "").lower()
    if "container" in lab:
        return "container"
    if "truck" in lab:
        return "truck"
    if "plate" in lab:
        return "plate"
    return "generic"


def _score_text(txt: str, label: str) -> float:
    if not _is_valid_text(txt):
        return 0.0

    cat = _label_category(label)

    if cat == "container":
        if _CONTAINER_RE.match(txt):
            if _HAS_ISO and validate_iso:
                try:
                    iso = validate_iso(txt)
                    if iso.is_valid:
                        return 1.0
                    if repair_iso:
                        rep = repair_iso(txt, max_edits=2)
                        if rep and rep.repaired_text:
                            return max(0.4, float(rep.score or 0.4))
                    return 0.35
                except Exception:
                    return 0.3
            return 0.5

        if re.match(r"^[A-Z]{3,4}\d{4,}$", txt):
            length_score = min(len(txt) / 11.0, 1.0) * 0.3
            return length_score

        alnum = re.sub(r"[^A-Z0-9]", "", txt)
        if len(alnum) >= _MIN_LEN["container"]:
            return 0.05
        return 0.0

    if cat == "truck":
        alnum = re.sub(r"[^A-Z0-9]", "", txt)
        if len(alnum) < _MIN_LEN["truck"]:
            return 0.0
        digit_ratio = sum(c.isdigit() for c in alnum) / max(len(alnum), 1)
        return 0.3 + 0.4 * digit_ratio

    if cat == "plate":
        alnum = re.sub(r"[^A-Z0-9]", "", txt)
        if len(alnum) < _MIN_LEN["plate"]:
            return 0.0
        return min(len(alnum) / 8.0, 1.0) * 0.6

    return min(len(txt) / 6.0, 1.0) * 0.3


def _is_vertical_container(label: str) -> bool:
    """True jika label ini adalah Container_ID_Vertikal."""
    lab = (label or "").strip()
    return lab in _CHAR_YOLO_LABELS or (
        "vertikal" in lab.lower() or "vertical" in lab.lower()
    ) and "container" in lab.lower()


def _passes_minimum(txt: str, label: str) -> bool:
    if not _is_valid_text(txt):
        return False
    cat = _label_category(label)
    alnum = re.sub(r"[^A-Z0-9]", "", txt)
    min_len = _MIN_LEN.get(cat, 3)
    return len(alnum) >= min_len

class OCRRouter:
    def __init__(
        self,
        easyocr_gpu: bool = True,
        always_crosscheck: bool = False,
        crosscheck_labels: Tuple[str, ...] = ("container",),
    ):
        self.gemini     = GeminiOCREngine(backoff_sec=600.0)
        self.easy       = EasyOCREngine(languages=["en"], gpu=easyocr_gpu)
        self.char_yolo  = CharYOLOEngine(
            model_path=(
                "/home/remote-user/AutoGate/AutoGate - Char Model"
                "/YoloV8_ImG_Proses/best.pt"
            ),
            conf=0.15,
            iou=0.30,
            imgsz=960,
            device="0",
            reading_order="ttb",
        )

        self.always_crosscheck   = always_crosscheck
        self.crosscheck_labels   = crosscheck_labels
        self._profiler = TimingProfiler.get_instance()

    def read(self, roi_bgr, label: str) -> Tuple[str, str]:
        if self.gemini.is_disabled():
            if _is_vertical_container(label):
                return self.char_yolo.read(roi_bgr, label)
            txt, eng = self.easy.read(roi_bgr, label)
            return txt, eng

        self._profiler.start("gemini")
        g_txt, g_eng = self.gemini.read(roi_bgr, label)
        self._profiler.stop("gemini")
        g_error = g_eng in ("gemini_error", "gemini_disabled")

        if g_error:
            if _is_vertical_container(label):
                print(f"[OCR] Gemini error → CharYOLO fallback | label={label}")
                self._profiler.start("char_yolo_fallback")
                txt, eng = self.char_yolo.read(roi_bgr, label)
                self._profiler.stop("char_yolo_fallback")
                return txt, f"char_yolo_fallback({g_eng})"
            self._profiler.start("easyocr_fallback")
            txt, eng = self.easy.read(roi_bgr, label)
            self._profiler.stop("easyocr_fallback")
            return txt, f"easyocr_fallback({g_eng})"

        g_score = _score_text(g_txt, label)
        cat     = _label_category(label)

        if cat == "container" and g_score >= 1.0:
            print(f"[OCR] Gemini ISO-valid: {g_txt} | label={label}")
            return g_txt, "gemini"

        need_crosscheck = (
            self.always_crosscheck
            or cat in self.crosscheck_labels
            or not _passes_minimum(g_txt, label)
        )

        if not need_crosscheck:
            return g_txt, g_eng

        self._profiler.start("easyocr_crosscheck")
        e_txt, e_eng = self.easy.read(roi_bgr, label)
        self._profiler.stop("easyocr_crosscheck")
        e_score = _score_text(e_txt, label)

        winner_txt, winner_eng, winner_score = self._pick_winner(
            g_txt, g_eng, g_score,
            e_txt, e_eng, e_score,
            label,
        )

        print(
            f"[OCR] label={label} | gemini={g_txt}({g_score:.2f}) | "
            f"easy={e_txt}({e_score:.2f}) | winner={winner_txt}({winner_eng})"
        )
        return winner_txt, winner_eng

    def _pick_winner(
        self,
        g_txt: str, g_eng: str, g_score: float,
        e_txt: str, e_eng: str, e_score: float,
        label: str,
    ) -> Tuple[str, str, float]:
        g_valid = _is_valid_text(g_txt)
        e_valid = _is_valid_text(e_txt)

        if not g_valid and not e_valid:
            return TIDAK_TERBACA, "both_failed", 0.0

        if not g_valid:
            return e_txt, e_eng, e_score

        if not e_valid:
            return g_txt, g_eng, g_score

        if e_score > g_score + 0.05:
            return e_txt, f"easyocr_winner(g={g_score:.2f},e={e_score:.2f})", e_score

        return g_txt, f"gemini_winner(g={g_score:.2f},e={e_score:.2f})", g_score