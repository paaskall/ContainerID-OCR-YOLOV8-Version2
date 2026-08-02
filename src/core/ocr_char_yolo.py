import os
from typing import Tuple, List, Dict, Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO as _YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False


_DEFAULT_MODEL_PATH = (
    "/home/remote-user/AutoGate/AutoGate - Char Model"
    "/YoloV8_ImG_Proses/best.pt"
)


def _iou_xyxy(a: list, b: list) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def _nms_charwise(dets: List[Dict], iou_th: float = 0.25) -> List[Dict]:
    """NMS sederhana berdasarkan confidence — filter bbox overlapping."""
    dets = sorted(dets, key=lambda d: d["conf"], reverse=True)
    keep = []
    for d in dets:
        ok = all(_iou_xyxy(d["xyxy"], k["xyxy"]) <= iou_th for k in keep)
        if ok:
            keep.append(d)
    return keep


def _dedup_by_slots_ttb(
    dets: List[Dict],
    slot_gap_ratio: float = 0.55,
) -> List[Dict]:
    """Deduplikasi slot vertikal (top-to-bottom)."""
    if not dets:
        return []

    dets = sorted(dets, key=lambda d: (d["xyxy"][1] + d["xyxy"][3]) / 2.0)

    heights = [max(1.0, d["xyxy"][3] - d["xyxy"][1]) for d in dets]
    med_h = float(np.median(heights))
    gap = max(6.0, med_h * slot_gap_ratio)

    slots: List[List[Dict]] = []
    cur: List[Dict] = [dets[0]]
    cur_cy = (dets[0]["xyxy"][1] + dets[0]["xyxy"][3]) / 2.0

    for d in dets[1:]:
        cy = (d["xyxy"][1] + d["xyxy"][3]) / 2.0
        if abs(cy - cur_cy) <= gap:
            cur.append(d)
            cur_cy = (cur_cy + cy) / 2.0
        else:
            slots.append(cur)
            cur = [d]
            cur_cy = cy
    slots.append(cur)

    return [max(slot, key=lambda x: x["conf"]) for slot in slots]


def _build_text(dets: List[Dict]) -> str:
    return "".join(d["name"] for d in dets)


class CharYOLOEngine:

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        conf: float = 0.15,
        iou: float = 0.30,
        imgsz: int = 960,
        device: str = "0",
        reading_order: str = "ttb",
    ):
        self.conf          = conf
        self.iou           = iou
        self.imgsz         = imgsz
        self.device        = device
        self.reading_order = reading_order
        self.model         = None
        self.names: Dict[int, str] = {}

        self._load_model(model_path)

    def _load_model(self, model_path: str):
        if not _HAS_YOLO:
            print("[CharYOLO] ultralytics tidak terinstall.")
            return
        if not os.path.exists(model_path):
            print(f"[CharYOLO] Model tidak ditemukan: {model_path}")
            return
        try:
            self.model = _YOLO(model_path, task="detect")
            try:
                self.model.model.fuse = lambda verbose=True: self.model.model
            except Exception:
                pass
            self.names = self.model.names or {}
            print(f"[CharYOLO] Model loaded: {model_path}")
            print(f"[CharYOLO] Classes ({len(self.names)}): {dict(self.names)}")
        except Exception as e:
            print(f"[CharYOLO] Gagal load model: {e}")
            self.model = None

    def is_available(self) -> bool:
        return self.model is not None

    def _predict(self, img_bgr: np.ndarray) -> List[Dict]:
        """Jalankan YOLO predict dan kembalikan list deteksi mentah."""
        res = self.model.predict(
            source=img_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            agnostic_nms=True,
            max_det=1000,
        )[0]

        dets = []
        if res.boxes is None or len(res.boxes) == 0:
            return dets

        for b in res.boxes:
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            conf = float(b.conf[0].cpu().numpy())
            cls  = int(b.cls[0].cpu().numpy())
            dets.append({
                "xyxy": xyxy,
                "conf": conf,
                "cls":  cls,
                "name": self.names.get(cls, str(cls)),
            })
        return dets

    def _postprocess(self, dets: List[Dict]) -> List[Dict]:
        """NMS → dedup slot → sort urutan baca."""
        dets = _nms_charwise(dets, iou_th=max(0.15, self.iou - 0.05))

        if self.reading_order == "ttb":
            dets = _dedup_by_slots_ttb(dets, slot_gap_ratio=0.55)
            dets = sorted(dets, key=lambda d: (d["xyxy"][1] + d["xyxy"][3]) / 2.0)
        else:
            dets = sorted(dets, key=lambda d: (d["xyxy"][0] + d["xyxy"][2]) / 2.0)

        return dets

    def read(self, roi_bgr: np.ndarray, label: str = "") -> Tuple[str, str]:
        """Baca teks dari ROI menggunakan char YOLO."""
        if not self.is_available():
            return "TIDAK_TERBACA", "char_yolo_unavailable"

        if roi_bgr is None or roi_bgr.size == 0:
            return "ROI_INVALID", "char_yolo"

        h, w = roi_bgr.shape[:2]
        if h < 200:
            scale = 200 / h
            roi_bgr = cv2.resize(
                roi_bgr,
                (max(1, int(w * scale)), 200),
                interpolation=cv2.INTER_CUBIC,
            )

        try:
            dets = self._predict(roi_bgr)
        except Exception as e:
            print(f"[CharYOLO] predict error: {e}")
            return "TIDAK_TERBACA", "char_yolo_error"

        if not dets:
            return "TIDAK_TERBACA", "char_yolo_no_det"

        dets    = self._postprocess(dets)
        text    = _build_text(dets)
        n_chars = len(dets)

        if not text:
            return "TIDAK_TERBACA", "char_yolo_empty"

        tag = f"char_yolo_ttb_{n_chars}chars"
        print(f"[CharYOLO] label={label} | n={n_chars} | text='{text}'")
        return text.upper(), tag