import re
from typing import Tuple, List, Optional

import cv2
import numpy as np

try:
    import easyocr
except Exception:
    easyocr = None

CONTAINER_REGEX = re.compile(r"^[A-Z]{4}\d{7}$")


class EasyOCREngine:
    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = False):
        if languages is None:
            languages = ["en"]

        if easyocr is None:
            raise RuntimeError("easyocr tidak terinstall / gagal import.")

        self.reader = easyocr.Reader(languages, gpu=gpu)
        self.allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def _normalize(self, text: str) -> str:
        if not text:
            return "TIDAK_TERBACA"
        t = text.strip().upper()
        t = t.replace(" ", "")
        t = re.sub(r"[^A-Z0-9]", "", t)
        return t or "TIDAK_TERBACA"

    def _score_text(self, text_norm: str, conf: float, label: str) -> float:
        if not text_norm or text_norm in ("TIDAK_TERBACA", "ROI_INVALID"):
            return -999.0

        score = float(conf)
        is_container = "container" in label.lower()
        is_truck     = "truck" in label.lower()

        if is_container:
            if len(text_norm) < 8:
                return -500.0
            if len(text_norm) < 11:
                score -= 0.6
            if CONTAINER_REGEX.match(text_norm):
                score += 3.0
            if len(text_norm) == 11:
                score += 0.4

        if is_truck:
            if len(text_norm) < 3:
                return -500.0

            alnum = re.sub(r"[^A-Z0-9]", "", text_norm)

            if re.match(r"^[A-Z]{1,3}\d{1,4}[A-Z]{0,3}$", alnum):
                score += 2.0

            if 6 <= len(alnum) <= 10:
                score += 1.0
            elif 4 <= len(alnum) <= 12:
                score += 0.3

            score += min(len(alnum) / 9.0, 1.0) * 0.5

            if len(alnum) < 5:
                score -= 1.0

        return score

    def _to_gray_clahe(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return gray

    def _adaptive_thresh(self, gray):
        thr = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 5
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
        return thr

    def _otsu_thresh(self, gray):
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
        return thr

    def _deskew(self, img_bgr: np.ndarray) -> np.ndarray:
        """Koreksi kemiringan teks (deskew)."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        coords = np.column_stack(np.where(thr > 0))
        if len(coords) < 50:
            return img_bgr

        rect  = cv2.minAreaRect(coords)
        angle = rect[-1]

        if angle < -45:
            angle = 90 + angle

        if abs(angle) > 20 or abs(angle) < 0.5:
            return img_bgr

        h, w   = img_bgr.shape[:2]
        center = (w // 2, h // 2)
        M      = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            img_bgr, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _crop_main_text_region(self, img_bgr: np.ndarray) -> np.ndarray:
        """Crop baris teks terbesar dari ROI — membuang teks kecil di sekitarnya."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
        dilated = cv2.dilate(thr, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img_bgr

        H, W     = img_bgr.shape[:2]
        min_area = (W * H) * 0.05

        regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < min_area or h < 15 or w < 30:
                continue
            regions.append((area, x, y, w, h))

        if not regions:
            return img_bgr

        regions.sort(reverse=True)
        _, x, y, w, h = regions[0]

        pad = 8
        x1  = max(0, x - pad)
        y1  = max(0, y - pad)
        x2  = min(W, x + w + pad)
        y2  = min(H, y + h + pad)

        crop = img_bgr[y1:y2, x1:x2]
        return crop if crop.size > 0 else img_bgr

    def _preprocess_truck_variants(self, img_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Preprocessing khusus truck_id dengan deskew + crop."""
        h, w = img_bgr.shape[:2]

        target_h = max(200, h)
        if h < target_h:
            scale   = target_h / h
            img_bgr = cv2.resize(
                img_bgr,
                (max(1, int(w * scale)), target_h),
                interpolation=cv2.INTER_CUBIC,
            )

        deskewed      = self._deskew(img_bgr)
        cropped       = self._crop_main_text_region(img_bgr)
        cropped_deskew = self._deskew(cropped)

        gray         = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        kernel_sharp = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
        gray_sharp   = np.clip(
            cv2.filter2D(gray, -1, kernel_sharp), 0, 255
        ).astype(np.uint8)
        clahe_high   = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))
        gray_sharp   = clahe_high.apply(gray_sharp)

        gray_ds      = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
        gray_sharp_ds = np.clip(
            cv2.filter2D(gray_ds, -1, kernel_sharp), 0, 255
        ).astype(np.uint8)
        clahe_high2  = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))
        gray_sharp_ds = clahe_high2.apply(gray_sharp_ds)

        hsv    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        v_ch   = hsv[:, :, 2]
        clahe_mid = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        v_cl   = clahe_mid.apply(v_ch)
        v_blur = cv2.GaussianBlur(v_cl, (3, 3), 0)
        _, otsu_v = cv2.threshold(v_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(otsu_v) < 127:
            otsu_v = cv2.bitwise_not(otsu_v)
        k      = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        otsu_v = cv2.morphologyEx(otsu_v, cv2.MORPH_OPEN, k)

        return [
            ("crop_deskew",    cropped_deskew),
            ("deskew",         deskewed),
            ("orig",           img_bgr),
            ("gray_sharp",     gray_sharp),
            ("gray_sharp_ds",  gray_sharp_ds),
            ("otsu_v",         otsu_v),
        ]

    def _preprocess_variants(self, img_bgr):
        gray = self._to_gray_clahe(img_bgr)
        variants = [
            ("orig", img_bgr),
            ("gray", gray),
            ("adap", self._adaptive_thresh(gray)),
            ("otsu", self._otsu_thresh(gray)),
        ]
        return variants

    def _rotate_truck_candidates(self, img_bgr):
        """Truck ID — hanya gunakan orientasi original."""
        return [
            ("orig", img_bgr),
        ]

    def _rotate_candidates(self, img_bgr):
        return [
            ("orig",      img_bgr),
            ("rot90_cw",  cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)),
            ("rot90_ccw", cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)),
            ("rot180",    cv2.rotate(img_bgr, cv2.ROTATE_180)),
        ]

    def _easyocr_read(self, img):
        return self.reader.readtext(
            img,
            detail=1,
            paragraph=False,
            allowlist=self.allowlist,
        )

    def _merge_results(self, results, is_vertical: bool) -> Tuple[str, float]:
        parts = []
        confs = []
        for item in results:
            if len(item) < 3:
                continue
            bbox, txt, conf = item[0], item[1], float(item[2])
            norm = self._normalize(txt)
            if norm in ("TIDAK_TERBACA", "ROI_INVALID"):
                continue

            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            cx = sum(xs) / 4.0
            cy = sum(ys) / 4.0

            parts.append((cx, cy, norm, conf))
            confs.append(conf)

        if not parts:
            return "TIDAK_TERBACA", 0.0

        if is_vertical:
            parts.sort(key=lambda x: (x[1], x[0]))
        else:
            parts.sort(key=lambda x: (x[0], x[1]))

        merged   = "".join([p[2] for p in parts])
        avg_conf = sum(confs) / max(1, len(confs))
        return merged if merged else "TIDAK_TERBACA", avg_conf

    def _extract_char_boxes(self, bin_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if np.mean(bin_img) > 127:
            bin_img = 255 - bin_img

        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        H, W  = cleaned.shape[:2]
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area  = w * h
            if area < 80 or h < 10 or w < 4:
                continue
            ratio = h / max(w, 1)
            if ratio < 0.7 or ratio > 10.0:
                continue
            if h > 0.95 * H or w > 0.95 * W:
                continue
            boxes.append((x, y, w, h))

        return boxes

    def _read_single_char(self, char_img_bgr) -> Tuple[str, float]:
        try:
            res = self.reader.readtext(
                char_img_bgr,
                detail=1,
                paragraph=False,
                allowlist=self.allowlist,
            )
        except Exception:
            return "", 0.0

        best_c    = ""
        best_conf = 0.0
        for item in res:
            if len(item) < 3:
                continue
            txt  = self._normalize(item[1])
            conf = float(item[2])
            if len(txt) == 1 and conf > best_conf:
                best_c    = txt
                best_conf = conf

        return best_c, best_conf

    def _charwise_read_vertical(self, roi_bgr) -> Tuple[str, str]:
        """Khusus container vertikal: threshold → char boxes → OCR per karakter."""
        best_text  = "TIDAK_TERBACA"
        best_score = -999.0
        best_tag   = "easyocr_charwise"

        h, w = roi_bgr.shape[:2]
        if h < 180:
            scale   = 180 / max(h, 1)
            roi_bgr = cv2.resize(
                roi_bgr,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_CUBIC,
            )

        for rot_tag, rot_img in self._rotate_candidates(roi_bgr):
            gray  = self._to_gray_clahe(rot_img)
            thr   = self._otsu_thresh(gray)
            boxes = self._extract_char_boxes(thr)
            if not boxes:
                continue

            boxes.sort(key=lambda b: (b[1], b[0]))

            chars = []
            confs = []
            for (x, y, bw, bh) in boxes:
                pad = 2
                x1  = max(0, x - pad)
                y1  = max(0, y - pad)
                x2  = min(rot_img.shape[1], x + bw + pad)
                y2  = min(rot_img.shape[0], y + bh + pad)
                crop = rot_img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                ch, cw = crop.shape[:2]
                if ch < 40:
                    s    = 40 / max(ch, 1)
                    crop = cv2.resize(
                        crop, (max(1, int(cw * s)), 40),
                        interpolation=cv2.INTER_CUBIC,
                    )

                c, cconf = self._read_single_char(crop)
                if c:
                    chars.append(c)
                    confs.append(cconf)

                if len(chars) >= 14:
                    break

            cand     = self._normalize("".join(chars))
            avg_conf = (sum(confs) / len(confs)) if confs else 0.0

            score = avg_conf
            if CONTAINER_REGEX.match(cand):
                score += 5.0
            score += max(0.0, 1.5 - abs(len(cand) - 11) * 0.25)

            if score > best_score:
                best_score = score
                best_text  = cand
                best_tag   = f"easyocr_charwise_{rot_tag}_len{len(cand)}"
                if CONTAINER_REGEX.match(best_text):
                    return best_text, best_tag

        return best_text, best_tag

    def read(self, roi_bgr, label: str) -> Tuple[str, str]:
        """Return: (best_text, engine_tag)"""
        if roi_bgr is None or roi_bgr.size == 0:
            return "ROI_INVALID", "easyocr"

        h, w = roi_bgr.shape[:2]
        if h < 140:
            scale   = 140 / max(h, 1)
            roi_bgr = cv2.resize(
                roi_bgr,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_CUBIC,
            )

        is_vertical  = ("vertikal" in label.lower()) or ("vertical" in label.lower())
        is_container = ("container" in label.lower())
        is_truck     = ("truck" in label.lower())

        best_text  = "TIDAK_TERBACA"
        best_score = -999.0
        best_tag   = "easyocr"

        if is_vertical:
            rot_cands = self._rotate_candidates(roi_bgr)
        elif is_truck:
            rot_cands = self._rotate_truck_candidates(roi_bgr)
        else:
            rot_cands = [("orig", roi_bgr)]

        for rot_tag, rot_img in rot_cands:
            pre_variants = (
                self._preprocess_truck_variants(rot_img)
                if is_truck
                else self._preprocess_variants(rot_img)
            )

            for pre_tag, img in pre_variants:
                try:
                    results = self._easyocr_read(img)
                except Exception as e:
                    print(f"[EASYOCR] read fail ({rot_tag}/{pre_tag}): {e}")
                    continue

                merged_text, merged_conf = self._merge_results(
                    results, is_vertical=is_vertical
                )
                merged_score = self._score_text(merged_text, merged_conf, label)

                if merged_score > best_score:
                    best_score = merged_score
                    best_text  = merged_text
                    best_tag   = f"easyocr_{rot_tag}_{pre_tag}_merged"
                    if is_container and CONTAINER_REGEX.match(best_text):
                        return best_text, best_tag

                for item in results:
                    if len(item) < 3:
                        continue
                    _, txt, conf = item[0], item[1], float(item[2])
                    norm  = self._normalize(txt)
                    score = self._score_text(norm, conf, label)
                    if score > best_score:
                        best_score = score
                        best_text  = norm
                        best_tag   = f"easyocr_{rot_tag}_{pre_tag}"
                        if is_container and CONTAINER_REGEX.match(best_text):
                            return best_text, best_tag

        if is_container and is_vertical and (
            not CONTAINER_REGEX.match(best_text) and len(best_text) < 10
        ):
            cw_text, cw_tag = self._charwise_read_vertical(roi_bgr)
            if CONTAINER_REGEX.match(cw_text):
                return cw_text, cw_tag
            if len(cw_text) > len(best_text):
                return cw_text, cw_tag

        return best_text, best_tag