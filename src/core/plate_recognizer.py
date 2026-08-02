import cv2
import os
import sys
import time
from datetime import datetime
from typing import Dict, Optional

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, project_root)

try:
    from config.gemini_config import gemini_config
    print("Gemini config imported successfully")
except ImportError as e:
    print(f"Failed to import gemini_config: {e}")

    class FallbackGeminiConfig:
        def __init__(self):
            self.configured = False

        def setup_api_key(self, api_key):
            return False

        def extract_text_from_image(self, image, label):
            return "TIDAK_TERBACA"

    gemini_config = FallbackGeminiConfig()

from ultralytics import YOLO

try:
    from src.core.database_handler import db_handler
    print("Database handler imported successfully")
except ImportError as e:
    print(f"Failed to import database handler: {e}")
    db_handler = None

from src.core.ocr_router import OCRRouter

INVALID_OCR = {"TIDAK_TERBACA", "ERROR", "ROI_INVALID", "", None}


class PlateRecognizer:

    OCR_CONF_THRESH = {
        "Container_ID": 0.15,
        "Container_ID_Vertikal": 0.15,
        "truck_id": 0.25,
        "plate_number": 0.25,
    }

    ROI_MIN_PIXELS = {
        "Container_ID": 2500,
        "Container_ID_Vertikal": 2000,
        "truck_id": 3500,
        "plate_number": 2500,
    }

    DEDUP_GRID = {
        "Container_ID": 40,
        "Container_ID_Vertikal": 40,
        "truck_id": 50,
        "plate_number": 50,
    }

    MAX_RETRY_PER_BBOX = 2
    RETRY_COOLDOWN_SEC = 5.0

    def __init__(self, model_path: str = "models/best.pt"):
        self.model = None
        self.class_names: Dict[int, str] = {}
        self.successful_reads: Dict[str, list] = {}

        self._last_session_id: Optional[int] = None

        self._bbox_state: Dict[str, Dict] = {}

        self.SESSION_OCR_LIMIT: Dict[str, int] = {
            "Container_ID": 5,
            "Container_ID_Vertikal": 2,
            "truck_id": 3,
            "plate_number": 2,
        }
        self._session_ocr_count: Dict[str, int] = {}

        self.debug_save_roi = True
        self.debug_folder = "/home/remote-user/debug_ocr/g2/debug_ocr_router"
        os.makedirs(self.debug_folder, exist_ok=True)

        self.capture_save_frame = True
        self.capture_folder = "/home/remote-user/debug_ocr/g2/debug_capture"
        os.makedirs(self.capture_folder, exist_ok=True)

        self.load_models(model_path)
        self.ocr = OCRRouter(easyocr_gpu=True)

    def set_gemini_api_key(self, api_key: str) -> bool:
        try:
            print(f"Setting Gemini API key: {api_key[:10]}...")
        except Exception:
            print("Setting Gemini API key...")
        try:
            return bool(gemini_config.setup_api_key(api_key))
        except Exception as e:
            print(f"Gemini setup failed: {e}")
            return False

    def load_models(self, model_path: str):
        try:
            model_abs_path = os.path.join(project_root, model_path)
            if os.path.exists(model_abs_path):
                print(f"Loading model from: {model_abs_path}")
                self.model = YOLO(model_abs_path, task="detect")
                try:
                    self.model.model.fuse = lambda verbose=True: self.model.model
                except Exception:
                    pass

                if hasattr(self.model, "names") and self.model.names:
                    self.class_names = self.model.names
                    print(f"YOLOv8 model loaded with classes: {dict(self.class_names)}")
                else:
                    self.class_names = {
                        0: "Container_ID",
                        1: "Container_ID_Vertikal",
                        2: "truck_id",
                        3: "plate_number",
                    }
            else:
                print(f"Model not found at {model_abs_path}")
                self.class_names = {
                    0: "Container_ID",
                    1: "Container_ID_Vertikal",
                    2: "truck_id",
                    3: "plate_number",
                }

            for name in self.class_names.values():
                self.successful_reads.setdefault(name, [])

        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            self.model = None
            self.class_names = {
                0: "Container_ID",
                1: "Container_ID_Vertikal",
                2: "truck_id",
                3: "plate_number",
            }
            for name in self.class_names.values():
                self.successful_reads.setdefault(name, [])

    def expand_roi_based_on_class(self, x1, y1, x2, y2, frame_shape, label: str):
        """- Container_ID_Vertikal jangan expand X terlalu besar."""
        h, w = frame_shape[:2]
        width = x2 - x1
        height = y2 - y1

        lab = (label or "").lower()

        if "container" in lab and ("vertikal" in lab or "vertical" in lab):
            expand_x = int(width * 0.20)
            expand_y = int(height * 0.12)
        elif "container" in lab:
            expand_x = int(width * 0.45)
            expand_y = int(height * 0.18)
        elif "plate" in lab:
            expand_x = int(width * 0.35)
            expand_y = int(height * 0.35)
        else:
            expand_x = int(width * 0.30)
            expand_y = int(height * 0.30)

        x1_exp = max(0, x1 - expand_x)
        y1_exp = max(0, y1 - expand_y)
        x2_exp = min(w, x2 + expand_x)
        y2_exp = min(h, y2 + expand_y)
        return x1_exp, y1_exp, x2_exp, y2_exp

    def get_color_for_class(self, label: str):
        colors = {
            "Container_ID": (0, 255, 0),
            "Container_ID_Vertikal": (0, 200, 0),
            "truck_id": (255, 0, 0),
            "plate_number": (0, 0, 255),
        }
        return colors.get(label, (255, 255, 0))

    def preprocess_for_ocr(self, roi_bgr, label: str):
        """Preprocess ringan: resize minimum height."""
        try:
            if roi_bgr is None or roi_bgr.size == 0:
                return None

            height, width = roi_bgr.shape[:2]
            lab = (label or "").lower()

            if "container" in lab:
                target_height = 220
            elif "plate" in lab:
                target_height = 160
            else:
                target_height = 180

            if height < target_height:
                scale = target_height / max(1, height)
                new_width = max(1, int(width * scale))
                roi_bgr = cv2.resize(roi_bgr, (new_width, target_height), interpolation=cv2.INTER_CUBIC)

            return roi_bgr
        except Exception as e:
            print(f"Preprocessing error for {label}: {e}")
            return None

    def _safe_name(self, s: str, max_len: int = 40) -> str:
        s = (s or "").strip().replace(" ", "")
        s = s.replace("/", "_").replace("\\", "_").replace(":", "_")
        s = "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-", "."))
        return s[:max_len] if s else "NA"

    def _save_capture_frame(
        self,
        frame,
        camera_id: int,
        session_id,
        capture_name: Optional[str] = None,
        normalized_type: Optional[str] = None,
    ):
        """Simpan full-frame yang menjadi sumber ROI debug."""
        if not self.capture_save_frame:
            return None
        try:
            if frame is None or getattr(frame, "size", 0) == 0:
                return None

            date_folder = time.strftime("%Y-%m-%d")
            sid = session_id if session_id is not None else "no_session"
            out_dir = os.path.join(self.capture_folder, date_folder, f"session_{sid}")
            if normalized_type:
                out_dir = os.path.join(out_dir, self._safe_name(normalized_type, 30))
            os.makedirs(out_dir, exist_ok=True)

            if capture_name:
                filename = f"{self._safe_name(capture_name, 120)}_FRAME.jpg"
            else:
                filename = f"cam{camera_id}_capture.jpg"

            capture_path = os.path.join(out_dir, filename)
            cv2.imwrite(capture_path, frame)
            return capture_path
        except Exception as e:
            print(f"[CAPTURE FRAME] save failed: {e}")
            return None

    def _save_roi_bundle(
        self,
        roi_raw_bgr,
        roi_pre_bgr,
        source_frame_bgr,
        normalized_type: str,
        class_name: str,
        confidence: float,
        camera_id: int,
        session_id,
        bbox_xyxy,
        roi_bbox_xyxy,
        ocr_engine: str,
        ocr_result: str,
    ):
        if not self.debug_save_roi:
            return

        try:
            date_folder = time.strftime("%Y-%m-%d")
            sid = session_id if session_id is not None else "no_session"
            out_dir = os.path.join(self.debug_folder, date_folder, f"session_{sid}", normalized_type)
            os.makedirs(out_dir, exist_ok=True)

            ts = int(time.time() * 1000)
            conf_str = f"{float(confidence):.2f}" if confidence is not None else "0.00"
            eng = self._safe_name(ocr_engine, 30)
            res = self._safe_name(ocr_result, 30)
            cls = self._safe_name(class_name, 30)

            base = f"{ts}_cam{camera_id}_{cls}_conf{conf_str}_{eng}_{res}"

            raw_path = os.path.join(out_dir, base + "_RAW.jpg")
            pre_path = os.path.join(out_dir, base + "_PRE.jpg")
            meta_path = os.path.join(out_dir, base + ".txt")
            
            capture_path = self._save_capture_frame(
                source_frame_bgr,
                camera_id,
                session_id,
                capture_name=base,
                normalized_type=normalized_type,
            )

            if roi_raw_bgr is not None and getattr(roi_raw_bgr, "size", 0) > 0:
                cv2.imwrite(raw_path, roi_raw_bgr)
            if roi_pre_bgr is not None and getattr(roi_pre_bgr, "size", 0) > 0:
                cv2.imwrite(pre_path, roi_pre_bgr)

            x1, y1, x2, y2 = (bbox_xyxy or (None, None, None, None))
            rx1, ry1, rx2, ry2 = (roi_bbox_xyxy or (None, None, None, None))
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(f"ts_ms={ts}\n")
                f.write(f"date={date_folder}\n")
                f.write(f"camera_id={camera_id}\n")
                f.write(f"session_id={sid}\n")
                f.write(f"class_name={class_name}\n")
                f.write(f"normalized_type={normalized_type}\n")
                f.write(f"confidence={confidence}\n")
                f.write(f"bbox_xyxy={x1},{y1},{x2},{y2}\n")
                f.write(f"roi_bbox_xyxy={rx1},{ry1},{rx2},{ry2}\n")
                f.write(f"ocr_engine={ocr_engine}\n")
                f.write(f"ocr_result={ocr_result}\n")
                f.write(f"capture_frame={capture_path or ''}\n")

        except Exception as e:
            print(f"[DEBUG ROI] save failed: {e}")

    def _sync_session_state(self, session):
        sid = None
        if session is not None:
            sid = getattr(session, "active_session_id", None)
            if sid is None:
                sid = getattr(session, "session_id", None)

        if sid != self._last_session_id:
            self._bbox_state.clear()
            self._session_ocr_count.clear()
            self._last_session_id = sid

    def _bbox_key(self, detection_type: str, x1: int, y1: int, x2: int, y2: int) -> str:
        grid = int(self.DEDUP_GRID.get(detection_type, 50))
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        bw_b = bw // (grid * 2)
        bh_b = bh // (grid * 2)
        return f"{detection_type}:{cx // grid}:{cy // grid}:{bw_b}:{bh_b}"

    def _should_process_bbox(self, key: str) -> bool:
        st = self._bbox_state.get(key)
        if not st:
            return True

        if st.get("success"):
            return False

        attempts = int(st.get("attempts", 0))
        if attempts >= int(self.MAX_RETRY_PER_BBOX):
            return False

        last_ts = float(st.get("last_ts", 0.0))
        if (time.time() - last_ts) < float(self.RETRY_COOLDOWN_SEC):
            return False

        return True

    def _mark_bbox_attempt(self, key: str, text: str, engine: str):
        st = self._bbox_state.get(
            key,
            {"attempts": 0, "success": False, "last_ts": 0.0, "last_engine": "", "last_text": ""},
        )
        st["attempts"] = int(st.get("attempts", 0)) + 1
        st["last_ts"] = time.time()
        st["last_engine"] = engine or ""
        st["last_text"] = text or ""
        st["success"] = (text not in INVALID_OCR)
        self._bbox_state[key] = st

    def _build_detections_text(self, detected_objects: list) -> str:
        """
        Membangun detections_text dengan format bersih tanpa engine name
        """
        container_value = None
        truck_value = None
        
        for obj in detected_objects:
            # Format obj: "Container_ID: ABC123 (easyocr)" atau "Container_ID: ABC123"
            if ":" not in obj:
                continue
                
            # Pisahkan label dan value
            label_part, rest = obj.split(":", 1)
            label_part = label_part.strip().lower()
            
            # Ambil value sebelum tanda kurung (engine name)
            value_part = rest.split("(")[0].strip() if "(" in rest else rest.strip()
            
            # Skip jika value tidak valid
            if not value_part or value_part.upper() in {"TIDAK_TERBACA", "ERROR", "ROI_INVALID"}:
                continue
            
            # Assign ke field yang sesuai
            if "container_id" in label_part:
                container_value = value_part
            elif "truck" in label_part or "plate" in label_part:
                truck_value = value_part
        
        # Bangun hasil hanya dengan value, tanpa engine name
        result_parts = []
        if container_value:
            result_parts.append(f"Container_ID: {container_value}")
        if truck_value:
            result_parts.append(f"truck_id: {truck_value}")
        
        return " | ".join(result_parts)

    def process_frame(self, frame, camera_id: int = 0, session=None):
        processed_frame = frame.copy()
        start_time = time.time()

        session_active = False
        if session is not None:
            try:
                session_active = bool(session.is_active())
            except Exception:
                session_active = getattr(session, "active_session_id", None) is not None

        self._sync_session_state(session)

        sess_txt = "Session: ACTIVE" if session_active else "Session: INACTIVE"
        cv2.putText(
            processed_frame,
            sess_txt,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if session_active else (0, 0, 255),
            2,
        )
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            processed_frame,
            timestamp,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2,
        )

        detections_text = ""
        detected_objects = []

        try:
            if self.model is None:
                return processed_frame, detections_text

            results = self.model.predict(frame, conf=0.55, verbose=False)

            # Simpan capture frame sekali per frame jika ada deteksi
            if any(len(r.boxes) > 0 for r in results if r.boxes is not None):
                _capture_sid = getattr(session, "active_session_id", None) if session is not None else None
                self._save_capture_frame(frame, camera_id, _capture_sid)

            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    normalized_type = self.normalize_detection_type(class_name)

                    x1_exp, y1_exp, x2_exp, y2_exp = self.expand_roi_based_on_class(
                        x1, y1, x2, y2, frame.shape, class_name
                    )

                    color = self.get_color_for_class(class_name)
                    cv2.rectangle(processed_frame, (x1_exp, y1_exp), (x2_exp, y2_exp), color, 2)
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

                    label = f"{class_name} {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(
                        processed_frame,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1),
                        color,
                        -1,
                    )
                    cv2.putText(
                        processed_frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                    if db_handler:
                        roi_coords = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                        try:
                            db_handler.log_detection(
                                camera_id=camera_id,
                                detection_type=normalized_type,
                                detected_text=class_name,
                                confidence=confidence,
                                roi_coordinates=str(roi_coords),
                                processed_duration=time.time() - start_time,
                                ocr_result="",
                            )
                        except Exception as e:
                            print(f"DB log_detection (yolo-only) error: {e}")

                    if not session_active:
                        continue

                    min_conf = float(self.OCR_CONF_THRESH.get(normalized_type, 0.25))
                    if confidence < min_conf:
                        continue

                    roi = frame[y1_exp:y2_exp, x1_exp:x2_exp]
                    if roi is None or roi.size == 0:
                        continue

                    min_pixels = int(self.ROI_MIN_PIXELS.get(normalized_type, 5000))
                    if roi.size < min_pixels:
                        session_id = getattr(session, "active_session_id", None) if session is not None else None
                        self._save_roi_bundle(
                            roi_raw_bgr=roi,
                            roi_pre_bgr=roi,
                            source_frame_bgr=frame,
                            normalized_type=normalized_type,
                            class_name=class_name,
                            confidence=confidence,
                            camera_id=camera_id,
                            session_id=session_id,
                            bbox_xyxy=(x1, y1, x2, y2),
                            roi_bbox_xyxy=(x1_exp, y1_exp, x2_exp, y2_exp),
                            ocr_engine="skip_small_roi",
                            ocr_result="ROI_TOO_SMALL",
                        )
                        continue

                    ocr_count_so_far = self._session_ocr_count.get(normalized_type, 0)
                    ocr_limit = int(self.SESSION_OCR_LIMIT.get(normalized_type, 10))
                    if ocr_count_so_far >= ocr_limit:
                        continue

                    key = self._bbox_key(normalized_type, x1, y1, x2, y2)
                    if not self._should_process_bbox(key):
                        continue

                    prep_bgr = self.preprocess_for_ocr(roi, class_name)
                    ocr_input_bgr = prep_bgr if prep_bgr is not None else roi

                    ocr_result, ocr_engine = self.ocr.read(ocr_input_bgr, class_name)

                    self._mark_bbox_attempt(key, ocr_result, ocr_engine)

                    self._session_ocr_count[normalized_type] = ocr_count_so_far + 1

                    session_id = getattr(session, "active_session_id", None) if session is not None else None
                    self._save_roi_bundle(
                        roi_raw_bgr=roi,
                        roi_pre_bgr=ocr_input_bgr,
                        source_frame_bgr=frame,
                        normalized_type=normalized_type,
                        class_name=class_name,
                        confidence=confidence,
                        camera_id=camera_id,
                        session_id=session_id,
                        bbox_xyxy=(x1, y1, x2, y2),
                        roi_bbox_xyxy=(x1_exp, y1_exp, x2_exp, y2_exp),
                        ocr_engine=ocr_engine,
                        ocr_result=ocr_result,
                    )

                    if ocr_result not in INVALID_OCR:
                        self.successful_reads.setdefault(class_name, []).append(ocr_result)

                    if session is not None:
                        try:
                            session.add_observation(
                                detection_type=normalized_type,
                                text=ocr_result,
                                confidence=confidence,
                                bbox=(x1, y1, x2, y2),
                            )
                        except Exception:
                            pass

                    if db_handler:
                        roi_coords = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                        try:
                            db_handler.log_detection(
                                camera_id=camera_id,
                                detection_type=normalized_type,
                                detected_text=ocr_result,
                                confidence=confidence,
                                roi_coordinates=str(roi_coords),
                                processed_duration=time.time() - start_time,
                                ocr_result=f"{ocr_result} | engine={ocr_engine}",
                            )
                        except Exception as e:
                            print(f"DB log_detection (ocr) error: {e}")

                    result_color = (0, 255, 0) if ocr_result not in INVALID_OCR else (0, 0, 255)
                    info = f"{class_name}: {ocr_result} ({ocr_engine})"
                    cv2.putText(
                        processed_frame,
                        info,
                        (x1_exp, y2_exp + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        result_color,
                        2,
                    )
                    detected_objects.append(info)

            # ============================================================
            # PERBAIKAN: Gunakan method baru untuk membangun detections_text
            # ============================================================
            if detected_objects:
                detections_text = self._build_detections_text(detected_objects)
            else:
                # Fallback ke successful_reads jika tidak ada deteksi baru
                latest_reads = []
                for cn, arr in self.successful_reads.items():
                    if arr:
                        latest_reads.append(f"{cn}: {arr[-1]}")
                if latest_reads:
                    detections_text = " | ".join(latest_reads[:2])

        except Exception as e:
            print(f"Error processing frame: {e}")
            import traceback
            traceback.print_exc()

        return processed_frame, detections_text

    def get_successful_reads(self):
        return self.successful_reads

    def clear_successful_reads(self):
        self.successful_reads = {name: [] for name in self.class_names.values()}
        print("Successful reads cleared")

    def normalize_detection_type(self, detection_type: str):
        type_mapping = {
            "container_id": "Container_ID",
            "container_id_vertical": "Container_ID_Vertikal",
            "container_id_vertikal": "Container_ID_Vertikal",
            "Container_ID": "Container_ID",
            "Container_ID_Vertikal": "Container_ID_Vertikal",
            "truck_id": "truck_id",
            "plate_number": "plate_number",
        }
        return type_mapping.get(detection_type, detection_type)