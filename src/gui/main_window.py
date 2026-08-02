import sys
import time
import queue
import threading

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGridLayout, QGroupBox,
    QStatusBar, QScrollArea
)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QImage, QPixmap
import cv2
import numpy as np

from .camera_widget import CameraWidget
from .styles import ModernStyle

from src.core.camera_manager import CameraManager
from src.core.plate_recognizer import PlateRecognizer

from src.hardware.scale_interface import ScaleInterface
from src.core.session_aggregator import SessionAggregator, VehiclePresenceMonitor
from src.core.database_handler import db_handler


class FrameWorker(QObject):

    frame_ready = pyqtSignal(int, bytes, int, int, str, float)

    def __init__(self, camera_id: int, camera_manager: CameraManager,
                 plate_recognizer: PlateRecognizer, session_ref):
        super().__init__()
        self.camera_id       = camera_id
        self.camera_manager  = camera_manager
        self.plate_recognizer = plate_recognizer
        self.session_ref     = session_ref

        self._running = False
        self._interval_ms = 80

    def start_worker(self):
        self._running = True
        self._loop()

    def stop_worker(self):
        self._running = False

    def _loop(self):
        while self._running:
            t_start = time.time()
            try:
                self._process_one_frame()
            except Exception as e:
                print(f"[Worker cam{self.camera_id}] error: {e}")

            elapsed = (time.time() - t_start) * 1000
            sleep_ms = max(0.0, self._interval_ms - elapsed)
            time.sleep(sleep_ms / 1000.0)

    def _process_one_frame(self):
        if not self.camera_manager.get_camera_status(self.camera_id):
            return

        frame = self.camera_manager.get_frame(self.camera_id)
        if frame is None:
            return

        processed_frame, detections_text = self.plate_recognizer.process_frame(
            frame,
            camera_id=self.camera_id,
            session=self.session_ref,
        )

        fps = self.camera_manager.get_camera_fps(self.camera_id)

        try:
            small = cv2.resize(processed_frame, (640, 360))
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            raw_bytes = rgb.tobytes()
        except Exception as e:
            print(f"[Worker cam{self.camera_id}] frame convert error: {e}")
            return

        self.frame_ready.emit(
            self.camera_id,
            raw_bytes, w, h,
            detections_text or "",
            float(fps),
        )


class MainWindow(QMainWindow):

    NUM_CAMERAS = 6

    def __init__(self):
        super().__init__()

        self.camera_manager   = CameraManager()
        self.plate_recognizer = PlateRecognizer()

        self.scale = ScaleInterface(
            url="http://10.35.53.55/autogate/public/py/G2.txt",
            poll_interval=1.5,
            timeout=1.0,
            threshold_start_kg=1100.0,
            threshold_end_kg=900.0,
            stable_samples=8,
            stable_tolerance_kg=30.0,
        )

        self.session = SessionAggregator(
            db_handler=db_handler,
            session_window_sec=30,
            max_keepalive_sec=90,
            min_gap_between_sessions_sec=1.0,
        )

        self.presence = VehiclePresenceMonitor(
            monitored_cameras={
                3: "primary",
                0: "secondary",
            },
            absent_confirm_sec=3.0,
            present_confirm_sec=0.5,
            min_present_sec=2.0,
        )
        self.presence.on_vehicle_absent  = self._on_vehicle_absent
        self.presence.on_vehicle_present = self._on_vehicle_present

        self._workers: list[FrameWorker] = []
        self._threads: list[QThread]     = []

        self._detection_counts = [0] * self.NUM_CAMERAS

        self.init_ui()
        self._scale_timer = QTimer()
        self._scale_timer.timeout.connect(self._tick_scale_and_session)
        self._scale_timer.start(100)

    def init_ui(self):
        self.setWindowTitle("AutoGate System - G2")
        self.setGeometry(80, 60, 1500, 950)
        self.setStyleSheet(ModernStyle.get_stylesheet())

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(14, 14, 14, 10)
        main_layout.setSpacing(12)
        central_widget.setLayout(main_layout)

        main_layout.addWidget(self.create_header())
        main_layout.addWidget(self.create_camera_section())
        main_layout.addWidget(self.create_control_panel())
        self.setup_status_bar()

    def create_header(self):
        header_widget = QWidget()
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(12)
        header_widget.setLayout(header_layout)

        # Container widget untuk logo dengan margin left
        logo_container = QWidget()
        logo_layout = QHBoxLayout()
        logo_layout.setContentsMargins(15, 0, 0, 0)
        logo_container.setLayout(logo_layout)
        
        logo_label = QLabel()
        logo_pixmap = QPixmap("assets/icons/pt-pelindo.png")
        logo_pixmap = logo_pixmap.scaled(250, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(logo_pixmap)
        logo_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        
        logo_layout.addWidget(logo_label)
        logo_layout.addStretch()  # agar logo tetap di kiri

        status_group = QGroupBox("System Status")
        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(12, 14, 12, 12)
        status_layout.setSpacing(8)

        self.system_status = QLabel("READY - 6 CCTV CONFIGURED")
        self.system_status.setObjectName("status")
        self.system_status.setStyleSheet("background-color: #2ecc71; color: white;")

        self.camera_count = QLabel("Active Cameras: 0/6")
        self.camera_count.setObjectName("camera_count")
        self.camera_count.setStyleSheet("color: #cfe0f2; font-weight: 800;")

        self.scale_status = QLabel("Scale: -")
        self.scale_status.setStyleSheet("color: #9aa7b2; font-weight: 700;")

        status_layout.addWidget(self.system_status)
        status_layout.addWidget(self.camera_count)
        status_layout.addWidget(self.scale_status)
        status_group.setLayout(status_layout)

        header_layout.addWidget(logo_container, 65)
        header_layout.addWidget(status_group, 35)
        return header_widget

    def create_camera_section(self):
        section_widget = QWidget()
        section_layout = QVBoxLayout()
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.setSpacing(10)
        section_widget.setLayout(section_layout)

        scroll_area = QScrollArea()
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setWidgetResizable(True)

        scroll_widget = QWidget()
        self.camera_grid = QGridLayout()
        self.camera_grid.setSpacing(12)
        self.camera_grid.setContentsMargins(12, 12, 12, 12)
        scroll_widget.setLayout(self.camera_grid)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setMinimumHeight(560)
        scroll_area.setMaximumHeight(680)

        self.camera_widgets: list[CameraWidget] = []
        camera_names = [
            "CCTV 1 - ENTRANCE",
            "CCTV 2 - EXIT",
            "CCTV 3 - WEIGHT BRIDGE",
            "CCTV 4 - CONTAINER AREA",
            "CCTV 5 - TRUCK PARKING",
            "CCTV 6 - OVERVIEW",
        ]

        for i, name in enumerate(camera_names):
            row = i // 3
            col = i % 3
            camera_widget = CameraWidget(i, name)
            camera_widget.setMaximumSize(460, 360)
            self.camera_widgets.append(camera_widget)
            self.camera_grid.addWidget(camera_widget, row, col)

        section_layout.addWidget(scroll_area)
        return section_widget

    def create_control_panel(self):
        panel = QGroupBox("System Controls")
        panel.setMaximumHeight(92)

        layout = QHBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(14, 12, 14, 12)

        self.start_all_btn = QPushButton("START CAMERAS")
        self.stop_all_btn  = QPushButton("STOP CAMERAS")
        self.stop_all_btn.setObjectName("Danger")

        self.start_all_btn.clicked.connect(self.start_all_cameras)
        self.stop_all_btn.clicked.connect(self.stop_all_cameras)

        layout.addWidget(self.start_all_btn)
        layout.addWidget(self.stop_all_btn)
        layout.addStretch()
        panel.setLayout(layout)
        return panel

    def setup_status_bar(self):
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)

        self.status_label    = QLabel("System initialized - Ready to start cameras")
        self.fps_label       = QLabel("Overall FPS: 0")
        self.detection_label = QLabel("Detections: 0")

        status_bar.addWidget(self.status_label, 70)
        status_bar.addPermanentWidget(self.fps_label)
        status_bar.addPermanentWidget(self.detection_label)

    def start_all_cameras(self):
        self.status_label.setText("Starting cameras...")
        active_count = 0

        for i, widget in enumerate(self.camera_widgets):
            if self.camera_manager.start_camera(i):
                active_count += 1
                widget.update_status("Connected", "#2ecc71")
            else:
                widget.update_status("Failed", "#e74c3c")

        self._start_workers()

        self.system_status.setText(f"RUNNING - {active_count}/6 CAMERAS ACTIVE")
        self.camera_count.setText(f"Active Cameras: {active_count}/6")

        if active_count > 0:
            self.system_status.setStyleSheet("background-color: #2ecc71; color: white;")
            self.status_label.setText(f"{active_count} cameras active and streaming")
        else:
            self.system_status.setStyleSheet("background-color: #e74c3c; color: white;")
            self.status_label.setText("All cameras failed to start")

    def _start_workers(self):
        """Buat dan mulai FrameWorker per kamera di QThread masing-masing."""
        self._stop_workers()

        for i in range(self.NUM_CAMERAS):
            if not self.camera_manager.get_camera_status(i):
                continue

            worker = FrameWorker(
                camera_id=i,
                camera_manager=self.camera_manager,
                plate_recognizer=self.plate_recognizer,
                session_ref=self.session,
            )
            thread = QThread()
            worker.moveToThread(thread)

            worker.frame_ready.connect(self._on_frame_ready)
            thread.started.connect(worker.start_worker)

            thread.start()

            self._workers.append(worker)
            self._threads.append(thread)

    def _stop_workers(self):
        """Hentikan semua worker thread dengan bersih."""
        if hasattr(self, "presence"):
            self.presence.reset()
        for worker in self._workers:
            try:
                worker.stop_worker()
            except Exception:
                pass

        for thread in self._threads:
            try:
                thread.quit()
                thread.wait(2000)
            except Exception:
                pass

        self._workers.clear()
        self._threads.clear()

    def stop_all_cameras(self):
        self._stop_workers()
        self.camera_manager.stop_all_cameras()

        for widget in self.camera_widgets:
            widget.update_status("Disconnected", "#e74c3c")
            widget.clear_frame()

        self.system_status.setText("STOPPED")
        self.system_status.setStyleSheet("background-color: #e74c3c; color: white;")
        self.camera_count.setText("Active Cameras: 0/6")
        self.status_label.setText("All cameras stopped")
        self._detection_counts = [0] * self.NUM_CAMERAS

    def _on_frame_ready(
        self,
        camera_id: int,
        raw_bytes: bytes,
        width: int,
        height: int,
        detections_text: str,
        fps: float,
    ):
        """Dipanggil di main thread via Qt signal-slot (thread-safe)."""
        if camera_id >= len(self.camera_widgets):
            return

        widget = self.camera_widgets[camera_id]

        try:
            qt_image = QImage(raw_bytes, width, height, width * 3, QImage.Format_RGB888)
            pixmap   = QPixmap.fromImage(qt_image)
        except Exception as e:
            print(f"[Main] QPixmap error cam{camera_id}: {e}")
            pixmap = None

        detections_dict = self._parse_detections(detections_text)
        widget.update_frame(pixmap, detections_dict)
        widget.update_fps(fps)

        has_det = bool(detections_text)
        self._detection_counts[camera_id] = 1 if has_det else 0

        # Feed presence monitor untuk cam 0 dan cam 3
        if camera_id in (0, 3):
            self.presence.update(camera_id, has_det)

        total = sum(self._detection_counts)
        self.detection_label.setText(f"Detections: {total}")

    def _tick_scale_and_session(self):
        try:
            raw, kg, ts, last_err, last_ok = self.scale.get_latest()
            scale_online = getattr(self.scale, "is_online", False)
            if not scale_online:
                err_txt = (last_err or "offline").splitlines()[0]
                if len(err_txt) > 64:
                    err_txt = err_txt[:61] + "..."
                self.scale_status.setText(f"Scale: OFFLINE ({err_txt})")
            elif kg is None:
                self.scale_status.setText(f"Scale: {raw or '-'}")
            else:
                self.scale_status.setText(f"Scale: {kg:.0f} kg")
        except Exception:
            pass

        try:
            while True:
                ev = self.scale.pop_event()
                if ev is None:
                    break
                if ev.kind == "START":
                    sid = self.session.start(
                        camera_id=2,
                        weight_raw=ev.weight_raw,
                        weight_kg=ev.weight_kg,
                        scale_id="jembatan_timbangan_2",
                    )
                    if sid is None:
                        # session.start() gagal (DB error / min_gap) →
                        # rearm scale agar tidak terjebak di _in_session=True
                        print("[Main] session.start() gagal → rearm scale")
                        self.scale.rearm()
                    else:
                        self.status_label.setText(f"SESSION START #{sid} | weight={ev.weight_kg} kg")
                        self.presence.reset()
                        for widget in self.camera_widgets:
                            widget.reset_for_new_session()
                elif ev.kind == "END":
                    finalized = self.session.finalize(force=True)
                    if finalized:
                        sid, summary = finalized
                        self.status_label.setText(
                            f"SESSION END {sid} | "
                            f"H={summary.get('best_container_h')} | "
                            f"V={summary.get('best_container_v')} | "
                            f"TR={summary.get('best_truck_id')} | "
                            f"PL={summary.get('best_plate_number')}"
                        )
                        self._broadcast_session_result(sid, summary)
                    else:
                        self.status_label.setText("SESSION END (no active session)")
        except Exception as e:
            print(f"[Main] scale event handling error: {e}")

        try:
            raw, kg, *_ = self.scale.get_latest()
            if kg is not None and kg >= 500 and self.session.is_active():
                was_active = self.session.is_active()
                self.session.keep_alive(extra_sec=2.0)

                # Jika keep_alive memaksa finalize karena max_keepalive_sec,
                # ScaleInterface masih bisa berada di state _in_session=True.
                # Rearm diperlukan agar kendaraan berikutnya bisa memicu START baru.
                if was_active and (not self.session.is_active()) and getattr(self.scale, "in_session", False):
                    print("[Main] session finalized by keep_alive/max timeout → rearm scale")
                    self.scale.rearm()
            else:
                # Safety net: jika session sudah tidak aktif tetapi scale masih merasa
                # in_session, reset scale agar START berikutnya tidak macet.
                if (not self.session.is_active()) and getattr(self.scale, "in_session", False):
                    print("[Main] session inactive while scale.in_session=True → rearm scale")
                    self.scale.rearm()
        except Exception:
            pass

    # ── Presence Monitor Callbacks ───────────────────────────────────────────

    def _on_vehicle_absent(self, camera_id: int, role: str):
        """Dipanggil dari VehiclePresenceMonitor — pindah ke main thread."""
        from PyQt5.QtCore import QMetaObject, Qt as _Qt
        self._pending_absent_info = (camera_id, role)
        QMetaObject.invokeMethod(self, "_on_vehicle_absent_main", _Qt.QueuedConnection)

    from PyQt5.QtCore import pyqtSlot as _slot

    @_slot()
    def _on_vehicle_absent_main(self):
        info = getattr(self, "_pending_absent_info", None)
        if not info:
            return
        camera_id, role = info
        self._pending_absent_info = None

        if not self.session.is_active():
            return

        print(f"[PRESENCE] cam{camera_id}({role}) ABSENT → finalize session")
        finalized = self.session.finalize(force=True)
        if finalized:
            sid, summary = finalized
            self.status_label.setText(
                f"SESSION END (cam{camera_id}/{role}) #{sid} | "
                f"H={summary.get('best_container_h')} | "
                f"TR={summary.get('best_truck_id')}"
            )
            self._broadcast_session_result(sid, summary)

            # Finalize ini terjadi karena kamera/presence, bukan karena Scale END.
            # Tanpa rearm, ScaleInterface bisa tetap _in_session=True sehingga
            # tidak akan membuat START untuk kendaraan berikutnya.
            try:
                self.scale.rearm()
            except Exception as e:
                print(f"[Main] scale rearm after vehicle absent failed: {e}")

    def _on_vehicle_present(self, camera_id: int):
        print(f"[PRESENCE] cam{camera_id} → PRESENT")

    def _broadcast_session_result(self, session_id: int, summary: dict):
        """Setelah session finalize, tampilkan hasil OCR terbaik di semua camera widget."""
        container_id = (
            summary.get("best_container_h")
            or summary.get("best_container_v")
        )
        truck_id = summary.get("best_truck_id")

        for widget in self.camera_widgets:
            widget.update_session_result(
                session_id=session_id,
                container_id=container_id,
                truck_id=truck_id,
            )

        print(
            f"[GUI] Session #{session_id} broadcast → "
            f"container={container_id} | truck={truck_id}"
        )

    def _parse_detections(self, detections_text: str) -> dict:
        """Parse detections_text → dict { 'container_id': str|None, 'truck_id': str|None }"""
        result = {"container_id": None, "truck_id": None}
        if not detections_text:
            return result

        for part in detections_text.split(" | "):
            part = part.strip()
            if ":" not in part:
                continue
                
            key_raw, _, val_raw = part.partition(":")
            key_raw = key_raw.strip().lower()
            
            # Clean value: remove any parentheses and extra spaces
            val_clean = val_raw.strip()
            if "(" in val_clean:
                val_clean = val_clean.split("(")[0].strip()
            
            # Skip invalid values
            if not val_clean or val_clean.upper() in {"TIDAK_TERBACA", "ERROR", "ROI_INVALID", "-"}:
                continue

            # Map to result dictionary
            if "container_id" in key_raw:
                result["container_id"] = val_clean
            elif "truck" in key_raw or "plate" in key_raw:
                result["truck_id"] = val_clean
            
            # Debug print untuk verifikasi
            print(f"[DEBUG] Parsed: {key_raw} → {val_clean}")

        print(f"[DEBUG] Final result: {result}")  # Debug
        return result

    def closeEvent(self, event):
        try:
            self._scale_timer.stop()
        except Exception:
            pass

        try:
            if hasattr(self, "scale") and self.scale:
                self.scale.stop()
        except Exception as e:
            print(f"Scale stop error: {e}")

        self.stop_all_cameras()
        event.accept()
