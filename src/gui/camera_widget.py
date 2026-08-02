from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QFrame, QGridLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap


class CameraWidget(QWidget):
    _COLOR_LIVE_VALUE   = "#1A1A1A"   
    _COLOR_FINAL_VALUE  = "#1A1A1A"   
    _COLOR_PENDING      = "#ABABAB"   

    _BG_LIVE  = "#FFFFFF"
    _BG_FINAL = "#F7F7F7"             

    _BORDER_LIVE  = "#E0E0E0"
    _BORDER_FINAL = "#3D3D3D"         

    def __init__(self, camera_id, camera_name, parent=None):
        super().__init__(parent)
        self.camera_id   = camera_id
        self.camera_name = camera_name
        self._is_final   = False
        self.init_ui()

    def init_ui(self):
        root = QVBoxLayout()
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        root.addLayout(self._build_topbar())
        root.addWidget(self._build_video())
        root.addLayout(self._build_inforow())
        root.addWidget(self._build_detection_panel())

        self.setLayout(root)

    def _build_topbar(self):
        row = QHBoxLayout()
        row.setSpacing(8)

        self.title_label = QLabel(self.camera_name)
        self.title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.title_label.setStyleSheet(
            "font-weight: 700; font-size: 14px; color: #1A1A1A; letter-spacing: 0.2px;"
        )

        self.badge = QLabel("DISCONNECTED")
        self.badge.setAlignment(Qt.AlignCenter)
        self.badge.setFixedHeight(24)
        self.badge.setStyleSheet(self._badge_style("#C0392B"))

        row.addWidget(self.title_label, 70)
        row.addWidget(self.badge, 30)
        return row

    def _build_video(self):
        self.video_label = QLabel()
        self.video_label.setMinimumSize(400, 250)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background-color: #F0F0F0;
            border: 1px solid #E0E0E0;
            border-radius: 8px;
            color: #ABABAB;
            font-size: 12px;
            letter-spacing: 0.2px;
        """)
        self.video_label.setText("Camera Not Active\nClick 'Start All Cameras'")
        return self.video_label

    def _build_inforow(self):
        row = QHBoxLayout()
        row.setSpacing(6)

        self.status_dot = QLabel()
        self.status_dot.setFixedSize(8, 8)
        self.status_dot.setStyleSheet(
            "background-color: #C0392B; border-radius: 4px;"
        )

        self.status_text = QLabel("Disconnected")
        self.status_text.setStyleSheet(
            "color: #C0392B; font-weight: 600; font-size: 11px;"
        )

        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet(
            "color: #ABABAB; font-weight: 500; font-size: 11px;"
        )

        row.addWidget(self.status_dot)
        row.addWidget(self.status_text)
        row.addStretch()
        row.addWidget(self.fps_label)
        return row

    def _build_detection_panel(self):
        """
        1 panel dengan QGridLayout:
          Col 0 — label kecil abu (Container ID / Truck ID)
          Col 1 — titik dua
          Col 2 — nilai (berubah sesuai state)
          Col 3 — badge FINAL (tersembunyi saat live)
        """
        self.ocr_box = QFrame()
        self._set_panel_style(final=False)

        grid = QGridLayout()
        grid.setContentsMargins(14, 12, 14, 12)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        def _lbl(text):
            l = QLabel(text)
            l.setStyleSheet(
                "color: #7A7A7A; font-weight: 600; font-size: 11px;"
                "letter-spacing: 0.4px; text-transform: uppercase;"
            )
            return l

        def _sep():
            l = QLabel(":")
            l.setStyleSheet("color: #D4D4D4; font-size: 11px;")
            return l

        self.container_id_value = QLabel("-")
        self.container_id_value.setWordWrap(True)
        self.container_id_value.setStyleSheet(self._value_style(self._COLOR_PENDING))

        self.truck_id_value = QLabel("-")
        self.truck_id_value.setWordWrap(True)
        self.truck_id_value.setStyleSheet(self._value_style(self._COLOR_PENDING))

        self.final_badge = QLabel("FINAL")
        self.final_badge.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.final_badge.setStyleSheet("""
            color: #FFFFFF;
            font-weight: 700;
            font-size: 10px;
            letter-spacing: 0.4px;
            padding: 2px 7px;
            background-color: #1A1A1A;
            border-radius: 4px;
        """)
        self.final_badge.hide()

        self.session_id_label = QLabel("")
        self.session_id_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.session_id_label.setStyleSheet(
            "color: #ABABAB; font-size: 10px; font-weight: 400;"
        )

        grid.addWidget(_lbl("Container ID"),       0, 0)
        grid.addWidget(_sep(),                     0, 1)
        grid.addWidget(self.container_id_value,    0, 2)
        grid.addWidget(self.final_badge,           0, 3)

        grid.addWidget(_lbl("Truck ID"),           1, 0)
        grid.addWidget(_sep(),                     1, 1)
        grid.addWidget(self.truck_id_value,        1, 2)
        grid.addWidget(self.session_id_label,      1, 3)

        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 0)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(3, 0)

        self.ocr_box.setLayout(grid)
        return self.ocr_box

    def _badge_style(self, bg: str) -> str:
        return f"""
            background-color: {bg};
            color: white;
            font-weight: 700;
            font-size: 10px;
            letter-spacing: 0.5px;
            border-radius: 5px;
            padding: 3px 9px;
        """

    def _value_style(self, color: str) -> str:
        return (
            f"color: {color}; font-weight: 600; font-size: 13px;"
            "letter-spacing: 0.1px;"
        )

    def _set_panel_style(self, final: bool):
        bg     = self._BG_FINAL     if final else self._BG_LIVE
        border = self._BORDER_FINAL if final else self._BORDER_LIVE
        width  = "1.5px"            if final else "1px"
        self.ocr_box.setStyleSheet(f"""
            QFrame {{
                background-color: {bg};
                border: {width} solid {border};
                border-radius: 8px;
            }}
        """)

    def update_status(self, status: str, color: str):
        self.status_text.setText(status)
        self.status_text.setStyleSheet(
            f"color: {color}; font-weight: 600; font-size: 11px;"
        )
        self.status_dot.setStyleSheet(
            f"background-color: {color}; border-radius: 4px;"
        )
        self.badge.setText(status.upper())
        self.badge.setStyleSheet(self._badge_style(color))

    def update_frame(self, pixmap: QPixmap, detections: dict = None):
        if pixmap and not pixmap.isNull():
            self.video_label.setPixmap(
                pixmap.scaled(
                    self.video_label.width(),
                    self.video_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

        if self._is_final:
            return

        if detections and isinstance(detections, dict):
            container = detections.get("container_id") or "-"
            truck     = detections.get("truck_id")     or "-"
        else:
            container = "-"
            truck     = "-"

        self.container_id_value.setText(container)
        self.container_id_value.setStyleSheet(
            self._value_style(
                self._COLOR_LIVE_VALUE if container != "-" else self._COLOR_PENDING
            )
        )

        self.truck_id_value.setText(truck)
        self.truck_id_value.setStyleSheet(
            self._value_style(
                self._COLOR_LIVE_VALUE if truck != "-" else self._COLOR_PENDING
            )
        )

    def update_session_result(
        self,
        session_id,
        container_id: str = None,
        truck_id: str = None,
    ):
        
        self._is_final = True

        container = container_id or "-"
        truck     = truck_id     or "-"

        self.container_id_value.setText(container)
        self.container_id_value.setStyleSheet(
            self._value_style(
                self._COLOR_FINAL_VALUE if container != "-" else self._COLOR_PENDING
            )
        )

        self.truck_id_value.setText(truck)
        self.truck_id_value.setStyleSheet(
            self._value_style(
                self._COLOR_FINAL_VALUE if truck != "-" else self._COLOR_PENDING
            )
        )

        self.final_badge.show()
        sid_txt = f"#{session_id}" if session_id else ""
        self.session_id_label.setText(sid_txt)

        self._set_panel_style(final=True)

    def reset_for_new_session(self):
        self._is_final = False
        self.container_id_value.setText("-")
        self.container_id_value.setStyleSheet(self._value_style(self._COLOR_PENDING))
        self.truck_id_value.setText("-")
        self.truck_id_value.setStyleSheet(self._value_style(self._COLOR_PENDING))
        self.final_badge.hide()
        self.session_id_label.setText("")
        self._set_panel_style(final=False)

    def clear_frame(self):
        self.video_label.setText("Camera Stopped")
        self.fps_label.setText("FPS: 0.0")
        self.reset_for_new_session()

    def update_fps(self, fps):
        try:
            self.fps_label.setText(f"FPS: {float(fps):.1f}")
        except Exception:
            self.fps_label.setText("FPS: 0.0")