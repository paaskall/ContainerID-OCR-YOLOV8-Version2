class ModernStyle:
    @staticmethod
    def get_stylesheet():
        return """
        /* =========================================================
           PALETTE
           --bg-base:      #F7F7F7   (surface utama)
           --bg-card:      #FFFFFF   (card / panel)
           --bg-subtle:    #EFEFEF   (input, hover ringan)
           --border:       #E0E0E0
           --border-focus: #3D3D3D
           --text-primary: #1A1A1A
           --text-muted:   #7A7A7A
           --text-faint:   #ABABAB
           --accent:       #1A1A1A   (tombol utama, badge aktif)
           --danger:       #C0392B
           --success:      #2E7D32
           ========================================================= */

        /* ── Base ─────────────────────────────────────────────── */
        QMainWindow {
            background-color: #F7F7F7;
        }

        QWidget {
            background-color: #F7F7F7;
            color: #1A1A1A;
            font-family: 'Segoe UI', 'SF Pro Text', Helvetica, sans-serif;
            font-size: 13px;
        }

        QLabel {
            background-color: transparent;
            color: #1A1A1A;
        }

        QLabel#title {
            font-size: 18px;
            font-weight: 700;
            color: #1A1A1A;
            letter-spacing: 0.3px;
        }

        QLabel#status {
            font-size: 12px;
            font-weight: 600;
            padding: 6px 12px;
            border-radius: 6px;
            background-color: #EFEFEF;
            color: #1A1A1A;
            border: 1px solid #E0E0E0;
        }

        /* ── GroupBox / Card ───────────────────────────────────── */
        QGroupBox {
            background-color: #FFFFFF;
            border: 1px solid #E0E0E0;
            border-radius: 10px;
            margin-top: 14px;
            padding: 16px;
            font-weight: 600;
            color: #1A1A1A;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 5px;
            color: #1A1A1A;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        /* ── Scroll ────────────────────────────────────────────── */
        QScrollArea {
            border: none;
            background-color: transparent;
        }

        QScrollBar:vertical {
            background: transparent;
            width: 8px;
            margin: 2px;
        }

        QScrollBar::handle:vertical {
            background: #D4D4D4;
            border-radius: 4px;
            min-height: 24px;
        }

        QScrollBar::handle:vertical:hover {
            background: #ABABAB;
        }

        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            height: 0px;
        }

        QScrollBar:horizontal {
            background: transparent;
            height: 8px;
            margin: 2px;
        }

        QScrollBar::handle:horizontal {
            background: #D4D4D4;
            border-radius: 4px;
            min-width: 24px;
        }

        QScrollBar::handle:horizontal:hover {
            background: #ABABAB;
        }

        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal {
            width: 0px;
        }

        /* ── Buttons ───────────────────────────────────────────── */
        QPushButton {
            background-color: #1A1A1A;
            color: #FFFFFF;
            border: none;
            padding: 9px 18px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 12px;
            letter-spacing: 0.3px;
            min-height: 36px;
        }

        QPushButton:hover {
            background-color: #2E2E2E;
        }

        QPushButton:pressed {
            background-color: #444444;
        }

        QPushButton:disabled {
            background-color: #EFEFEF;
            color: #ABABAB;
        }

        /* ── Danger Button ─────────────────────────────────────── */
        QPushButton#Danger {
            background-color: #C0392B;
            color: #FFFFFF;
        }

        QPushButton#Danger:hover {
            background-color: #A93226;
        }

        QPushButton#Danger:pressed {
            background-color: #922B21;
        }

        /* ── Ghost Button ──────────────────────────────────────── */
        QPushButton#Ghost {
            background-color: transparent;
            border: 1px solid #D4D4D4;
            color: #3D3D3D;
        }

        QPushButton#Ghost:hover {
            background-color: #EFEFEF;
            border-color: #ABABAB;
        }

        QPushButton#Ghost:pressed {
            background-color: #E0E0E0;
        }

        /* ── Inputs ────────────────────────────────────────────── */
        QLineEdit,
        QComboBox,
        QTextEdit {
            background-color: #FFFFFF;
            border: 1px solid #E0E0E0;
            border-radius: 8px;
            padding: 7px 11px;
            color: #1A1A1A;
            selection-background-color: #D4D4D4;
        }

        QLineEdit:focus,
        QComboBox:focus,
        QTextEdit:focus {
            border: 1px solid #3D3D3D;
            background-color: #FFFFFF;
        }

        QLineEdit:disabled,
        QComboBox:disabled,
        QTextEdit:disabled {
            background-color: #F7F7F7;
            color: #ABABAB;
            border-color: #EFEFEF;
        }

        QComboBox::drop-down {
            border: none;
            width: 20px;
        }

        QComboBox::down-arrow {
            width: 10px;
            height: 10px;
        }

        QComboBox QAbstractItemView {
            background-color: #FFFFFF;
            border: 1px solid #E0E0E0;
            border-radius: 6px;
            selection-background-color: #EFEFEF;
            selection-color: #1A1A1A;
            padding: 4px;
        }

        /* ── Table ─────────────────────────────────────────────── */
        QTableWidget {
            background-color: #FFFFFF;
            border: 1px solid #E0E0E0;
            border-radius: 10px;
            gridline-color: #F0F0F0;
            selection-background-color: #EFEFEF;
            selection-color: #1A1A1A;
        }

        QTableWidget::item {
            padding: 8px 10px;
            border: none;
            color: #1A1A1A;
        }

        QTableWidget::item:selected {
            background-color: #EFEFEF;
            color: #1A1A1A;
        }

        QHeaderView::section {
            background-color: #F7F7F7;
            border: none;
            border-bottom: 1px solid #E0E0E0;
            padding: 8px 10px;
            font-weight: 700;
            font-size: 11px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            color: #7A7A7A;
        }

        /* ── Tab ───────────────────────────────────────────────── */
        QTabWidget::pane {
            border: 1px solid #E0E0E0;
            border-radius: 10px;
            background-color: #FFFFFF;
        }

        QTabBar::tab {
            background-color: transparent;
            color: #7A7A7A;
            padding: 8px 18px;
            margin-right: 2px;
            border: none;
            font-weight: 600;
            font-size: 12px;
        }

        QTabBar::tab:selected {
            color: #1A1A1A;
            border-bottom: 2px solid #1A1A1A;
        }

        QTabBar::tab:hover:!selected {
            color: #3D3D3D;
        }

        /* ── Splitter ──────────────────────────────────────────── */
        QSplitter::handle {
            background-color: #E0E0E0;
        }

        QSplitter::handle:horizontal {
            width: 1px;
        }

        QSplitter::handle:vertical {
            height: 1px;
        }

        /* ── Status Bar ────────────────────────────────────────── */
        QStatusBar {
            background-color: #FFFFFF;
            border-top: 1px solid #E0E0E0;
            color: #7A7A7A;
            font-size: 12px;
        }

        QStatusBar QLabel {
            color: #7A7A7A;
            font-weight: 500;
            padding: 0 6px;
        }

        /* ── ToolTip ───────────────────────────────────────────── */
        QToolTip {
            background-color: #1A1A1A;
            color: #FFFFFF;
            border: none;
            border-radius: 5px;
            padding: 5px 9px;
            font-size: 12px;
        }

        /* ── CheckBox & RadioButton ────────────────────────────── */
        QCheckBox,
        QRadioButton {
            color: #3D3D3D;
            spacing: 7px;
        }

        QCheckBox::indicator,
        QRadioButton::indicator {
            width: 15px;
            height: 15px;
            border: 1.5px solid #D4D4D4;
            border-radius: 4px;
            background-color: #FFFFFF;
        }

        QCheckBox::indicator:checked {
            background-color: #1A1A1A;
            border-color: #1A1A1A;
        }

        QRadioButton::indicator {
            border-radius: 8px;
        }

        QRadioButton::indicator:checked {
            background-color: #1A1A1A;
            border-color: #1A1A1A;
        }

        /* ── Progress Bar ──────────────────────────────────────── */
        QProgressBar {
            background-color: #EFEFEF;
            border: none;
            border-radius: 4px;
            height: 6px;
            text-align: center;
            color: transparent;
        }

        QProgressBar::chunk {
            background-color: #1A1A1A;
            border-radius: 4px;
        }

        /* ── Menu ──────────────────────────────────────────────── */
        QMenuBar {
            background-color: #FFFFFF;
            border-bottom: 1px solid #E0E0E0;
            padding: 2px;
        }

        QMenuBar::item {
            padding: 5px 12px;
            border-radius: 5px;
            color: #3D3D3D;
        }

        QMenuBar::item:selected {
            background-color: #EFEFEF;
            color: #1A1A1A;
        }

        QMenu {
            background-color: #FFFFFF;
            border: 1px solid #E0E0E0;
            border-radius: 8px;
            padding: 4px;
        }

        QMenu::item {
            padding: 7px 16px;
            border-radius: 5px;
            color: #3D3D3D;
        }

        QMenu::item:selected {
            background-color: #EFEFEF;
            color: #1A1A1A;
        }

        QMenu::separator {
            height: 1px;
            background-color: #E0E0E0;
            margin: 4px 8px;
        }
        """