"""
gui/main_window.py
Main application window — QStackedWidget hosting all screens.

Screen indices
--------------
  0  CameraScreen       (camera selection)
  1  CalibrationScreen  (zone calibration)
  2  GoldenCycleScreen  (record golden cycles)
  3  MonitorScreen      (live monitoring)
  4  SummaryScreen      (AI summary after stop)

Navigation wiring
-----------------
  CameraScreen.camera_confirmed  → show Calibration
  CalibrationScreen.config_saved → check DB for existing golden → dialog or GoldenCycle
  GoldenCycleScreen.golden_ready → show Monitor + start_session()
  MonitorScreen.session_stopped  → show Summary + load_session()
  SummaryScreen.back_to_monitor  → show Monitor + start_session()
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QWidget,
)

from core.dtw_comparator import GoldenReference
from data.config_handler import ConfigHandler
from data.database import DatabaseManager
from gui.calibration_screen import CalibrationScreen
from gui.camera_screen import CameraScreen
from gui.ai_settings_screen import AISettingsScreen
from gui.detector_settings_screen import DetectorSettingsScreen
from gui.font_settings_screen import FontSettingsScreen
from gui.golden_cycle_screen import GoldenCycleScreen
from gui.monitor_screen import MonitorScreen
from gui.summary_screen import SummaryScreen
from gui import theme as _theme


def _build_stylesheet() -> str:
    """Build the global dark stylesheet from current theme constants."""
    return f"""
QMainWindow, QWidget {{
    background: #0d1b2a;
    color: #cfd8dc;
    font-family: "{_theme.FONT_FAMILY}", Tahoma, sans-serif;
    font-size: {_theme.FONT_SIZE_BODY}pt;
}}
QMenuBar {{
    background: #0a1520;
    color: #cfd8dc;
}}
QMenuBar::item:selected {{ background: #1565c0; }}
QMenu {{ background: #0f2233; color: #cfd8dc; border: 1px solid #1e3040; }}
QMenu::item:selected {{ background: #1565c0; }}
QStatusBar {{ background: #0a1520; color: #607d8b; }}
"""

# Screen index constants
IDX_CAMERA        = 0
IDX_CALIBRATION   = 1
IDX_GOLDEN        = 2
IDX_MONITOR       = 3
IDX_SUMMARY       = 4
IDX_FONT_SETTINGS       = 5
IDX_AI_SETTINGS         = 6
IDX_DETECTOR_SETTINGS   = 7


class MainWindow(QMainWindow):
    def __init__(
        self,
        config: ConfigHandler,
        db: DatabaseManager,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._db     = db
        self.setWindowTitle("AI Production Cycle Monitor")
        self.setMinimumSize(1280, 780)
        self.setStyleSheet(_build_stylesheet())

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._build_screens()
        self._build_menu()
        self._build_status_bar()

        # Start on camera screen
        self._stack.setCurrentIndex(IDX_CAMERA)

    # ------------------------------------------------------------------
    # Screen construction
    # ------------------------------------------------------------------

    def _build_screens(self) -> None:
        # 0 — Camera
        self._camera_screen = CameraScreen(self._config)
        self._stack.addWidget(self._camera_screen)
        self._camera_screen.camera_saved.connect(
            lambda: self._go(IDX_CALIBRATION)
        )

        # 1 — Calibration
        self._calibration_screen = CalibrationScreen(self._config)
        self._stack.addWidget(self._calibration_screen)
        self._calibration_screen.calibration_saved.connect(
            self._on_calibration_saved
        )

        # 2 — Golden Cycle
        self._golden_screen = GoldenCycleScreen(self._config, self._db)
        self._stack.addWidget(self._golden_screen)
        self._golden_screen.golden_ready.connect(self._on_golden_ready)

        # 3 — Monitor
        self._monitor_screen = MonitorScreen(self._config, self._db)
        self._stack.addWidget(self._monitor_screen)
        self._monitor_screen.session_stopped.connect(self._on_session_stopped)

        # 4 — Summary
        self._summary_screen = SummaryScreen(self._config, self._db)
        self._stack.addWidget(self._summary_screen)
        self._summary_screen.back_to_monitor.connect(self._on_back_to_monitor)

        # 5 — Font Settings
        self._font_settings_screen = FontSettingsScreen(self._config)
        self._stack.addWidget(self._font_settings_screen)
        self._font_settings_screen.back_requested.connect(
            lambda: self._go(IDX_CAMERA)
        )
        self._font_settings_screen.settings_saved.connect(
            self._on_font_settings_saved
        )

        # 6 — AI Settings
        self._ai_settings_screen = AISettingsScreen(self._config)
        self._stack.addWidget(self._ai_settings_screen)
        self._ai_settings_screen.back_requested.connect(
            lambda: self._go(IDX_CAMERA)
        )

        # 7 — Detector Settings
        self._detector_settings_screen = DetectorSettingsScreen(self._config)
        self._stack.addWidget(self._detector_settings_screen)
        self._detector_settings_screen.back_requested.connect(
            lambda: self._go(IDX_CAMERA)
        )
        self._detector_settings_screen.config_saved.connect(
            self._monitor_screen.notify_detector_config_changed
        )
        self._detector_settings_screen.config_saved.connect(
            self._golden_screen.notify_detector_config_changed
        )

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        nav = menubar.addMenu("นำทาง")
        for label, idx in [
            ("🎥  กล้อง",         IDX_CAMERA),
            ("⚙️  Calibration",   IDX_CALIBRATION),
            ("🏅  Golden Cycle",  IDX_GOLDEN),
            ("🏭  Live Monitor",  IDX_MONITOR),
            ("📊  AI Summary",    IDX_SUMMARY),
        ]:
            act = QAction(label, self)
            act.triggered.connect(lambda checked, i=idx: self._go(i))
            nav.addAction(act)

        tools = menubar.addMenu("เครื่องมือ")
        act_rerecord = QAction("🔄  Re-record Golden Standard", self)
        act_rerecord.setToolTip("บันทึก Golden Standard ใหม่โดยไม่ต้องรีสตาร์ทแอป")
        act_rerecord.triggered.connect(self._rerecord_golden)
        tools.addAction(act_rerecord)

        act_fonts = QAction("🔤  ตั้งค่าฟ้อนต์", self)
        act_fonts.setToolTip("เปลี่ยน font family และขนาดตัวอักษรทั่วทั้งแอป")
        act_fonts.triggered.connect(lambda: self._go(IDX_FONT_SETTINGS))
        tools.addAction(act_fonts)

        act_ai = QAction("🤖  ตั้งค่า AI", self)
        act_ai.setToolTip("ปรับ Model, Prompt Template, Output Format")
        act_ai.triggered.connect(lambda: self._go(IDX_AI_SETTINGS))
        tools.addAction(act_ai)

        act_detector = QAction("🤚  ตั้งค่า Hand Detector", self)
        act_detector.setToolTip("เลือก Backend, Device (CPU/CUDA), ดู Status")
        act_detector.triggered.connect(lambda: self._go(IDX_DETECTOR_SETTINGS))
        tools.addAction(act_detector)

    def _build_status_bar(self) -> None:
        self._status_lbl = QLabel("พร้อมใช้งาน")
        self._status_lbl.setFont(_theme.make_font(_theme.FONT_SIZE_CAPTION))
        self.statusBar().addWidget(self._status_lbl)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _go(self, idx: int) -> None:
        self._stack.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # After calibration: check for existing golden standard
    # ------------------------------------------------------------------

    def _on_calibration_saved(self, cfg: dict) -> None:
        station_id = cfg.get("station_id", self._config.active_station) or ""
        gc_row = None
        if station_id:
            try:
                gc_row = self._db.load_golden_cycle(station_id)
            except Exception:
                gc_row = None

        if gc_row is not None:
            # Ask the user whether to reuse or re-record
            msg = QMessageBox(self)
            msg.setWindowTitle("Golden Standard พบข้อมูลเดิม")
            msg.setText(
                f"<b>Station: {station_id}</b><br><br>"
                f"พบ Golden Standard ที่บันทึกไว้แล้ว "
                f"({gc_row.num_source_cycles} cycles, "
                f"บันทึกเมื่อ {gc_row.recorded_at.strftime('%Y-%m-%d %H:%M') if gc_row.recorded_at else '—'})<br><br>"
                "ต้องการทำอะไร?"
            )
            msg.setIcon(QMessageBox.Question)
            btn_use    = msg.addButton("ใช้ Golden Standard เดิม", QMessageBox.AcceptRole)
            btn_rerecord = msg.addButton("บันทึกใหม่", QMessageBox.RejectRole)
            msg.setDefaultButton(btn_use)
            msg.exec_()

            if msg.clickedButton() == btn_use:
                golden_ref = self._build_golden_ref_from_row(gc_row)
                self._go(IDX_MONITOR)
                self._monitor_screen.start_session(golden_ref)
                self._status_lbl.setText("กำลัง Monitor... (ใช้ Golden เดิม)")
                return

        # No existing golden — or user chose to re-record
        self._go(IDX_GOLDEN)

    @staticmethod
    def _build_golden_ref_from_row(gc_row) -> GoldenReference:
        """Reconstruct a GoldenReference from a GoldenCycle ORM row."""
        # zone_standard_times keys are stored as strings in JSON
        standard_times = {
            int(k): float(v)
            for k, v in (gc_row.zone_standard_times or {}).items()
        }
        zone_order = sorted(standard_times.keys())
        return GoldenReference(
            raw_trajectory      = gc_row.trajectory_points or [],
            standard_times      = standard_times,
            total_standard_time = float(gc_row.standard_total_sec or 0),
            zone_order          = zone_order,
            n_cycles_recorded   = int(gc_row.num_source_cycles or 0),
        )

    # ------------------------------------------------------------------
    # Re-record Golden Standard (from Tools menu or Monitor screen)
    # ------------------------------------------------------------------

    def _rerecord_golden(self) -> None:
        """Navigate to GoldenCycleScreen to record a fresh golden standard."""
        reply = QMessageBox.question(
            self,
            "Re-record Golden Standard",
            "ต้องการบันทึก Golden Standard ใหม่ใช่ไหม?\n"
            "ข้อมูล Golden เดิมจะถูกแทนที่เมื่อบันทึกเสร็จ",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._go(IDX_GOLDEN)

    def _on_golden_ready(self, golden_ref=None) -> None:
        self._go(IDX_MONITOR)
        self._monitor_screen.start_session(golden_ref)
        self._status_lbl.setText("กำลัง Monitor...")

    def _on_session_stopped(self, session_id: int) -> None:
        self._go(IDX_SUMMARY)
        self._summary_screen.load_session(session_id)
        self._status_lbl.setText(f"สรุปผล Session #{session_id}")

    def _on_back_to_monitor(self) -> None:
        self._go(IDX_MONITOR)
        self._monitor_screen.start_session()
        self._status_lbl.setText("กำลัง Monitor...")

    def _on_font_settings_saved(self, settings: dict) -> None:
        """Apply saved font settings and refresh the global stylesheet."""
        _theme.apply_font_settings(settings)
        self.setStyleSheet(_build_stylesheet())
        self._status_lbl.setFont(_theme.make_font(_theme.FONT_SIZE_CAPTION))
        self._status_lbl.setText(
            "OK: font '" + settings.get("font_family", "") + "' -- restart to apply all screens"
        )
