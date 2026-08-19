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
  CalibrationScreen.config_saved → show GoldenCycle
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
    QStackedWidget,
    QStatusBar,
    QWidget,
)

from data.config_handler import ConfigHandler
from data.database import DatabaseManager
from gui.calibration_screen import CalibrationScreen
from gui.camera_screen import CameraScreen
from gui.golden_cycle_screen import GoldenCycleScreen
from gui.monitor_screen import MonitorScreen
from gui.summary_screen import SummaryScreen

_DARK = """
QMainWindow, QWidget {
    background: #0d1b2a;
    color: #cfd8dc;
    font-family: Consolas, Monospace;
}
QMenuBar {
    background: #0a1520;
    color: #cfd8dc;
}
QMenuBar::item:selected { background: #1565c0; }
QMenu { background: #0f2233; color: #cfd8dc; border: 1px solid #1e3040; }
QMenu::item:selected { background: #1565c0; }
QStatusBar { background: #0a1520; color: #607d8b; }
"""

# Screen index constants
IDX_CAMERA      = 0
IDX_CALIBRATION = 1
IDX_GOLDEN      = 2
IDX_MONITOR     = 3
IDX_SUMMARY     = 4


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
        self.setStyleSheet(_DARK)

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
            lambda _: self._go(IDX_GOLDEN)
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

    def _build_status_bar(self) -> None:
        self._status_lbl = QLabel("พร้อมใช้งาน")
        self._status_lbl.setFont(QFont("Consolas", 8))
        self.statusBar().addWidget(self._status_lbl)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _go(self, idx: int) -> None:
        self._stack.setCurrentIndex(idx)

    def _on_golden_ready(self, golden_ref=None) -> None:
        self._go(IDX_MONITOR)
        try:
            self._monitor_screen.start_session(golden_ref)
        except Exception:
            self._monitor_screen.start_session()
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
