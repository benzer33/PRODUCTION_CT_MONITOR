"""
gui/monitor_screen.py
Live Production Monitor Screen.

Layout
------
┌──────────────────────────────────────────┬─────────────────────┐
│                                          │  ZONE STATUS PANEL  │
│          LIVE VIDEO FEED                 │  Zone 1: 2.1s/3.0s  │
│     (ghost overlay + zone polygons)      │  Zone 2: ░░░ 68%   │
│                                          │  Zone 3: waiting    │
│  Cycle #3  ██████████░░░░  65%           ├─────────────────────┤
│                                          │  CYCLE STATS        │
│  ⚠ Zone 2: 4.2s (+40% CRITICAL)         │  Total:  12 cycles  │
│                                          │  Pass:   10         │
│                                          │  Fail:    2         │
├──────────────────────────────────────────┤  Avg:   12.3s       │
│  [Stop & View Summary]                   │  Std:   11.1s       │
└──────────────────────────────────────────┴─────────────────────┘
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QFrame, QGroupBox, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QScrollArea,
    QVBoxLayout, QWidget,
)

from core.dtw_comparator import GoldenReference
from data.config_handler import ConfigHandler
from data.database import DatabaseManager
from gui.widgets.video_widget import VideoWidget
from gui.widgets.zone_widget import ZonePanel
from vision.vision_thread import VisionThread


# ---------------------------------------------------------------------------
# Alert ticker widget
# ---------------------------------------------------------------------------

class AlertTicker(QLabel):
    """Scrolling alert banner shown at the bottom of the video area."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFont(QFont("Consolas", 10, QFont.Bold))
        self.setAlignment(Qt.AlignCenter)
        self.setFixedHeight(30)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)
        self.hide()

    def show_alert(self, message: str, level: str = "WARNING") -> None:
        if level == "CRITICAL":
            self.setStyleSheet(
                "color: #ff1744; background-color: #3e0000; "
                "border: 1px solid #d50000; border-radius: 3px;"
            )
        else:
            self.setStyleSheet(
                "color: #ffab00; background-color: #3e2800; "
                "border: 1px solid #ffab00; border-radius: 3px;"
            )
        self.setText(f"  ⚠  {message}  ")
        self.show()
        self._hide_timer.start(5000)


# ---------------------------------------------------------------------------
# Cycle stat mini-card
# ---------------------------------------------------------------------------

class CycleSummaryCard(QGroupBox):
    """Small widget showing live session aggregate stats."""

    def __init__(self, parent=None) -> None:
        super().__init__("Session Stats", parent)
        self.setStyleSheet("""
            QGroupBox {
                color: #607d8b; border: 1px solid #263238;
                border-radius: 4px; margin-top: 8px;
                font-family: Consolas; font-size: 9px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
        """)
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        self._labels: dict[str, QLabel] = {}
        for key, display in [
            ("total",   "Total Cycles"),
            ("pass",    "Pass"),
            ("fail",    "Fail"),
            ("seq_err", "Seq. Errors"),
            ("avg",     "Avg Time"),
        ]:
            row = QHBoxLayout()
            lbl_k = QLabel(display + ":")
            lbl_k.setFont(QFont("Consolas", 8))
            lbl_k.setStyleSheet("color: #607d8b;")
            lbl_v = QLabel("—")
            lbl_v.setFont(QFont("Consolas", 9, QFont.Bold))
            lbl_v.setStyleSheet("color: #eceff1;")
            row.addWidget(lbl_k)
            row.addStretch()
            row.addWidget(lbl_v)
            layout.addLayout(row)
            self._labels[key] = lbl_v

    def update_stats(
        self,
        total: int, passed: int, failed: int,
        seq_errors: int, avg_time: float,
    ) -> None:
        self._labels["total"].setText(str(total))
        self._labels["pass"].setText(str(passed))
        self._labels["pass"].setStyleSheet(
            "color: #00c853; font-family: Consolas; font-size: 9px; font-weight: bold;"
        )
        self._labels["fail"].setText(str(failed))
        self._labels["fail"].setStyleSheet(
            f"color: {'#d50000' if failed else '#eceff1'}; "
            "font-family: Consolas; font-size: 9px; font-weight: bold;"
        )
        self._labels["seq_err"].setText(str(seq_errors))
        self._labels["avg"].setText(f"{avg_time:.2f}s" if avg_time else "—")


# ---------------------------------------------------------------------------
# Monitor Screen
# ---------------------------------------------------------------------------

class MonitorScreen(QWidget):
    """Live production monitoring screen with ghost overlay."""

    session_stopped = pyqtSignal(int)   # emits session_id when stopping

    def __init__(
        self,
        config: ConfigHandler,
        db: DatabaseManager,
        golden_ref: GoldenReference | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._config      = config
        self._db          = db
        self._golden_ref  = golden_ref
        self._vision_thread: VisionThread | None = None
        self._session_id: int | None = None
        self._cycle_id:   int | None = None
        self._cycle_number = 0

        # Aggregate stats
        self._total = self._pass = self._fail = self._seq_err = 0
        self._cycle_times: list[float] = []
        self._all_cycle_records: list[dict] = []

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # Header
        hdr = QHBoxLayout()
        self._lbl_title = QLabel("🏭  LIVE PRODUCTION MONITOR")
        self._lbl_title.setFont(QFont("Consolas", 14, QFont.Bold))
        self._lbl_title.setStyleSheet("color: #00bcd4;")
        hdr.addWidget(self._lbl_title)
        hdr.addStretch()

        self._lbl_session = QLabel("Session: —")
        self._lbl_session.setFont(QFont("Consolas", 9))
        self._lbl_session.setStyleSheet("color: #607d8b;")
        hdr.addWidget(self._lbl_session)
        root.addLayout(hdr)

        # Body
        body = QHBoxLayout()
        body.setSpacing(12)

        # ---- Left: video area ----
        video_col = QVBoxLayout()
        self._video = VideoWidget()
        self._video.setMinimumSize(640, 360)
        video_col.addWidget(self._video)

        # Progress bar (cycle %)
        prog_row = QHBoxLayout()
        self._lbl_cycle_num = QLabel("Cycle #0")
        self._lbl_cycle_num.setFont(QFont("Consolas", 9, QFont.Bold))
        self._lbl_cycle_num.setStyleSheet("color: #00bcd4;")
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(8)
        self._progress_bar.setStyleSheet("""
            QProgressBar { background:#152028; border-radius:4px; border:none; }
            QProgressBar::chunk { background:#1565c0; border-radius:4px; }
        """)
        self._lbl_state = QLabel("IDLE")
        self._lbl_state.setFont(QFont("Consolas", 9))
        self._lbl_state.setStyleSheet("color: #607d8b;")
        prog_row.addWidget(self._lbl_cycle_num)
        prog_row.addWidget(self._progress_bar, 1)
        prog_row.addWidget(self._lbl_state)
        video_col.addLayout(prog_row)

        # Alert ticker
        self._alert_ticker = AlertTicker()
        video_col.addWidget(self._alert_ticker)

        body.addLayout(video_col, 3)

        # ---- Right: zone panel + stats ----
        right_col = QVBoxLayout()
        right_col.setSpacing(10)

        zones_cfg   = self._config.get_zones()
        threshold   = self._config.get_alert_threshold()
        self._zone_panel = ZonePanel(
            zones_cfg,
            warning_pct  = max(1, threshold // 2),
            critical_pct = threshold,
        )
        right_col.addWidget(self._zone_panel)

        self._stats_card = CycleSummaryCard()
        right_col.addWidget(self._stats_card)

        right_col.addStretch()

        # Control buttons
        self._btn_stop = QPushButton("⏹  Stop & View Summary")
        self._btn_stop.setStyleSheet(self._btn_style("#b71c1c"))
        self._btn_stop.setFixedHeight(42)
        self._btn_stop.clicked.connect(self._stop_session)
        right_col.addWidget(self._btn_stop)

        body.addLayout(right_col, 1)
        root.addLayout(body)

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------

    def start_session(self, golden_ref: GoldenReference | None = None) -> None:
        """Start monitoring. Call from main window after navigation."""
        if golden_ref:
            self._golden_ref = golden_ref

        self._session_id = self._db.start_session(
            station_id    = self._config.active_station,
            operator_name = None,
        )
        self._lbl_session.setText(f"Session #{self._session_id}")

        # Reset aggregate
        self._total = self._pass = self._fail = self._seq_err = 0
        self._cycle_times = []
        self._all_cycle_records = []
        self._cycle_number = 0

        # Start vision thread
        self._vision_thread = VisionThread(self._config)
        self._vision_thread.set_mode(VisionThread.MODE_MONITOR)
        if self._golden_ref:
            self._vision_thread.set_golden_reference(self._golden_ref)

        self._vision_thread.frame_ready.connect(self._video.set_frame)
        self._vision_thread.cycle_complete.connect(self._on_cycle_complete)
        self._vision_thread.state_changed.connect(self._on_state_changed)
        self._vision_thread.alert_fired.connect(self._on_alert)
        self._vision_thread.stats_updated.connect(self._on_stats_updated)
        self._vision_thread.error_occurred.connect(self._on_error)
        self._vision_thread.start()

    def _stop_session(self) -> None:
        if self._vision_thread:
            self._vision_thread.stop()
            self._vision_thread = None

        if self._session_id:
            self._db.end_session(self._session_id)
            self.session_stopped.emit(self._session_id)

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_cycle_complete(self, record: dict) -> None:
        self._cycle_number = record.get("cycle_number", self._cycle_number + 1)
        total_time = record.get("total_time", 0.0)
        status     = record.get("status", "pass")
        self._cycle_times.append(total_time)
        self._all_cycle_records.append(record)

        # Persist to DB
        gc = self._config.get_golden_cycle()
        std_times_raw = gc.get("standard_times", {})
        std_total = sum(float(v) for v in std_times_raw.values()) if std_times_raw else 0.0
        dev_pct = ((total_time - std_total) / std_total * 100) if std_total > 0 else None

        if self._session_id:
            cid = self._db.start_cycle(self._session_id, self._cycle_number)
            self._db.complete_cycle(
                cycle_id        = cid,
                status          = status,
                cycle_time_sec  = total_time,
                standard_time_sec = std_total or None,
                deviation_pct   = dev_pct,
                zone_times      = record.get("zone_times"),
                sequence_errors = record.get("sequence_errors"),
                dtw_score       = record.get("dtw_score"),
            )

        # Update aggregate
        self._total += 1
        if status == "pass":
            self._pass += 1
        elif status == "sequence_error":
            self._seq_err += 1
        else:
            self._fail += 1

        avg = sum(self._cycle_times) / len(self._cycle_times) if self._cycle_times else 0.0
        self._stats_card.update_stats(
            self._total, self._pass, self._fail, self._seq_err, avg
        )

    def _on_state_changed(self, state_name: str) -> None:
        self._lbl_state.setText(state_name)
        colors = {
            "IDLE":          "#607d8b",
            "ZONE_1_ACTIVE": "#1565c0",
            "TRANSIT_1_2":   "#37474f",
            "ZONE_2_ACTIVE": "#00838f",
            "TRANSIT_2_3":   "#37474f",
            "ZONE_3_ACTIVE": "#558b2f",
            "COMPLETED":     "#00c853",
        }
        self._lbl_state.setStyleSheet(
            f"color: {colors.get(state_name, '#eceff1')}; "
            "font-family: Consolas; font-size: 9px;"
        )

    def _on_alert(self, alert: dict) -> None:
        msg   = alert.get("message", "")
        level = alert.get("level", "WARNING")
        self._alert_ticker.show_alert(msg, level)

    def _on_stats_updated(self, stats: dict) -> None:
        self._zone_panel.update_stats(stats)
        prog = int(stats.get("cycle_progress", 0))
        self._progress_bar.setValue(prog)
        n = stats.get("cycle_number", 0)
        self._lbl_cycle_num.setText(f"Cycle #{n}")

    def _on_error(self, msg: str) -> None:
        self._alert_ticker.show_alert(f"ERROR: {msg}", "CRITICAL")

    # ------------------------------------------------------------------
    # Data access for summary screen
    # ------------------------------------------------------------------

    def get_all_cycle_records(self) -> list[dict]:
        return list(self._all_cycle_records)

    # ------------------------------------------------------------------
    # Style
    # ------------------------------------------------------------------

    @staticmethod
    def _btn_style(accent="#263238") -> str:
        return f"""
            QPushButton {{
                background-color: {accent};
                color: #eceff1;
                border: 1px solid #455a64;
                border-radius: 4px;
                padding: 6px;
                font-family: Consolas;
                font-size: 10px;
            }}
            QPushButton:hover {{ border-color: #00bcd4; }}
        """
