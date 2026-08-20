"""
gui/golden_cycle_screen.py
Golden Cycle Recording Screen.

Workflow
--------
1. Operator performs 3–5 correct cycles while system records trajectories
2. Progress indicator shows cycle count and per-cycle stats
3. After minimum cycles reached, "Compute Standard" button becomes active
4. GoldenCycleProcessor aggregates all recorded cycles → median standard times
5. Standard is saved to config JSON + SQLite, then screen emits golden_ready
"""

from __future__ import annotations

import statistics

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QFrame, QGroupBox, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QVBoxLayout, QWidget,
)

from core.dtw_comparator import GoldenCycleProcessor, GoldenReference
from core.cycle_tracker import CycleRecord, CycleTracker
from core.point_trigger_detector import TriggerPoint
from data.config_handler import ConfigHandler
from data.database import DatabaseManager
from gui.widgets.video_widget import VideoWidget
from gui.monitor_screen import _trigger_points_from_config
from vision.point_tracker_thread import PointTrackerThread


class CycleResultRow(QFrame):
    """One row showing a completed recording cycle's stats."""

    def __init__(self, record: dict, zone_names: dict[int, str], parent=None) -> None:
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #152028;
                border: 1px solid #263238;
                border-radius: 4px;
            }
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        n     = record.get("cycle_number", "?")
        total = record.get("total_time", 0.0)
        status = record.get("status", "—")
        color  = "#00c853" if status == "pass" else "#d50000"

        layout.addWidget(self._mk_label(f"Cycle #{n}", bold=True, color="#eceff1"))
        layout.addWidget(self._mk_label(f"{total:.2f}s", color="#00bcd4"))

        zone_times = record.get("zone_times", {})
        for zid_str, t in zone_times.items():
            zid  = int(zid_str)
            name = zone_names.get(zid, f"Z{zid}")
            layout.addWidget(self._mk_label(f"{name}: {t:.2f}s", color="#b0bec5"))

        layout.addStretch()
        layout.addWidget(self._mk_label(status.upper(), bold=True, color=color))

    @staticmethod
    def _mk_label(text: str, bold=False, color="#b0bec5") -> QLabel:
        lbl = QLabel(text)
        font = QFont("Consolas", 8, QFont.Bold if bold else QFont.Normal)
        lbl.setFont(font)
        lbl.setStyleSheet(f"color: {color};")
        lbl.setContentsMargins(0, 0, 8, 0)
        return lbl


class GoldenCycleScreen(QWidget):
    """Screen for recording golden reference cycles."""

    golden_ready   = pyqtSignal(object)   # GoldenReference
    recording_done = pyqtSignal()

    def __init__(
        self,
        config: ConfigHandler,
        db: DatabaseManager,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._db     = db
        self._tracker_thread: PointTrackerThread | None = None
        self._cycle_tracker:  CycleTracker | None = None
        self._recorded_cycles: list[dict] = []
        self._golden_ref: GoldenReference | None = None

        # Track previous state per point to detect ACTIVE → COOLDOWN transition
        self._prev_point_states: dict[int, str] = {}

        self._min_cycles = config.get_min_golden_cycles()
        self._max_cycles = config.get_max_golden_cycles()

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # Header
        hdr = QHBoxLayout()
        title = QLabel("🎯  GOLDEN CYCLE RECORDING")
        title.setFont(QFont("Consolas", 14, QFont.Bold))
        title.setStyleSheet("color: #00bcd4;")
        hdr.addWidget(title)
        hdr.addStretch()

        self._lbl_cycle_count = QLabel(f"0 / {self._min_cycles}–{self._max_cycles} cycles")
        self._lbl_cycle_count.setFont(QFont("Consolas", 10))
        self._lbl_cycle_count.setStyleSheet("color: #ffab00;")
        hdr.addWidget(self._lbl_cycle_count)
        root.addLayout(hdr)

        # Instructions banner
        instr = QLabel(
            f"Perform {self._min_cycles}–{self._max_cycles} correct production cycles. "
            "The system will calculate the standard (median) time for each zone."
        )
        instr.setFont(QFont("Consolas", 9))
        instr.setStyleSheet(
            "color: #b0bec5; background-color: #1a2f3a; "
            "border-left: 3px solid #00bcd4; padding: 6px;"
        )
        instr.setWordWrap(True)
        root.addWidget(instr)

        # Body
        body = QHBoxLayout()

        # ---- Left: video + state ----
        left = QVBoxLayout()
        self._video = VideoWidget()
        self._video.setMinimumSize(520, 292)
        left.addWidget(self._video)

        # State HUD
        state_row = QHBoxLayout()
        self._lbl_state = QLabel("STATE: IDLE")
        self._lbl_state.setFont(QFont("Consolas", 11, QFont.Bold))
        self._lbl_state.setStyleSheet("color: #607d8b;")
        self._lbl_elapsed = QLabel("0.0s")
        self._lbl_elapsed.setFont(QFont("Consolas", 16, QFont.Bold))
        self._lbl_elapsed.setStyleSheet("color: #eceff1;")
        state_row.addWidget(self._lbl_state)
        state_row.addStretch()
        state_row.addWidget(self._lbl_elapsed)
        left.addLayout(state_row)

        # Alert banner
        self._lbl_alert = QLabel("")
        self._lbl_alert.setFont(QFont("Consolas", 10, QFont.Bold))
        self._lbl_alert.setStyleSheet(
            "color: #d50000; background-color: #3e0000; "
            "border: 1px solid #d50000; border-radius: 3px; padding: 4px;"
        )
        self._lbl_alert.setWordWrap(True)
        self._lbl_alert.hide()
        left.addWidget(self._lbl_alert)

        body.addLayout(left, 3)

        # ---- Right: controls + cycle log ----
        right = QVBoxLayout()
        right.setSpacing(10)

        # Controls
        self._btn_start = QPushButton("▶  Start Recording")
        self._btn_start.setStyleSheet(self._btn_style("#1b5e20"))
        self._btn_start.setFixedHeight(40)
        self._btn_start.clicked.connect(self._start_recording)
        right.addWidget(self._btn_start)

        self._btn_stop = QPushButton("⏹  Stop Recording")
        self._btn_stop.setStyleSheet(self._btn_style("#b71c1c"))
        self._btn_stop.setFixedHeight(40)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_recording)
        right.addWidget(self._btn_stop)

        self._btn_compute = QPushButton("⚙  Compute Golden Standard")
        self._btn_compute.setStyleSheet(self._btn_style("#0d47a1"))
        self._btn_compute.setFixedHeight(40)
        self._btn_compute.setEnabled(False)
        self._btn_compute.clicked.connect(self._compute_standard)
        right.addWidget(self._btn_compute)

        # Progress
        right.addWidget(QLabel("Recording progress:").setFont(QFont("Consolas", 8)) or QLabel("Recording progress:"))
        self._progress = QProgressBar()
        self._progress.setRange(0, self._min_cycles)
        self._progress.setValue(0)
        self._progress.setFixedHeight(8)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet("""
            QProgressBar { background:#152028; border-radius:4px; border:none; }
            QProgressBar::chunk { background:#00c853; border-radius:4px; }
        """)
        right.addWidget(self._progress)

        # Cycle log
        log_label = QLabel("Recorded Cycles:")
        log_label.setFont(QFont("Consolas", 8))
        log_label.setStyleSheet("color: #607d8b;")
        right.addWidget(log_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background:#0d1b2a; border:none;")
        self._log_container = QWidget()
        self._log_layout    = QVBoxLayout(self._log_container)
        self._log_layout.setContentsMargins(0, 0, 0, 0)
        self._log_layout.setSpacing(4)
        self._log_layout.addStretch()
        scroll.setWidget(self._log_container)
        right.addWidget(scroll)

        # Standard preview
        self._std_box = QGroupBox("Computed Standard Times")
        self._std_box.setStyleSheet(self._group_style())
        self._std_layout = QVBoxLayout(self._std_box)
        right.addWidget(self._std_box)

        right.addStretch()

        self._btn_use = QPushButton("✔  Use This Standard & Continue →")
        self._btn_use.setStyleSheet(self._btn_style("#1565c0"))
        self._btn_use.setFixedHeight(42)
        self._btn_use.setEnabled(False)
        self._btn_use.clicked.connect(self._use_standard)
        right.addWidget(self._btn_use)

        body.addLayout(right, 2)
        root.addLayout(body)

    # ------------------------------------------------------------------
    # Recording control
    # ------------------------------------------------------------------

    def _start_recording(self) -> None:
        self._recorded_cycles = []
        self._prev_point_states = {}   # reset prev-state tracker

        # Build trigger points from config (polygon centroid adapter)
        trigger_points = _trigger_points_from_config(self._config)
        zone_ids = [tp.point_id for tp in trigger_points]

        self._cycle_tracker = CycleTracker(
            zone_ids          = zone_ids,
            on_cycle_complete = self._cb_cycle_complete,
            on_alert          = self._cb_alert,
            on_state_change   = self._cb_state_change,
        )

        self._tracker_thread = PointTrackerThread(self._config, trigger_points)
        self._tracker_thread.frame_ready.connect(self._video.set_frame)
        self._tracker_thread.error_occurred.connect(self._on_error)
        self._tracker_thread.point_triggered.connect(self._on_point_triggered)
        self._tracker_thread.point_state_changed.connect(self._on_point_state_changed)
        self._tracker_thread.hand_position_updated.connect(self._on_hand_position)
        self._tracker_thread.start()

        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._btn_compute.setEnabled(False)

    def _stop_recording(self) -> None:
        if self._tracker_thread:
            self._tracker_thread.stop()
            self._tracker_thread = None
        self._cycle_tracker = None
        self._prev_point_states = {}   # clear stale states

        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._lbl_state.setText("STATE: STOPPED")

        if len(self._recorded_cycles) >= self._min_cycles:
            self._btn_compute.setEnabled(True)

    def _compute_standard(self) -> None:
        if not self._recorded_cycles:
            return

        # Convert dicts back to lightweight objects GoldenCycleProcessor can use
        from types import SimpleNamespace

        fake_records = []
        for r in self._recorded_cycles:
            from core.cycle_tracker import CycleRecord, ZoneTiming
            rec = CycleRecord(
                cycle_number    = r["cycle_number"],
                start_time      = r.get("start_time", 0),
                end_time        = r.get("end_time", 0),
                trajectory      = r.get("trajectory", []),
                sequence_errors = r.get("sequence_errors", []),
                status          = r.get("status", "pass"),
            )
            for zid_str, t in r.get("zone_times", {}).items():
                zid = int(zid_str)
                zt  = ZoneTiming(zone_id=zid)
                zt.exit_time  = t
                zt.enter_time = 0
                rec.zone_timings[zid] = zt
            fake_records.append(rec)

        zone_ids = [z["id"] for z in self._config.get_zones()]
        self._golden_ref = GoldenCycleProcessor.process(fake_records, zone_ids)

        # Show standard times
        for i in reversed(range(self._std_layout.count())):
            w = self._std_layout.itemAt(i).widget()
            if w:
                w.deleteLater()

        zones = {z["id"]: z.get("name", f"Zone {z['id']}") for z in self._config.get_zones()}
        total = 0.0
        for zid, t in sorted(self._golden_ref.standard_times.items()):
            name = zones.get(zid, f"Zone {zid}")
            lbl = QLabel(f"  {name}: {t:.2f}s")
            lbl.setFont(QFont("Consolas", 9))
            lbl.setStyleSheet("color: #00c853;")
            self._std_layout.addWidget(lbl)
            total += t

        total_lbl = QLabel(f"  Total: {total:.2f}s")
        total_lbl.setFont(QFont("Consolas", 10, QFont.Bold))
        total_lbl.setStyleSheet("color: #00bcd4;")
        self._std_layout.addWidget(total_lbl)

        self._btn_use.setEnabled(True)

    def _use_standard(self) -> None:
        if not self._golden_ref:
            return

        # Save to config JSON
        zone_std_str = {str(k): v for k, v in self._golden_ref.standard_times.items()}
        self._config.set_golden_cycle(
            standard_times    = zone_std_str,
            trajectory_points = self._golden_ref.raw_trajectory,
            recorded_cycles   = self._recorded_cycles,
        )
        self._config.save()

        # Save to SQLite
        self._db.save_golden_cycle(
            station_id          = self._config.active_station,
            num_source_cycles   = len(self._recorded_cycles),
            standard_total_sec  = sum(self._golden_ref.standard_times.values()),
            zone_standard_times = {str(k): v for k, v in self._golden_ref.standard_times.items()},
            trajectory_points   = self._golden_ref.raw_trajectory,
            raw_cycle_times     = [r["total_time"] for r in self._recorded_cycles],
        )

        self.golden_ready.emit(self._golden_ref)
        self.recording_done.emit()

    # ------------------------------------------------------------------
    # PointTrackerThread signal handlers
    # ------------------------------------------------------------------

    def _on_point_triggered(self, point_id: int, timestamp: float,
                            hand_x: float, hand_y: float) -> None:
        if self._cycle_tracker:
            self._cycle_tracker.on_zone_event(point_id, "enter")
            self._cycle_tracker.tick(hand_x, hand_y)

    def _on_point_state_changed(self, point_id: int, state_name: str) -> None:
        """Relay zone exit to CycleTracker on ACTIVE → COOLDOWN transition."""
        prev = self._prev_point_states.get(point_id, "")
        self._prev_point_states[point_id] = state_name

        if prev == "ACTIVE" and state_name == "COOLDOWN":
            if self._cycle_tracker:
                self._cycle_tracker.on_zone_event(point_id, "exit")

        self._lbl_state.setText(f"STATE: P{point_id}:{state_name}")

    def _on_hand_position(self, hand_x: float, hand_y: float) -> None:
        if self._cycle_tracker:
            self._cycle_tracker.tick(hand_x, hand_y)

    def _on_error(self, msg: str) -> None:
        self._lbl_alert.setText(f"⚠ ERROR: {msg}")
        self._lbl_alert.show()
        QTimer.singleShot(5000, self._lbl_alert.hide)

    # ------------------------------------------------------------------
    # CycleTracker callbacks
    # ------------------------------------------------------------------

    def _cb_cycle_complete(self, record: CycleRecord) -> None:
        record_dict = {
            "cycle_number":    record.cycle_number,
            "total_time":      record.total_time,
            "status":          record.status,
            "zone_times":      record.zone_times_dict(),
            "sequence_errors": record.sequence_errors,
            "trajectory":      record.trajectory,
            "start_time":      record.start_time,
            "end_time":        record.end_time,
        }
        self._on_cycle_complete(record_dict)

    def _cb_alert(self, zone_id: int, message: str) -> None:
        self._on_alert({"message": message, "level": "WARNING"})

    def _cb_state_change(self, state) -> None:
        state_name = state.name if hasattr(state, "name") else str(state)
        elapsed_hint = 0.0
        self._on_stats({"cycle_elapsed": elapsed_hint, "state": state_name})

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_cycle_complete(self, record: dict) -> None:
        self._recorded_cycles.append(record)

        # Add row to log
        zones = {z["id"]: z.get("name", f"Zone {z['id']}") for z in self._config.get_zones()}
        row = CycleResultRow(record, zones)
        # Insert before stretch
        self._log_layout.insertWidget(self._log_layout.count() - 1, row)

        # Update counter
        n = len(self._recorded_cycles)
        self._lbl_cycle_count.setText(
            f"{n} / {self._min_cycles}–{self._max_cycles} cycles"
        )
        self._progress.setValue(min(n, self._min_cycles))

        if n >= self._min_cycles:
            self._lbl_cycle_count.setStyleSheet("color: #00c853;")
            self._btn_compute.setEnabled(True)

        if n >= self._max_cycles:
            self._stop_recording()

    def _on_alert(self, alert: dict) -> None:
        msg = alert.get("message", "")
        self._lbl_alert.setText(f"⚠ {msg}")
        self._lbl_alert.show()
        QTimer.singleShot(4000, self._lbl_alert.hide)

    def _on_stats(self, stats: dict) -> None:
        elapsed = stats.get("cycle_elapsed", 0.0)
        self._lbl_elapsed.setText(f"{elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Style helpers
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
            QPushButton:disabled {{ color: #455a64; border-color: #263238; }}
        """

    @staticmethod
    def _group_style() -> str:
        return """
            QGroupBox {
                color: #607d8b;
                border: 1px solid #263238;
                border-radius: 4px;
                margin-top: 8px;
                font-family: Consolas;
                font-size: 9px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
        """
