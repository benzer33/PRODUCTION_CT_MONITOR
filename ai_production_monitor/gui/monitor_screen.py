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

from collections import deque

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPointF
from PyQt5.QtGui import QFont, QPainter, QPen, QColor, QBrush
from PyQt5.QtWidgets import (
    QCheckBox, QFrame, QGroupBox, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QScrollArea,
    QVBoxLayout, QWidget,
)

from core.ghost_interpolator import (
    interpolate_ghost_position,
    scale_to_widget,
)

from core.cycle_tracker import CycleTracker, CycleRecord
from core.dtw_comparator import GoldenReference
from core.point_trigger_detector import TriggerPoint
from data.config_handler import ConfigHandler
from data.database import DatabaseManager
from gui.theme import (
    FONT_FAMILY, FONT_SIZE_BODY, FONT_SIZE_CAPTION, FONT_SIZE_LABEL,
    FONT_SIZE_METRIC, FONT_SIZE_SECTION, FONT_SIZE_STATUS, FONT_SIZE_TITLE,
    make_font,
    btn_stylesheet,
    CLR_ACCENT, CLR_MUTED,
)
from gui.widgets.video_widget import VideoWidget
from gui.widgets.zone_widget import ZonePanel
from vision.point_tracker_thread import PointTrackerThread


# ---------------------------------------------------------------------------
# Alert ticker widget
# ---------------------------------------------------------------------------

class AlertTicker(QLabel):
    """Scrolling alert banner shown at the bottom of the video area."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFont(make_font(FONT_SIZE_BODY, bold=True))
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
                f"font-family: {FONT_FAMILY}; font-size: {FONT_SIZE_BODY}px;"
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
            lbl_k.setFont(make_font(FONT_SIZE_LABEL))
            lbl_k.setStyleSheet("color: #607d8b;")
            lbl_v = QLabel("—")
            lbl_v.setFont(make_font(FONT_SIZE_BODY, bold=True))
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
            f"color: #00c853; font-family: {FONT_FAMILY}; font-size: {FONT_SIZE_BODY}px; font-weight: bold;"
        )
        self._labels["fail"].setText(str(failed))
        self._labels["fail"].setStyleSheet(
            f"color: {'#d50000' if failed else '#eceff1'}; "
            f"font-family: {FONT_FAMILY}; font-size: {FONT_SIZE_BODY}px; font-weight: bold;"
        )
        self._labels["seq_err"].setText(str(seq_errors))
        self._labels["avg"].setText(f"{avg_time:.2f}s" if avg_time else "—")


# ---------------------------------------------------------------------------
# Ghost overlay widget
# ---------------------------------------------------------------------------

# How many seconds of trail to keep (shown as fading dots behind the ghost)
_TRAIL_SECS: float = 0.8
# Max number of trail samples retained (sampled at ~30 fps → ~24 pts)
_TRAIL_MAX: int = 60
# Radius of the main ghost circle (widget pixels)
_GHOST_RADIUS: int = 10


class GhostOverlayWidget(QWidget):
    """Transparent overlay that sits on top of VideoWidget.

    Call `set_ghost(state, wx, wy, delta_sec)` to push a new position, then
    `update()` is called automatically.  When hidden (show_ghost=False) the
    caller must not call set_ghost() at all — this widget just won't paint.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        # Make the widget fully transparent to mouse events and background
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")

        # Current ghost state (widget-space pixels)
        self._wx: float = 0.0
        self._wy: float = 0.0
        self._delta_sec: float | None = None   # None = unknown
        self._clamped: bool = False

        # Trail: deque of (wx, wy) — newest at right
        self._trail: deque[tuple[float, float]] = deque(maxlen=_TRAIL_MAX)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_ghost(
        self,
        wx: float,
        wy: float,
        delta_sec: float | None,
        clamped: bool,
    ) -> None:
        """Push a new ghost position and trigger repaint."""
        self._trail.append((self._wx, self._wy))
        self._wx = wx
        self._wy = wy
        self._delta_sec = delta_sec
        self._clamped = clamped
        self.update()

    def reset(self) -> None:
        """Clear trail and ghost position (call when session stops)."""
        self._trail.clear()
        self._delta_sec = None
        self._clamped = False
        self.update()

    # ------------------------------------------------------------------
    # Qt paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        if not self._trail and self._delta_sec is None:
            return  # nothing to draw yet

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # --- Trail ---
        n = len(self._trail)
        for i, (tx, ty) in enumerate(self._trail):
            # alpha fades from very faint (oldest) to moderate (newest)
            alpha = int(20 + 120 * (i / max(n - 1, 1)))
            color = QColor(0, 200, 83, alpha)   # green
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(color))
            radius = max(2, int(_GHOST_RADIUS * 0.5 * (i / max(n - 1, 1))))
            painter.drawEllipse(QPointF(tx, ty), radius, radius)

        # --- Main ghost circle (dashed border, semi-transparent fill) ---
        fill_color = QColor(0, 200, 83, 80)     # semi-transparent green
        border_color = QColor(0, 230, 118, 220) # bright green border

        pen = QPen(border_color, 2, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(fill_color))
        painter.drawEllipse(QPointF(self._wx, self._wy), _GHOST_RADIUS, _GHOST_RADIUS)

        # --- Delta label ---
        if self._delta_sec is not None:
            sign = "+" if self._delta_sec >= 0 else "-"
            label = f"{sign}{abs(self._delta_sec):.1f}s"
            # Red when behind (positive delta = slow), green when ahead
            text_color = QColor("#ff1744") if self._delta_sec > 0 else QColor("#00e676")
            painter.setPen(text_color)
            font = QFont("Consolas", 9, QFont.Bold)
            painter.setFont(font)
            # Position label just to the right and slightly above the circle
            painter.drawText(
                int(self._wx + _GHOST_RADIUS + 4),
                int(self._wy - 4),
                label,
            )

        painter.end()


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
        self._tracker_thread: PointTrackerThread | None = None
        self._cycle_tracker:  CycleTracker | None = None
        self._session_id: int | None = None
        self._cycle_id:   int | None = None
        self._cycle_number = 0

        # Track previous state per point to detect ACTIVE → COOLDOWN transition
        self._prev_point_states: dict[int, str] = {}

        # Aggregate stats
        self._total = self._pass = self._fail = self._seq_err = 0
        self._cycle_times: list[float] = []
        self._all_cycle_records: list[dict] = []

        # Ghost overlay state
        self._last_hand_x: float = 0.0
        self._last_hand_y: float = 0.0
        self._last_frame_w: int = 0
        self._last_frame_h: int = 0

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
        self._lbl_title.setFont(make_font(FONT_SIZE_TITLE, bold=True))
        self._lbl_title.setStyleSheet("color: #00bcd4;")
        hdr.addWidget(self._lbl_title)
        hdr.addStretch()

        self._lbl_session = QLabel("Session: —")
        self._lbl_session.setFont(make_font(FONT_SIZE_LABEL))
        self._lbl_session.setStyleSheet("color: #607d8b;")
        hdr.addWidget(self._lbl_session)
        root.addLayout(hdr)

        # Body
        body = QHBoxLayout()
        body.setSpacing(12)

        # ---- Left: video area ----
        video_col = QVBoxLayout()

        # Video container (positions ghost overlay relative to VideoWidget)
        video_container = QWidget()
        video_container.setMinimumSize(640, 360)
        video_container_layout = QVBoxLayout(video_container)
        video_container_layout.setContentsMargins(0, 0, 0, 0)

        self._video = VideoWidget(video_container)
        video_container_layout.addWidget(self._video)

        # Ghost overlay — transparent child widget, resized in resizeEvent
        self._ghost_overlay = GhostOverlayWidget(self._video)
        self._ghost_overlay.resize(self._video.size())
        # Visibility follows the persisted config setting
        if self._config.get_show_ghost():
            self._ghost_overlay.show()
        else:
            self._ghost_overlay.hide()

        # Toggle: Show Golden Master
        ghost_row = QHBoxLayout()
        self._chk_ghost = QCheckBox("แสดงมาสเตอร์เขียว  (Show Golden Master)")
        self._chk_ghost.setFont(make_font(FONT_SIZE_LABEL))
        self._chk_ghost.setStyleSheet("color: #80cbc4;")
        self._chk_ghost.setChecked(self._config.get_show_ghost())
        self._chk_ghost.toggled.connect(self._on_ghost_toggled)
        ghost_row.addWidget(self._chk_ghost)
        ghost_row.addStretch()
        video_col.addLayout(ghost_row)

        video_col.addWidget(video_container)

        # Progress bar (cycle %)
        prog_row = QHBoxLayout()
        self._lbl_cycle_num = QLabel("Cycle #0")
        self._lbl_cycle_num.setFont(make_font(FONT_SIZE_BODY, bold=True))
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
        self._lbl_state.setFont(make_font(FONT_SIZE_STATUS, bold=True))
        self._lbl_state.setStyleSheet(f"color: {CLR_MUTED};")
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
        self._prev_point_states = {}   # reset prev-state tracker

        # Build trigger points from config (polygon centroid adapter)
        trigger_points = _trigger_points_from_config(self._config)

        # Wire CycleTracker — runs in GUI thread, callbacks are invoked from
        # the tracker thread via point_triggered signal (queued connection)
        gc = self._config.get_golden_cycle()
        std_times_raw = gc.get("standard_times", {})
        std_times = {int(k): float(v) for k, v in std_times_raw.items()}
        zone_ids  = [tp.point_id for tp in trigger_points]

        self._cycle_tracker = CycleTracker(
            zone_ids          = zone_ids,
            standard_times    = std_times,
            on_cycle_complete = self._cb_cycle_complete,
            on_alert          = self._cb_alert,
            on_sequence_error = self._cb_sequence_error,
            on_state_change   = self._cb_state_change,
        )

        # Start PointTrackerThread
        self._tracker_thread = PointTrackerThread(self._config, trigger_points)
        self._tracker_thread.frame_ready.connect(self._video.set_frame)
        self._tracker_thread.frame_ready.connect(self._on_frame_for_ghost)
        self._tracker_thread.error_occurred.connect(self._on_error)
        self._tracker_thread.point_triggered.connect(self._on_point_triggered)
        self._tracker_thread.point_state_changed.connect(self._on_point_state_changed)
        self._tracker_thread.hand_position_updated.connect(self._on_hand_position)
        self._tracker_thread.start()

    def _stop_session(self) -> None:
        if self._tracker_thread:
            self._tracker_thread.stop()
            self._tracker_thread = None
        self._cycle_tracker = None
        self._prev_point_states = {}   # clear stale states
        self._ghost_overlay.reset()

        if self._session_id:
            self._db.end_session(self._session_id)
            self.session_stopped.emit(self._session_id)

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Ghost overlay helpers
    # ------------------------------------------------------------------

    def _on_ghost_toggled(self, checked: bool) -> None:
        """Persist toggle and show/hide overlay immediately."""
        self._config.set_show_ghost(checked)
        if not checked:
            self._ghost_overlay.reset()
            self._ghost_overlay.hide()
        else:
            self._ghost_overlay.show()

    def _on_frame_for_ghost(self, frame) -> None:
        """Called on every new frame — resize overlay to match VideoWidget
        and compute ghost position if toggle is on."""
        # Keep overlay filling the video widget
        if self._ghost_overlay.size() != self._video.size():
            self._ghost_overlay.resize(self._video.size())

        if not self._chk_ghost.isChecked():
            return

        # Cache frame dimensions for coordinate scaling
        if frame is not None and frame.size > 0:
            h, w = frame.shape[:2]
            self._last_frame_w = w
            self._last_frame_h = h

        self._update_ghost()

    def _update_ghost(self) -> None:
        """Compute ghost position from golden trajectory and elapsed time,
        then push it to GhostOverlayWidget."""
        if not self._golden_ref:
            return
        if not self._cycle_tracker:
            return

        elapsed = self._cycle_tracker.cycle_elapsed
        if elapsed is None or elapsed < 0:
            return

        golden_total = self._golden_ref.total_standard_time
        traj = self._golden_ref.raw_trajectory

        ghost = interpolate_ghost_position(traj, elapsed, golden_total)
        if ghost is None:
            return

        # Scale frame-pixel → widget-pixel using same letterbox transform
        fw = self._last_frame_w or self._video.width()
        fh = self._last_frame_h or self._video.height()
        wx, wy = scale_to_widget(
            ghost,
            frame_w=fw, frame_h=fh,
            widget_w=self._video.width(), widget_h=self._video.height(),
        )

        # Delta: positive = operator is slower than golden (label goes red)
        # We compute it as "elapsed - (ghost.t_norm * golden_total)"
        # i.e. how many seconds behind/ahead the operator is in wall-clock time
        delta_sec: float | None = None
        if golden_total > 0:
            ghost_elapsed = ghost.t_norm * golden_total
            delta_sec = round(elapsed - ghost_elapsed, 1)

        self._ghost_overlay.set_ghost(wx, wy, delta_sec, ghost.clamped)

    # ------------------------------------------------------------------
    # Cycle complete callback
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

    # ------------------------------------------------------------------
    # PointTrackerThread signal handlers
    # ------------------------------------------------------------------

    def _on_point_triggered(self, point_id: int, timestamp: float,
                            hand_x: float, hand_y: float) -> None:
        """Relay point trigger → CycleTracker as zone 'enter' event."""
        if self._cycle_tracker:
            self._cycle_tracker.on_zone_event(point_id, "enter")
            self._cycle_tracker.tick(hand_x, hand_y)

    def _on_point_state_changed(self, point_id: int, state_name: str) -> None:
        """Update state HUD and relay zone exit to CycleTracker.

        The ACTIVE → COOLDOWN transition means the hand has left the point
        after a confirmed trigger — this is the semantic equivalent of a zone
        'exit' event that CycleTracker needs to advance the cycle state
        machine and eventually call _complete_cycle().
        """
        prev = self._prev_point_states.get(point_id, "")
        self._prev_point_states[point_id] = state_name

        # Detect hand-left-point: ACTIVE → COOLDOWN
        if prev == "ACTIVE" and state_name == "COOLDOWN":
            if self._cycle_tracker:
                self._cycle_tracker.on_zone_event(point_id, "exit")

        # HUD update
        self._lbl_state.setText(f"P{point_id}:{state_name}")
        armed_states = {"ARMED", "TRIGGERED_PENDING", "ACTIVE"}
        color = "#00c853" if state_name in armed_states else "#607d8b"
        self._lbl_state.setStyleSheet(
            f"color: {color}; font-family: {FONT_FAMILY}; font-size: {FONT_SIZE_STATUS}px; font-weight: bold;"
        )

    def _on_hand_position(self, hand_x: float, hand_y: float) -> None:
        """Forward hand position to CycleTracker for real-time stats."""
        self._last_hand_x = hand_x
        self._last_hand_y = hand_y
        if self._cycle_tracker:
            self._cycle_tracker.tick(hand_x, hand_y)

    # ------------------------------------------------------------------
    # CycleTracker callbacks (called from tracker thread — queued)
    # ------------------------------------------------------------------

    def _cb_cycle_complete(self, record: CycleRecord) -> None:
        """Convert CycleRecord → dict and call the existing handler."""
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
        self._on_alert({"message": message, "level": "WARNING", "zone_id": zone_id})

    def _cb_sequence_error(self, expected: int, actual: int) -> None:
        msg = f"Sequence error: expected Point {expected}, got Point {actual}"
        self._on_alert({"message": msg, "level": "CRITICAL"})

    def _cb_state_change(self, state) -> None:
        """Update progress bar from CycleTracker state."""
        from core.cycle_tracker import CycleState
        state_name = state.name if hasattr(state, "name") else str(state)
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
            f"font-family: {FONT_FAMILY}; font-size: {FONT_SIZE_STATUS}px; font-weight: bold;"
        )

    def _on_state_changed(self, state_name: str) -> None:
        """Legacy-compatible handler kept for internal use."""
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
            f"font-family: {FONT_FAMILY}; font-size: {FONT_SIZE_STATUS}px; font-weight: bold;"
        )

    def _on_alert(self, alert: dict) -> None:
        msg   = alert.get("message", "")
        level = alert.get("level", "WARNING")
        self._alert_ticker.show_alert(msg, level)

    def _on_stats_updated(self, stats: dict) -> None:
        self._zone_panel.update_stats(stats)
        prog = int(stats.get("cycle_progress", 0))
        self._progress_bar.setValue(prog)
        n = stats.get("cycle_number", self._cycle_number)
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
        return btn_stylesheet(accent)


# ---------------------------------------------------------------------------
# Module-level helper — polygon centroid adapter
# ---------------------------------------------------------------------------

def _trigger_points_from_config(config: ConfigHandler) -> list[TriggerPoint]:
    """Convert zone polygon config → TriggerPoint list.

    Each zone's centroid becomes (x, y) and half of the bounding-box
    short-axis becomes the radius.  Works for both old polygon configs and
    any future point-based zone that already has x/y/radius keys.
    """
    points: list[TriggerPoint] = []
    for zone in config.get_zones():
        zid  = zone.get("id", 0)
        name = zone.get("name", f"Point {zid}")

        # New-style: zone already has x, y, radius
        if "x" in zone and "y" in zone:
            points.append(TriggerPoint(
                point_id = zid,
                name     = name,
                x        = float(zone["x"]),
                y        = float(zone["y"]),
                radius   = float(zone.get("radius", 30)),
            ))
            continue

        # Old-style: derive centroid from polygon
        poly = zone.get("polygon", [])
        if not poly:
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        # radius = half of min(width, height) of bounding box
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        radius = max(20.0, min(w, h) / 2.0)
        points.append(TriggerPoint(
            point_id = zid,
            name     = name,
            x        = cx,
            y        = cy,
            radius   = radius,
        ))
    return points
