"""
vision/vision_thread.py
QThread that drives the camera + hand tracking + cycle logic pipeline.

Architecture
------------
VisionThread (QThread)
  │ runs in background thread
  ├── CameraHandler      — grabs raw frames
  ├── HandTracker        — MediaPipe hand landmark detection
  ├── ZoneDetector       — zone entry/exit events
  ├── CycleTracker       — state machine (calls AlertManager internally)
  └── DTWComparator      — ghost overlay sync + live scoring

Qt Signals emitted to GUI thread
---------------------------------
  frame_ready(np.ndarray)         — annotated BGR frame for display
  hand_position(float, float)     — wrist px coords
  zone_event(int, str)            — (zone_id, "enter"|"exit")
  cycle_complete(dict)            — serialised CycleRecord
  state_changed(str)              — CycleState name
  alert_fired(dict)               — serialised AlertEvent
  stats_updated(dict)             — per-zone live stats (for side panel)
  error_occurred(str)             — camera/tracking error message
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from core.alert_manager  import AlertEvent, AlertManager, ThresholdConfig
from core.cycle_tracker  import CycleRecord, CycleState, CycleTracker
from core.dtw_comparator import DTWComparator, GoldenReference
from core.sequence_checker import SequenceChecker, SequenceViolation
from core.zone_detector  import ZoneDetector
from data.config_handler import ConfigHandler
from vision.camera_handler import CameraHandler
from vision.hand_tracker   import HandTracker


class VisionThread(QThread):
    """
    Background QThread that owns all vision-processing objects and emits
    results to the GUI via Qt signals.

    Usage
    -----
        vt = VisionThread(config)
        vt.frame_ready.connect(my_video_widget.set_frame)
        vt.start()
        ...
        vt.stop()
    """

    # ---- Signals ----
    frame_ready    = pyqtSignal(object)    # np.ndarray BGR
    hand_position  = pyqtSignal(float, float)
    zone_event     = pyqtSignal(int, str)
    cycle_complete = pyqtSignal(dict)
    state_changed  = pyqtSignal(str)
    alert_fired    = pyqtSignal(dict)
    stats_updated  = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    # ---- Operating modes ----
    MODE_IDLE    = "idle"
    MODE_PREVIEW = "preview"    # camera open, no tracking
    MODE_GOLDEN  = "golden"     # recording golden cycles
    MODE_MONITOR = "monitor"    # live production monitoring

    def __init__(self, config: ConfigHandler, parent=None) -> None:
        super().__init__(parent)
        self._config   = config
        self._mode     = self.MODE_IDLE
        self._running  = False

        # Sub-systems (created lazily in run())
        self._camera:       CameraHandler     | None = None
        self._tracker:      HandTracker       | None = None
        self._zones:        ZoneDetector      | None = None
        self._cycles:       CycleTracker      | None = None
        self._alerts:       AlertManager      | None = None
        self._dtw:          DTWComparator     | None = None
        self._seq_checker:  SequenceChecker   | None = None

        # DatabaseManager — inject from outside (optional, enables DB logging)
        self._db = None   # set via set_database(db)

        # Current DB cycle_id (set by _on_cycle_start / vision thread)
        self._db_cycle_id: int | None = None
        self._db_session_id: int | None = None

        # Draw-overlay toggles
        self.show_landmarks   = True
        self.show_zones       = True
        self.show_ghost       = True
        self.show_zone_labels = True

        # Stats throttle — emit stats at most 10× per second
        self._last_stats_emit = 0.0

    # ------------------------------------------------------------------
    # Public control API (call from GUI thread)
    # ------------------------------------------------------------------

    def set_database(self, db, session_id: int | None = None) -> None:
        """
        Inject DatabaseManager สำหรับ alert + violation logging
        เรียกก่อน start() จาก GUI thread
        """
        self._db = db
        self._db_session_id = session_id

    def set_mode(self, mode: str) -> None:
        self._mode = mode

    def set_golden_reference(self, ref: GoldenReference) -> None:
        self._golden_ref = ref
        if self._dtw:
            self._dtw.set_golden(ref)

    def stop(self) -> None:
        self._running = False
        self.wait(3000)

    # ------------------------------------------------------------------
    # QThread.run — main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._running = True

        station_id = self._config.active_station
        cam_cfg    = self._config.get_camera_config(station_id)
        zones_cfg  = self._config.get_zones(station_id)
        threshold  = self._config.get_alert_threshold(station_id)

        # ---- Initialise camera ----
        self._camera = CameraHandler(
            camera_type  = cam_cfg.get("type", "webcam"),
            device_index = cam_cfg.get("device_index", 0),
            url          = cam_cfg.get("url", ""),
            width        = cam_cfg.get("width", 1280),
            height       = cam_cfg.get("height", 720),
            fps          = cam_cfg.get("fps", 30),
        )
        if not self._camera.open():
            self.error_occurred.emit("Failed to open camera")
            return

        fw, fh = self._camera.frame_size

        # ---- Initialise sub-systems ----
        self._tracker = HandTracker()
        self._zones   = ZoneDetector(zones_cfg, frame_size=(fw, fh))
        self._alerts  = AlertManager(
            on_alert      = self._on_alert,
            on_alert_with_db = self._on_alert_persist,
            critical_pct  = threshold,
            warning_pct   = max(1, threshold // 2),
        )
        self._dtw     = DTWComparator()
        if self._golden_ref:
            self._dtw.set_golden(self._golden_ref)

        # ---- Sequence checker ----
        zone_ids_ordered = [z["id"] for z in zones_cfg]
        self._seq_checker = SequenceChecker(
            expected_sequence = zone_ids_ordered,
            on_violation      = self._on_sequence_violation,
        )

        # ---- Load standard times from config ----
        gc_cfg = self._config.get_golden_cycle(station_id)
        std_times_raw: dict = gc_cfg.get("standard_times", {})
        std_times = {int(k): float(v) for k, v in std_times_raw.items()}

        self._cycles = CycleTracker(
            zone_ids          = [z["id"] for z in zones_cfg],
            alert_threshold   = threshold,
            standard_times    = std_times or None,
            on_cycle_complete = self._on_cycle_complete,
            on_zone_enter     = self._on_zone_enter,
            on_zone_exit      = self._on_zone_exit,
            on_alert          = lambda zid, msg: self._alerts.trigger_threshold(
                zid, 0, 0   # alert manager handles real-time via tick
            ),
            on_sequence_error = self._on_sequence_error,
            on_state_change   = self._on_state_change,
        )

        # Golden recording mode
        if self._mode == self.MODE_GOLDEN:
            self._cycles.recording_mode = True

        # ---- Main capture loop ----
        while self._running:
            ok, frame = self._camera.read()
            if not ok or frame is None:
                self.error_occurred.emit("Camera read failed")
                time.sleep(0.05)
                continue

            if self._mode == self.MODE_PREVIEW:
                # Just stream frames, no processing
                self.frame_ready.emit(frame)
                continue

            # ---- Hand tracking ----
            hand = self._tracker.process(frame)
            if hand.detected:
                self.hand_position.emit(hand.x, hand.y)

            # ---- Zone detection ----
            events = self._zones.update(hand.x, hand.y, hand.detected)
            for zone_id, event_type in events:
                self._cycles.on_zone_event(zone_id, event_type)
                self.zone_event.emit(zone_id, event_type)

                # Sequence violation detection
                if event_type == "enter" and self._seq_checker:
                    self._seq_checker.observe_enter(zone_id)

            # ---- CycleTracker tick (trajectory + threshold check) ----
            self._cycles.tick(hand.x, hand.y)

            # Real-time zone threshold alerting
            if self._cycles.current_zone_id is not None and std_times:
                zid      = self._cycles.current_zone_id
                elapsed  = self._cycles.current_zone_elapsed
                standard = std_times.get(zid, 0)
                if standard > 0:
                    self._alerts.check_realtime(
                        zone_id      = zid,
                        elapsed_sec  = elapsed,
                        standard_sec = standard,
                        cycle_id     = self._db_cycle_id,
                    )

            # ---- Annotate frame ----
            annotated = self._annotate(frame, hand)

            self.frame_ready.emit(annotated)

            # ---- Stats (throttled) ----
            now = time.monotonic()
            if now - self._last_stats_emit > 0.1:
                self._last_stats_emit = now
                self._emit_stats()

        # ---- Cleanup ----
        if self._tracker:
            self._tracker.close()
        if self._camera:
            self._camera.release()

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def _annotate(self, frame: np.ndarray, hand) -> np.ndarray:
        annotated = frame.copy()

        # Draw zones
        if self.show_zones and self._zones:
            active = set(self._zones.occupied_zone_ids())
            self._zones.draw_zones(annotated, alpha=0.20, active_zone_ids=active)

        # Draw hand landmarks
        if self.show_landmarks and hand.detected and self._tracker:
            self._tracker.draw_landmarks(annotated, hand)

        # Draw ghost overlay
        if self.show_ghost and self._dtw and self._dtw.has_golden() and self._cycles:
            prog   = self._cycles.cycle_progress_pct()
            gx, gy = self._dtw.ghost_position_at_progress(prog)
            if gx > 0 and gy > 0:
                # Ghost "shadow" — large translucent circle
                overlay = annotated.copy()
                cv2.circle(overlay, (int(gx), int(gy)), 22, (0, 255, 80), -1)
                cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0, annotated)
                cv2.circle(annotated, (int(gx), int(gy)), 22, (0, 255, 80), 2)

        # Draw cycle state HUD
        if self._cycles and self._cycles.state != CycleState.IDLE:
            elapsed = self._cycles.cycle_elapsed
            state   = self._cycles.state.name
            cv2.putText(
                annotated,
                f"Cycle #{self._cycles.cycle_number}  {elapsed:.1f}s  [{state}]",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
            )

        return annotated

    # ------------------------------------------------------------------
    # Callbacks from CycleTracker (called in THIS thread)
    # ------------------------------------------------------------------

    def _on_cycle_complete(self, record: CycleRecord) -> None:
        # Reset sequence checker สำหรับ cycle ถัดไป
        if self._seq_checker:
            self._seq_checker.reset()

        # Compute DTW score
        dtw_score = 0.0
        if self._dtw and self._dtw.has_golden():
            dtw_score = self._dtw.compare(record.trajectory)

        payload = {
            "cycle_number":    record.cycle_number,
            "total_time":      record.total_time,
            "zone_times":      record.zone_times_dict(),
            "sequence_errors": record.sequence_errors,
            "status":          record.status,
            "dtw_score":       dtw_score,
            "trajectory":      record.trajectory,
            "start_time":      record.start_time,
            "end_time":        record.end_time,
        }
        self.cycle_complete.emit(payload)

    def _on_zone_enter(self, zone_id: int, _elapsed: float) -> None:
        pass   # zone_event signal already emitted above

    def _on_zone_exit(self, zone_id: int, duration: float) -> None:
        pass

    def _on_sequence_error(self, expected: int, actual: int) -> None:
        # legacy callback จาก CycleTracker — ยังคงส่งต่อ alert
        if self._alerts:
            self._alerts.trigger_sequence_violation(
                expected_zone = expected,
                actual_zone   = actual,
                cycle_id      = self._db_cycle_id,
            )

    def _on_sequence_violation(self, v: SequenceViolation) -> None:
        """callback จาก SequenceChecker — emit Qt signal + persist DB"""
        # emit alert signal ไป GUI
        if self._alerts:
            self._alerts.trigger_sequence_violation(
                expected_zone = v.expected_zone or 0,
                actual_zone   = v.actual_zone,
                cycle_id      = self._db_cycle_id,
                detail        = v.violation_type.name,
            )

        # persist DB
        if self._db and self._db_cycle_id is not None:
            try:
                self._db.log_sequence_violation(
                    violation_type  = v.violation_type.name,
                    actual_zone     = v.actual_zone,
                    message         = v.message,
                    cycle_id        = self._db_cycle_id,
                    expected_zone   = v.expected_zone,
                    skipped_zones   = v.skipped_zones,
                    sequence_so_far = v.sequence_so_far,
                )
            except Exception:
                pass  # DB write ไม่ควรทำให้ thread พัง

        # emit ไป GUI ด้วย dict payload
        self.alert_fired.emit({
            "type":           "SEQUENCE_VIOLATION",
            "level":          "CRITICAL",
            "zone_id":        v.actual_zone,
            "message":        v.message,
            "violation_type": v.violation_type.name,
            "expected_zone":  v.expected_zone,
            "skipped_zones":  v.skipped_zones,
        })

    def _on_state_change(self, state: CycleState) -> None:
        self.state_changed.emit(state.name)

    def _on_alert(self, event: AlertEvent) -> None:
        self.alert_fired.emit(event.to_dict())

    def _on_alert_persist(self, event: AlertEvent, cycle_id: int | None) -> None:
        """DB persistence สำหรับ threshold alerts"""
        if self._db and cycle_id is not None:
            try:
                self._db.log_alert(
                    alert_level  = event.alert_level.name,
                    message      = event.message,
                    alert_type   = event.alert_type.name,
                    cycle_id     = cycle_id,
                    zone_id      = event.zone_id,
                    elapsed_sec  = event.elapsed_sec,
                    standard_sec = event.standard_sec,
                    over_pct     = event.over_pct,
                )
            except Exception:
                pass  # DB write ไม่ควรทำให้ thread พัง

    # ------------------------------------------------------------------
    # Stats helper
    # ------------------------------------------------------------------

    def _emit_stats(self) -> None:
        if not self._cycles:
            return
        stats = self._cycles.get_zone_status()
        stats["cycle_number"]   = self._cycles.cycle_number
        stats["cycle_elapsed"]  = self._cycles.cycle_elapsed
        stats["cycle_progress"] = self._cycles.cycle_progress_pct()
        stats["state"]          = self._cycles.state.name
        self.stats_updated.emit(stats)

    # ------------------------------------------------------------------
    # Golden-cycle extraction (call after recording mode completes)
    # ------------------------------------------------------------------

    def get_recorded_cycles(self) -> list[CycleRecord]:
        if self._cycles:
            return list(self._cycles.recorded_cycles)
        return []
