"""
vision/point_tracker_thread.py
QThread wrapper สำหรับ PointTriggerDetector

เชื่อม core logic (PointTriggerDetector) กับ GUI thread ผ่าน Qt signals
ทำงานใน background thread ไม่บล็อก UI

Signals ที่ emit ออกไป
──────────────────────
point_triggered(point_id, timestamp, x, y)
    → เมื่อ touch ยืนยันแล้ว  ส่งให้ CycleTracker

hand_position_updated(x, y)
    → ทุกเฟรมที่มีมือ  ส่งให้ overlay painter วาดตำแหน่งมือสด

point_state_changed(point_id, state_name)
    → เมื่อ state machine เปลี่ยน  ส่งให้ GUI แสดงสีจุดตาม state

frame_ready(np.ndarray)
    → annotated frame สำหรับ VideoWidget

error_occurred(str)
    → ข้อความ error (กล้องหลุด ฯลฯ)

cycle_trajectory_ready(list)
    → list ของ dict {x,y,timestamp,t_norm} เมื่อ cycle เสร็จ
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from core.point_trigger_detector import (
    FrameResult,
    PointState,
    PointTriggerDetector,
    TriggerPoint,
    TrajectoryPoint,
)
from data.config_handler import ConfigHandler
from vision.camera_handler import CameraManager, camera_manager_from_config
from vision.skeleton_overlay import landmarks_to_pixels


# สีแสดงสถานะต่างๆ บน overlay (BGR)
_STATE_COLORS: dict[PointState, tuple[int, int, int]] = {
    PointState.IDLE:              (80,  80,  80),    # เทา
    PointState.WAITING_FOR_CLEAR: (0,   165, 255),   # ส้ม
    PointState.ARMED:             (0,   200, 80),    # เขียว
    PointState.TRIGGERED_PENDING: (0,   220, 255),   # เหลือง
    PointState.ACTIVE:            (0,   255, 0),     # เขียวสว่าง
    PointState.COOLDOWN:          (80,  80,  200),   # น้ำเงิน
}


class PointTrackerThread(QThread):
    """
    QThread ที่รัน camera loop + PointTriggerDetector

    Usage
    ─────
        thread = PointTrackerThread(config, trigger_points)
        thread.point_triggered.connect(cycle_tracker.on_point_triggered)
        thread.frame_ready.connect(video_widget.set_frame)
        thread.start()
        ...
        thread.stop()
        thread.finish_trajectory()  # เรียกหลัง cycle เสร็จ
    """

    # ── Qt Signals ──────────────────────────────────────────────────
    point_triggered       = pyqtSignal(int, float, float, float)
    # (point_id, timestamp, hand_x, hand_y)

    hand_position_updated = pyqtSignal(float, float)
    # (x, y) ทุกเฟรมที่มีมือ

    point_state_changed   = pyqtSignal(int, str)
    # (point_id, state_name)

    frame_ready           = pyqtSignal(object)
    # np.ndarray BGR annotated

    error_occurred        = pyqtSignal(str)

    cycle_trajectory_ready = pyqtSignal(list)
    # list of dict {x,y,timestamp,t_norm}

    # Visual-only: emits list of (x, y) pixel tuples for the 21 hand landmarks.
    # NOTE: trigger logic uses ONLY hand_position_updated / point_triggered.
    # This signal is consumed exclusively by the skeleton overlay renderer.
    hand_skeleton_updated = pyqtSignal(list)
    # list of (float, float) pixel coords, length 21 when hand detected, [] otherwise

    def __init__(
        self,
        config:         ConfigHandler,
        trigger_points: list[TriggerPoint],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._config   = config
        self._points   = trigger_points
        self._running  = False

        # PointTriggerDetector สร้างใน run() (ใน thread ที่ถูกต้อง)
        self._detector: Optional[PointTriggerDetector] = None
        self._camera:   Optional[CameraManager]         = None

        # Overlay options
        self.show_points    = True
        self.show_hand      = True
        self.show_state_hud = True

    # ------------------------------------------------------------------
    # Control API (เรียกจาก GUI thread)
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._running = False
        self.wait(3000)

    def reset_cycle(self) -> None:
        """เรียกตอนเริ่ม production cycle ใหม่ (thread-safe ผ่าน flag)"""
        if self._detector:
            self._detector.reset_cycle()

    def request_trajectory(self) -> None:
        """
        ขอ trajectory ของ cycle ล่าสุด → emit cycle_trajectory_ready signal
        เรียกหลัง cycle เสร็จแล้ว
        """
        if self._detector:
            pts = self._detector.finish_trajectory()
            self.cycle_trajectory_ready.emit([p.to_dict() for p in pts])

    # ------------------------------------------------------------------
    # QThread.run — main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._running = True

        # ── เปิดกล้อง ──────────────────────────────────────────────
        station_id = self._config.active_station
        cam_cfg    = self._config.get_camera_config(station_id)
        self._camera = camera_manager_from_config(cam_cfg)
        result = self._camera.open()
        if not result.success:
            self.error_occurred.emit(result.message)
            return

        # ── สร้าง PointTriggerDetector ──────────────────────────────
        station_cfg = self._config.get_station(station_id) or {}
        self._detector = PointTriggerDetector(
            trigger_points    = self._points,
            trigger_confirm   = station_cfg.get("trigger_confirm_frames", 5),
            clear_confirm     = station_cfg.get("clear_confirm_frames", 8),
            use_palm_centroid = True,
            on_trigger        = self._cb_trigger,
            on_state_change   = self._cb_state_change,
            on_hand_position  = self._cb_hand_pos,
        )
        self._detector.start()
        self._detector.reset_cycle()  # เริ่มด้วย WAITING_FOR_CLEAR เสมอ

        # ── Main frame loop ─────────────────────────────────────────
        _consecutive_fails = 0
        _MAX_CONSECUTIVE_FAILS = 60          # ~6 s at 100 ms sleep before giving up
        while self._running:
            ok, frame = self._camera.read()
            if not ok or frame is None:
                _consecutive_fails += 1
                if _consecutive_fails == 1:
                    self.error_occurred.emit("Camera read failed — retrying…")
                if _consecutive_fails >= _MAX_CONSECUTIVE_FAILS:
                    self.error_occurred.emit("Camera lost — stopping.")
                    self._running = False
                    break
                time.sleep(0.1)
                continue
            _consecutive_fails = 0

            result = self._detector.process_frame(frame)

            # Emit skeleton landmarks (visual-only — no trigger logic here)
            if result.full_landmarks:
                h, w = frame.shape[:2]
                self.hand_skeleton_updated.emit(
                    landmarks_to_pixels(result.full_landmarks, w, h)
                )
            else:
                self.hand_skeleton_updated.emit([])

            # Annotate frame สำหรับแสดงผล
            annotated = self._annotate(frame, result)
            self.frame_ready.emit(annotated)

        # ── Cleanup ─────────────────────────────────────────────────
        if self._detector:
            self._detector.close()
        if self._camera:
            self._camera.release()

    # ------------------------------------------------------------------
    # Callbacks จาก PointTriggerDetector (ทำงานใน thread นี้)
    # ------------------------------------------------------------------

    def _cb_trigger(
        self,
        point_id:  int,
        timestamp: float,
        hand_pos:  tuple[float, float],
    ) -> None:
        """เรียกเมื่อ trigger ยืนยัน → emit Qt signal ไป GUI/CycleTracker"""
        self.point_triggered.emit(
            point_id, timestamp,
            float(hand_pos[0]), float(hand_pos[1]),
        )

    def _cb_state_change(
        self,
        point_id: int,
        new_state: PointState,
    ) -> None:
        self.point_state_changed.emit(point_id, new_state.name)

    def _cb_hand_pos(self, x: float, y: float) -> None:
        self.hand_position_updated.emit(x, y)

    # ------------------------------------------------------------------
    # Frame annotation
    # ------------------------------------------------------------------

    def _annotate(
        self,
        frame: np.ndarray,
        result: FrameResult,
    ) -> np.ndarray:
        out = frame.copy()

        if self.show_points and self._detector:
            for pt in self._detector.trigger_points:
                machine  = self._detector.get_machine(pt.point_id)
                state    = machine.state if machine else PointState.IDLE
                color    = _STATE_COLORS.get(state, (100, 100, 100))
                progress = machine.confirm_progress if machine else 0.0

                # วงกลมรัศมีจุด (filled overlay เบาๆ)
                overlay = out.copy()
                cv2.circle(
                    overlay,
                    (int(pt.x), int(pt.y)),
                    int(pt.radius),
                    color, -1,
                )
                cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)

                # วงกลมขอบ
                cv2.circle(
                    out,
                    (int(pt.x), int(pt.y)),
                    int(pt.radius),
                    color, 2,
                )

                # Progress arc (TRIGGERED_PENDING)
                if progress > 0:
                    self._draw_progress_arc(
                        out, (int(pt.x), int(pt.y)),
                        int(pt.radius) + 6,
                        progress, color,
                    )

                # Label (ชื่อ + state)
                label = f"{pt.name} [{state.name}]"
                cv2.putText(
                    out, label,
                    (int(pt.x) - int(pt.radius), int(pt.y) - int(pt.radius) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1, cv2.LINE_AA,
                )

                # แสดง counter ตอน TRIGGERED_PENDING
                if machine and state == PointState.TRIGGERED_PENDING:
                    cnt_text = f"{machine.on_count}/{machine._trigger_frames}"
                    cv2.putText(
                        out, cnt_text,
                        (int(pt.x) - 12, int(pt.y) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 220, 255), 2, cv2.LINE_AA,
                    )

        # วาดตำแหน่งมือ
        if self.show_hand and result.hand_detected:
            hx, hy = int(result.hand_x), int(result.hand_y)
            cv2.circle(out, (hx, hy), 10, (0, 200, 255), -1)
            cv2.circle(out, (hx, hy), 14, (0, 200, 255), 2)

        return out

    @staticmethod
    def _draw_progress_arc(
        frame:    np.ndarray,
        center:   tuple[int, int],
        radius:   int,
        progress: float,
        color:    tuple[int, int, int],
    ) -> None:
        """
        วาด arc แสดงความคืบหน้าของ TRIGGERED_PENDING รอบจุด
        progress: 0.0–1.0
        """
        if progress <= 0:
            return
        end_angle = int(-90 + 360 * progress)   # เริ่มที่ 12 นาฬิกา
        axes = (radius, radius)
        cv2.ellipse(
            frame, center, axes,
            angle=0,
            startAngle=-90,
            endAngle=end_angle,
            color=color,
            thickness=3,
            lineType=cv2.LINE_AA,
        )
