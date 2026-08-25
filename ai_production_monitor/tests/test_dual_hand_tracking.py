"""
tests/test_dual_hand_tracking.py
Phase-1 dual-hand tracking unit tests.

Verifies:
  1. (point_id, "Left") and (point_id, "Right") state machines are completely
     independent: triggering one does NOT change the state of the other.
  2. Both hands can be in different states simultaneously at the same point.
  3. zone_hands_dict() on CycleRecord carries the correct hand labels.
  4. on_zone_event() backward-compat: calling without handedness still works.

No camera / MediaPipe required — uses _process_hand_position() injection hook.
"""

from __future__ import annotations

import sys
import os
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.point_trigger_detector import (
    PointState,
    PointTriggerDetector,
    TriggerPoint,
)
from core.cycle_tracker import CycleTracker, ZoneTiming


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRIGGER_CONFIRM = 3
CLEAR_CONFIRM   = 4

ON_POINT  = (100.0, 100.0)
OUTSIDE   = (200.0, 200.0)


def make_point(pid: int = 1) -> TriggerPoint:
    return TriggerPoint(
        point_id=pid, name=f"P{pid}",
        x1=80.0, y1=80.0, x2=120.0, y2=120.0,
    )


def make_detector(pids=(1,)):
    """Create a PointTriggerDetector with per-hand logging."""
    trigger_log: list[tuple[int, str]] = []  # (point_id, handedness)

    def on_trig(point_id, ts, pos, handedness=""):
        trigger_log.append((point_id, handedness))

    det = PointTriggerDetector(
        trigger_points  = [make_point(p) for p in pids],
        trigger_confirm = TRIGGER_CONFIRM,
        clear_confirm   = CLEAR_CONFIRM,
        on_trigger      = on_trig,
    )
    det.reset_all()
    return det, trigger_log


def push_frames(det, x, y, detected, n, handedness="Right", ts_start=0.0, dt=1/30):
    """Inject `n` identical frames for one hand."""
    ts = ts_start
    for _ in range(n):
        det._process_hand_position(x=x, y=y, detected=detected,
                                   timestamp=ts, handedness=handedness)
        ts += dt
    return ts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHandIndependence:
    """Core requirement: Left and Right machines at the same point are isolated."""

    def test_left_trigger_does_not_affect_right_state(self):
        """
        Left hand goes through WAITING_FOR_CLEAR → ARMED → TRIGGERED_PENDING
        → ACTIVE.  Right hand's state should still be WAITING_FOR_CLEAR.
        """
        det, log = make_detector()

        # Clear left hand first (move off-point)
        push_frames(det, *OUTSIDE, True, CLEAR_CONFIRM + 1, handedness="Left")
        # Clear right hand
        push_frames(det, *OUTSIDE, True, CLEAR_CONFIRM + 1, handedness="Right")

        # Left hand enters trigger zone
        push_frames(det, *ON_POINT, True, TRIGGER_CONFIRM + 1, handedness="Left")

        left_state  = det.get_state(1, "Left")
        right_state = det.get_state(1, "Right")

        assert left_state == PointState.ACTIVE, f"Expected Left=ACTIVE, got {left_state}"
        assert right_state != PointState.ACTIVE, \
            f"Right should NOT be ACTIVE after only Left triggered, got {right_state}"

    def test_right_trigger_does_not_affect_left_state(self):
        """Mirror of previous test — right triggers, left unaffected."""
        det, log = make_detector()

        push_frames(det, *OUTSIDE, True, CLEAR_CONFIRM + 1, handedness="Left")
        push_frames(det, *OUTSIDE, True, CLEAR_CONFIRM + 1, handedness="Right")

        push_frames(det, *ON_POINT, True, TRIGGER_CONFIRM + 1, handedness="Right")

        left_state  = det.get_state(1, "Left")
        right_state = det.get_state(1, "Right")

        assert right_state == PointState.ACTIVE, f"Expected Right=ACTIVE, got {right_state}"
        assert left_state != PointState.ACTIVE, \
            f"Left should NOT be ACTIVE after only Right triggered, got {left_state}"

    def test_both_hands_can_be_in_different_states_simultaneously(self):
        """
        Left = ACTIVE (triggered),  Right = WAITING_FOR_CLEAR (still on-point
        at startup) — both at the same zone, same time.
        """
        det, log = make_detector()

        # Keep Right on-point from start → stays WAITING_FOR_CLEAR (never cleared)
        push_frames(det, *ON_POINT, True, TRIGGER_CONFIRM + 2, handedness="Right")

        # Clear Left and trigger it
        push_frames(det, *OUTSIDE, True, CLEAR_CONFIRM + 1, handedness="Left")
        push_frames(det, *ON_POINT, True, TRIGGER_CONFIRM + 1, handedness="Left")

        left_state  = det.get_state(1, "Left")
        right_state = det.get_state(1, "Right")

        assert left_state == PointState.ACTIVE
        assert right_state == PointState.WAITING_FOR_CLEAR

    def test_trigger_log_records_handedness(self):
        """on_trigger callback receives the correct handedness string."""
        det, log = make_detector()

        push_frames(det, *OUTSIDE, True, CLEAR_CONFIRM + 1, handedness="Left")
        push_frames(det, *ON_POINT, True, TRIGGER_CONFIRM + 1, handedness="Left")

        assert len(log) >= 1
        pid, hand = log[-1]
        assert hand == "Left", f"Expected 'Left' in trigger log, got '{hand}'"

    def test_separate_trigger_logs_per_hand(self):
        """Triggering both hands logs both handedness values."""
        det, log = make_detector()

        push_frames(det, *OUTSIDE, True, CLEAR_CONFIRM + 1, handedness="Left")
        push_frames(det, *OUTSIDE, True, CLEAR_CONFIRM + 1, handedness="Right")
        push_frames(det, *ON_POINT, True, TRIGGER_CONFIRM + 1, handedness="Left")
        push_frames(det, *ON_POINT, True, TRIGGER_CONFIRM + 1, handedness="Right")

        hands_logged = {h for _, h in log}
        assert "Left"  in hands_logged, "Left trigger not logged"
        assert "Right" in hands_logged, "Right trigger not logged"


class TestCycleTrackerHandedness:
    """ZoneTiming.hand and zone_hands_dict() are populated correctly."""

    def _make_tracker(self):
        records = []
        def on_complete(rec):
            records.append(rec)
        tracker = CycleTracker(
            zone_ids=[1, 2, 3],
            on_cycle_complete=on_complete,
        )
        return tracker, records

    def test_zone_timing_records_hand(self):
        """ZoneTiming.hand is set to the handedness passed to on_zone_event."""
        tracker, _ = self._make_tracker()
        tracker.on_zone_event(1, "enter", "Left")
        zt = tracker._zone_timings.get(1)
        assert zt is not None
        assert zt.hand == "Left"

    def test_zone_hands_dict_populated(self):
        """zone_hands_dict() returns a mapping of zone_id→hand after cycle."""
        tracker, records = self._make_tracker()

        # Simulate a full cycle: enter+exit zones 1, 2, 3
        t = time.monotonic()
        tracker.on_zone_event(1, "enter", "Right")
        tracker.on_zone_event(1, "exit")
        tracker.on_zone_event(2, "enter", "Left")
        tracker.on_zone_event(2, "exit")
        tracker.on_zone_event(3, "enter", "Right")
        tracker.on_zone_event(3, "exit")

        assert len(records) == 1
        rec = records[0]
        hands = rec.zone_hands_dict()
        assert hands.get("1") == "Right"
        assert hands.get("2") == "Left"
        assert hands.get("3") == "Right"

    def test_on_zone_event_backward_compat_no_handedness(self):
        """Calling on_zone_event without handedness arg should not raise."""
        tracker, _ = self._make_tracker()
        # Must not raise TypeError
        tracker.on_zone_event(1, "enter")
        zt = tracker._zone_timings.get(1)
        assert zt is not None
        assert zt.hand == ""   # default empty string
