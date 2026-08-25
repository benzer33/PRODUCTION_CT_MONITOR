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


# ---------------------------------------------------------------------------
# Regression: overlapping dual-hand transitions and unstable label handling
# ---------------------------------------------------------------------------

class TestStateChangedHandednessKey:
    """Verify the two-dict state-tracking logic used by monitor_screen.

    monitor_screen._on_point_state_changed now keeps two separate caches:
      - _prev_point_states        keyed by point_id  → drives cycle exit
      - _prev_point_states_by_hand keyed by (point_id, hand) → HUD display

    This class exercises both concerns using the same logic extracted as a
    pure-Python helper so no PyQt5 / camera is required.
    """

    # ------------------------------------------------------------------
    # Helpers that mirror monitor_screen._on_point_state_changed exactly
    # ------------------------------------------------------------------

    @staticmethod
    def _simulate(
        state_log: list[tuple[int, str, str]],
    ) -> list[int]:
        """Simulate the new two-dict logic; return list of point_ids for which
        an exit event was fired (one entry per firing, duplicates allowed if a
        point fires more than once).
        """
        prev_point:    dict[int, str]              = {}
        prev_by_hand:  dict[tuple[int, str], str]  = {}
        exits:         list[int]                   = []

        for pid, state_name, hand in state_log:
            if state_name == "ACTIVE":
                prev_point[pid] = "ACTIVE"
            elif state_name == "COOLDOWN":
                other_active = any(
                    s == "ACTIVE"
                    for (p, h), s in prev_by_hand.items()
                    if p == pid and h != hand
                )
                if not other_active:
                    prev_point[pid] = "COOLDOWN"
            else:
                if prev_point.get(pid, "") != "ACTIVE":
                    prev_point[pid] = state_name

            prev_was_active = prev_point.get(pid) == "COOLDOWN" and \
                              (pid not in prev_point or True)  # checked below

            prev_by_hand[(pid, hand)] = state_name

        # Re-run properly (the helper above has a logic ordering issue for
        # the exit check — redo cleanly).
        prev_point   = {}
        prev_by_hand = {}
        exits        = []

        for pid, state_name, hand in state_log:
            prev_any = prev_point.get(pid, "")

            if state_name == "ACTIVE":
                prev_point[pid] = "ACTIVE"
            elif state_name == "COOLDOWN":
                other_active = any(
                    s == "ACTIVE"
                    for (p, h), s in prev_by_hand.items()
                    if p == pid and h != hand
                )
                if not other_active:
                    prev_point[pid] = "COOLDOWN"
            else:
                if prev_point.get(pid, "") != "ACTIVE":
                    prev_point[pid] = state_name

            if prev_any == "ACTIVE" and prev_point.get(pid) == "COOLDOWN":
                exits.append(pid)

            prev_by_hand[(pid, hand)] = state_name

        return exits

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def _make_detector(self):
        p1 = TriggerPoint(id=1, x=100.0, y=100.0, radius=40.0)
        state_log:   list[tuple[int, str, str]] = []
        trigger_log: list[tuple[int, str]]      = []

        detector = PointTriggerDetector(
            trigger_points  = [p1],
            trigger_confirm = 2,
            clear_confirm   = 2,
            on_trigger      = lambda pid, ts, pos, hand="": trigger_log.append(
                (pid, hand)
            ),
            on_state_change = lambda pid, state, hand="": state_log.append(
                (pid, state.name, hand)
            ),
        )
        detector.start()
        return detector, state_log, trigger_log

    def test_single_hand_normal_exit_fires_once(self):
        """Normal single-hand cycle: ACTIVE → COOLDOWN must fire exactly 1 exit."""
        log = [
            (1, "ARMED",    "Right"),
            (1, "ACTIVE",   "Right"),
            (1, "COOLDOWN", "Right"),
        ]
        exits = self._simulate(log)
        assert exits == [1], f"Expected exactly 1 exit for point 1, got {exits}"

    def test_label_flip_on_exit_does_not_lose_exit_event(self):
        """Regression: MediaPipe label flips from 'Right' (ACTIVE) to 'Left'
        (COOLDOWN) on the exit frame — the exit event must still be fired.

        This is the primary bug being fixed: with the old (point_id, hand) key,
        ACTIVE was stored under ('Right') but COOLDOWN arrived under ('Left'),
        so the ACTIVE→COOLDOWN edge was never detected and the cycle stalled.
        """
        log = [
            (1, "ARMED",    "Right"),
            (1, "ACTIVE",   "Right"),   # hand correctly labelled on ACTIVE frame
            (1, "COOLDOWN", "Left"),    # label flipped on exit frame — the bug
        ]
        exits = self._simulate(log)
        assert len(exits) == 1, (
            f"Label flip on exit must not suppress the exit event. exits={exits}"
        )
        assert exits[0] == 1

    def test_label_flip_other_direction(self):
        """Same bug, opposite flip direction: Left→Right on exit."""
        log = [
            (1, "ACTIVE",   "Left"),
            (1, "COOLDOWN", "Right"),   # label flipped
        ]
        exits = self._simulate(log)
        assert len(exits) == 1, (
            f"Left→Right label flip on exit must still produce exit. exits={exits}"
        )

    def test_two_hands_both_active_exit_fires_once_after_last_hand_leaves(self):
        """Both hands at the same point: the point-level exit should fire
        exactly once — only after the LAST active hand has departed.

        Cycle progression is a single-threaded counter; only one exit per
        point is expected regardless of how many hands were involved.
        """
        log = [
            (1, "ACTIVE",   "Left"),    # Left goes ACTIVE
            (1, "ACTIVE",   "Right"),   # Right also goes ACTIVE
            (1, "COOLDOWN", "Left"),    # Left exits — Right still ACTIVE → no exit yet
            (1, "COOLDOWN", "Right"),   # Right exits — point now clear → exit fires
        ]
        exits = self._simulate(log)
        assert len(exits) == 1, (
            f"Should fire exactly 1 point-level exit when both hands leave. "
            f"exits={exits}"
        )
        assert exits[0] == 1

    def test_two_hands_both_active_left_exits_first_no_premature_exit(self):
        """When Left exits but Right is still ACTIVE, no exit must fire yet."""
        log = [
            (1, "ACTIVE",   "Left"),
            (1, "ACTIVE",   "Right"),
            (1, "COOLDOWN", "Left"),    # Left gone, Right still ACTIVE
        ]
        exits = self._simulate(log)
        assert exits == [], (
            f"Exit must not fire while any hand is still ACTIVE. exits={exits}"
        )

    def test_detector_label_flip_no_lost_exit_via_real_state_machine(self):
        """End-to-end: drive the real PointTriggerDetector, verify COOLDOWN
        arrives in state_log, then confirm the two-dict simulate catches it
        even if we pretend the label flipped.
        """
        detector, state_log, _ = self._make_detector()
        m = detector.get_machine(1, "Right")

        # Clear
        for _ in range(4):
            m.update(on_point=False, hand_pos=(200.0, 200.0))
        state_log.clear()

        # Trigger
        for _ in range(3):
            m.update(on_point=True, hand_pos=(100.0, 100.0))

        # Exit
        m.update(on_point=False, hand_pos=(200.0, 200.0))

        assert any(s == "COOLDOWN" for _, s, _ in state_log), (
            "Detector must emit COOLDOWN on exit"
        )

        # Simulate a label flip: replace the handedness on the COOLDOWN entry
        flipped_log = []
        for pid, s, h in state_log:
            if s == "COOLDOWN" and h == "Right":
                flipped_log.append((pid, s, "Left"))   # pretend label flipped
            else:
                flipped_log.append((pid, s, h))

        exits = self._simulate(flipped_log)
        assert len(exits) == 1, (
            f"Simulated label flip must not suppress exit. exits={exits}"
        )
