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
# Regression: overlapping dual-hand transitions at the same point
# ---------------------------------------------------------------------------

class TestStateChangedHandednessKey:
    """Verify that on_state_change now carries handedness so that two hands
    working the same point near-simultaneously cannot overwrite each other's
    previous-state cache key.

    Scenario
    --------
    - Left  hand: ARMED → ACTIVE → COOLDOWN   (trigger + exit)
    - Right hand: overlaps with ACTIVE while Left is still in its own ACTIVE
      phase, then transitions to COOLDOWN independently.

    Expected: 2 distinct exit events captured, one per hand.
    The exit for Left must never be suppressed by Right's activity, and
    vice versa.
    """

    def _make_detector(self):
        p1 = TriggerPoint(id=1, x=100.0, y=100.0, radius=40.0)
        state_log: list[tuple[int, str, str]] = []
        trigger_log: list[tuple[int, str]] = []

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

    # Simulate the _prev_point_states keying behaviour that lives in the GUI,
    # mirroring monitor_screen._on_point_state_changed, to assert correctness
    # of the fix without requiring PyQt5.
    @staticmethod
    def _simulate_prev_states(
        state_log: list[tuple[int, str, str]],
    ) -> list[tuple[int, str]]:
        """Return exit events as (point_id, handedness) pairs.

        Uses (point_id, handedness) as key — the post-fix behaviour — and
        returns one entry per ACTIVE→COOLDOWN edge.
        """
        prev: dict[tuple[int, str], str] = {}
        exits: list[tuple[int, str]] = []
        for pid, state_name, hand in state_log:
            key = (pid, hand)
            if prev.get(key, "") == "ACTIVE" and state_name == "COOLDOWN":
                exits.append((pid, hand))
            prev[key] = state_name
        return exits

    @staticmethod
    def _simulate_prev_states_broken(
        state_log: list[tuple[int, str, str]],
    ) -> list[tuple[int, str]]:
        """Same but uses only point_id as key — the pre-fix (broken) behaviour."""
        prev: dict[int, str] = {}
        exits: list[tuple[int, str]] = []
        for pid, state_name, hand in state_log:
            if prev.get(pid, "") == "ACTIVE" and state_name == "COOLDOWN":
                exits.append((pid, hand))
            prev[pid] = state_name
        return exits

    def test_two_hands_same_point_each_gets_own_exit_event(self):
        """Both hands must produce an independent exit event; neither is lost."""
        detector, state_log, _ = self._make_detector()

        L = detector.get_machine(1, "Left")
        R = detector.get_machine(1, "Right")

        # Clear both machines first
        for _ in range(4):
            L.update(on_point=False, hand_pos=(200.0, 200.0))
            R.update(on_point=False, hand_pos=(200.0, 200.0))

        state_log.clear()

        # Left enters and triggers
        for _ in range(3):
            L.update(on_point=True, hand_pos=(100.0, 100.0))

        # Right enters and triggers while Left is still ACTIVE
        for _ in range(3):
            R.update(on_point=True, hand_pos=(100.0, 100.0))

        # Left exits first → COOLDOWN
        L.update(on_point=False, hand_pos=(200.0, 200.0))

        # Right exits → COOLDOWN
        R.update(on_point=False, hand_pos=(200.0, 200.0))

        # Verify both hands reached COOLDOWN (i.e. triggered and exited)
        left_states  = [s for pid, s, h in state_log if h == "Left"]
        right_states = [s for pid, s, h in state_log if h == "Right"]
        assert "COOLDOWN" in left_states,  "Left hand should reach COOLDOWN"
        assert "COOLDOWN" in right_states, "Right hand should reach COOLDOWN"

        # Fixed keying produces 2 exit events (one per hand)
        exits_fixed  = self._simulate_prev_states(state_log)
        assert len(exits_fixed) == 2, (
            f"Expected 2 exit events (one per hand), got {exits_fixed}"
        )
        assert (1, "Left")  in exits_fixed
        assert (1, "Right") in exits_fixed

    def test_broken_keying_would_lose_exit_event(self):
        """Demonstrate that the OLD single-key approach CAN drop an exit event
        when the two hands interleave at the same point.

        This test is deliberately asserting the broken behaviour to document
        WHY the fix was needed.  If the detector emits states in a particular
        interleaved order, a point_id-only key loses the Left exit event.
        """
        # Manually craft a state_log that represents the race condition:
        # Right's ACTIVE arrives between Left's ACTIVE and Left's COOLDOWN,
        # overwriting the prev-state so Left's COOLDOWN is no longer preceded
        # by ACTIVE in the prev dict.
        interleaved_log = [
            (1, "ACTIVE",   "Left"),   # Left goes ACTIVE
            (1, "ACTIVE",   "Right"),  # Right goes ACTIVE — overwrites prev[1]
            (1, "COOLDOWN", "Left"),   # Left exits — but prev[1] == "ACTIVE" (Right's)
            (1, "COOLDOWN", "Right"),  # Right exits
        ]
        exits_fixed  = self._simulate_prev_states(interleaved_log)
        exits_broken = self._simulate_prev_states_broken(interleaved_log)

        # Fixed: both exits detected
        assert len(exits_fixed) == 2, f"Fixed should catch both exits: {exits_fixed}"
        # Broken: might lose one or detect wrong hand's exit
        # In this specific interleaving Right's ACTIVE overwrites prev[1],
        # so Left's COOLDOWN is still preceded by ACTIVE in prev[1] — but
        # if the order were Left:ACTIVE, Right:ACTIVE, Right:COOLDOWN, Left:COOLDOWN
        # the broken version would only count Right's exit.
        # We assert they differ to show the fix matters for some orderings.
        interleaved_log_v2 = [
            (1, "ACTIVE",   "Left"),
            (1, "ACTIVE",   "Right"),
            (1, "COOLDOWN", "Right"),  # Right exits first — overwrites prev[1]
            (1, "COOLDOWN", "Left"),   # Left exits — prev[1] is now "COOLDOWN", not "ACTIVE"
        ]
        exits_broken_v2 = self._simulate_prev_states_broken(interleaved_log_v2)
        exits_fixed_v2  = self._simulate_prev_states(interleaved_log_v2)
        # Broken: only 1 exit (Left's is missed)
        assert len(exits_broken_v2) == 1, (
            f"Broken keying should miss Left exit in v2 ordering: {exits_broken_v2}"
        )
        # Fixed: still 2 exits
        assert len(exits_fixed_v2) == 2, (
            f"Fixed keying should still catch both exits in v2: {exits_fixed_v2}"
        )
