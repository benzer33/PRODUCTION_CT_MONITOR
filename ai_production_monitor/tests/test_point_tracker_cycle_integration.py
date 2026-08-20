"""
tests/test_point_tracker_cycle_integration.py
=============================================
Integration tests: PointTriggerDetector (StateMachine) + CycleTracker bridge

ทดสอบว่าเมื่อ PointStateMachine เปลี่ยน state ACTIVE → COOLDOWN
(= มือออกจากจุดหลัง trigger) สัญญาณ "exit" ถูกส่งต่อไปยัง CycleTracker
และ CycleTracker เรียก on_cycle_complete callback ครบหนึ่ง cycle

การทดสอบนี้ simulate พฤติกรรมของ _on_point_state_changed ใน
monitor_screen.py / golden_cycle_screen.py โดยไม่ต้องการ PyQt5 หรือกล้อง

รัน:
    python -m pytest tests/test_point_tracker_cycle_integration.py -v
"""

from __future__ import annotations

import time
from typing import Any

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.cycle_tracker import CycleTracker, CycleRecord
from core.point_trigger_detector import (
    PointState,
    PointTriggerDetector,
    TriggerPoint,
)


# ============================================================================
# Helper: simulate the GUI bridge (what _on_point_state_changed does)
# ============================================================================

class PointTrackerBridge:
    """
    Mimics the logic inside _on_point_state_changed() in the GUI screens:

        prev = self._prev_point_states.get(point_id, "")
        self._prev_point_states[point_id] = state_name
        if prev == "ACTIVE" and state_name == "COOLDOWN":
            self._cycle_tracker.on_zone_event(point_id, "exit")

    and the logic inside _on_point_triggered():

        self._cycle_tracker.on_zone_event(point_id, "enter")

    Used to drive CycleTracker from PointTriggerDetector events without
    requiring PyQt5 or a real camera.
    """

    def __init__(self, cycle_tracker: CycleTracker) -> None:
        self._tracker    = cycle_tracker
        self._prev_state: dict[int, str] = {}

    # called when point_triggered fires (ACTIVE entered, signal emitted once)
    def on_point_triggered(
        self,
        point_id: int,
        timestamp: float,
        hand_x: float,
        hand_y: float,
    ) -> None:
        self._tracker.on_zone_event(point_id, "enter")
        self._tracker.tick(hand_x, hand_y)

    # called when point_state_changed fires
    def on_point_state_changed(self, point_id: int, state_name: str) -> None:
        prev = self._prev_state.get(point_id, "")
        self._prev_state[point_id] = state_name
        if prev == "ACTIVE" and state_name == "COOLDOWN":
            self._tracker.on_zone_event(point_id, "exit")

    def reset(self) -> None:
        self._prev_state.clear()


# ============================================================================
# Helper: drive PointTriggerDetector through one full trigger cycle
# ============================================================================

def _trigger_point(machine, trigger_frames: int = 5, clear_frames: int = 8):
    """
    Drive a single PointStateMachine through:
      WAITING_FOR_CLEAR → ARMED → TRIGGERED_PENDING → ACTIVE → COOLDOWN → ARMED

    Returns list of (state_changed, triggered, new_state) per frame.
    """
    events = []

    def _step(on_point: bool):
        result = machine.update(on_point=on_point, hand_pos=(100.0, 100.0))
        events.append(result)
        return result

    # clear phase: off_point until ARMED
    for _ in range(clear_frames + 1):
        _step(on_point=False)

    # enter phase: on_point until ACTIVE
    for _ in range(trigger_frames + 1):
        _step(on_point=True)

    # exit phase: off_point until COOLDOWN (first off frame enters COOLDOWN)
    _step(on_point=False)

    return events


def _make_trigger_point(pid: int, x: float = 100.0, y: float = 100.0, r: float = 40.0):
    return TriggerPoint(point_id=pid, name=f"P{pid}", x=x, y=y, radius=r)


# ============================================================================
# Tests
# ============================================================================

class TestBridgeEnterExit:
    """Unit tests for PointTrackerBridge state-change detection."""

    def _make_bridge_and_tracker(self, zone_ids=(1, 2, 3)):
        completed: list[CycleRecord] = []
        tracker = CycleTracker(
            zone_ids=list(zone_ids),
            on_cycle_complete=completed.append,
        )
        bridge = PointTrackerBridge(tracker)
        return bridge, tracker, completed

    def test_enter_only_does_not_complete_cycle(self):
        """ถ้าเรียกแต่ enter ไม่มี exit → on_cycle_complete ต้องไม่ถูกเรียก"""
        bridge, tracker, completed = self._make_bridge_and_tracker()
        # simulate: P1 enter, P2 enter, P3 enter (no exits)
        bridge.on_point_triggered(1, 0.0, 100.0, 100.0)
        bridge.on_point_triggered(2, 1.0, 100.0, 100.0)
        bridge.on_point_triggered(3, 2.0, 100.0, 100.0)
        assert len(completed) == 0, "cycle ต้องไม่เสร็จถ้าไม่มี exit"

    def test_active_to_cooldown_emits_exit(self):
        """ACTIVE → COOLDOWN transition ต้องส่ง 'exit' ไปยัง CycleTracker"""
        bridge, tracker, completed = self._make_bridge_and_tracker()
        # drive P1 through full enter→exit sequence
        bridge.on_point_triggered(1, 0.0, 100.0, 100.0)
        bridge.on_point_state_changed(1, "ACTIVE")      # prev="" → no exit
        bridge.on_point_state_changed(1, "COOLDOWN")    # prev="ACTIVE" → exit!
        # CycleTracker should have recorded zone 1 exit timing
        assert 1 in tracker._zone_timings, "zone 1 ควรมีใน _zone_timings หลัง exit"
        assert tracker._zone_timings[1].exit_time > 0

    def test_non_active_to_cooldown_does_not_emit_exit(self):
        """TRIGGERED_PENDING → COOLDOWN (ไม่ผ่าน ACTIVE) ต้องไม่ส่ง exit"""
        bridge, tracker, completed = self._make_bridge_and_tracker()
        # Simulate: state goes straight to COOLDOWN without entering ACTIVE
        bridge.on_point_state_changed(1, "TRIGGERED_PENDING")
        bridge.on_point_state_changed(1, "COOLDOWN")
        # no enter was called, so zone_timings should be empty
        assert 1 not in tracker._zone_timings

    def test_multiple_transitions_only_exit_once(self):
        """State เปลี่ยน ACTIVE→COOLDOWN ครั้งเดียว → exit ถูกเรียกครั้งเดียว"""
        exits: list[tuple[int, str]] = []
        tracker = CycleTracker(
            zone_ids=[1],
            on_cycle_complete=lambda r: None,
        )
        original_on_zone = tracker.on_zone_event

        def spy(zone_id, event):
            if event == "exit":
                exits.append((zone_id, event))
            original_on_zone(zone_id, event)

        tracker.on_zone_event = spy  # type: ignore[method-assign]
        bridge = PointTrackerBridge(tracker)

        bridge.on_point_triggered(1, 0.0, 0.0, 0.0)
        bridge.on_point_state_changed(1, "ACTIVE")
        bridge.on_point_state_changed(1, "COOLDOWN")  # exit
        bridge.on_point_state_changed(1, "ARMED")     # no exit
        bridge.on_point_state_changed(1, "COOLDOWN")  # prev="ARMED" → no exit
        assert len(exits) == 1


class TestFullCycleIntegration:
    """Integration: PointStateMachine → Bridge → CycleTracker → on_cycle_complete"""

    def _run_full_cycle(
        self,
        zone_ids: list[int],
        trigger_frames: int = 3,
        clear_frames: int = 4,
    ) -> list[CycleRecord]:
        """
        Simulate one complete production cycle through all zones.
        Returns list of completed CycleRecords.
        """
        completed: list[CycleRecord] = []
        tracker = CycleTracker(
            zone_ids=zone_ids,
            on_cycle_complete=completed.append,
        )
        bridge = PointTrackerBridge(tracker)

        # Create trigger points and machines (simulated — no camera)
        machines = {
            pid: _make_trigger_point(pid).__class__  # type: ignore
            for pid in zone_ids
        }

        # Use PointTriggerDetector to manage machines
        points  = [_make_trigger_point(pid) for pid in zone_ids]
        detector = PointTriggerDetector(
            trigger_points   = points,
            trigger_confirm  = trigger_frames,
            clear_confirm    = clear_frames,
            on_trigger       = lambda pid, ts, pos: bridge.on_point_triggered(
                pid, ts, pos[0], pos[1]
            ),
            on_state_change  = lambda pid, state: bridge.on_point_state_changed(
                pid, state.name
            ),
        )
        detector.start()
        detector.reset_cycle()

        # For each zone: clear → enter → trigger → exit
        for pid in zone_ids:
            machine = detector.get_machine(pid)
            assert machine is not None

            # clear phase
            for _ in range(clear_frames + 2):
                machine.update(on_point=False, hand_pos=(200.0, 200.0))

            # enter + trigger phase
            for _ in range(trigger_frames + 1):
                machine.update(on_point=True, hand_pos=(100.0, 100.0))

            # exit phase (first off frame → COOLDOWN)
            result = machine.update(on_point=False, hand_pos=(200.0, 200.0))
            # manually fire state-change for COOLDOWN (detector fires callback)
            # The detector already calls on_state_change via _set_state,
            # so bridge.on_point_state_changed already fired during machine.update

        return completed

    def test_one_complete_cycle_fires_callback(self):
        """จำลอง 1 cycle เต็ม (P1→P2→P3 enter→exit) → on_cycle_complete ถูกเรียก 1 ครั้ง"""
        completed = self._run_full_cycle([1, 2, 3])
        assert len(completed) == 1, (
            f"on_cycle_complete ควรถูกเรียก 1 ครั้ง แต่ได้ {len(completed)}"
        )

    def test_cycle_record_has_all_zone_timings(self):
        """CycleRecord ต้องมี timing ครบทั้ง 3 zone"""
        completed = self._run_full_cycle([1, 2, 3])
        assert len(completed) == 1
        rec = completed[0]
        for zid in [1, 2, 3]:
            assert zid in rec.zone_timings, f"zone {zid} ควรมีใน zone_timings"
            assert rec.zone_timings[zid].duration() > 0, \
                f"zone {zid} ควรมี duration > 0"

    def test_cycle_record_total_time_positive(self):
        """CycleRecord.total_time ต้องมากกว่า 0"""
        completed = self._run_full_cycle([1, 2, 3])
        assert completed[0].total_time > 0

    def test_two_consecutive_cycles(self):
        """ทำ 2 รอบติดกัน → on_cycle_complete ถูกเรียก 2 ครั้ง"""
        completed: list[CycleRecord] = []
        tracker = CycleTracker(
            zone_ids=[1, 2, 3],
            on_cycle_complete=completed.append,
        )
        bridge = PointTrackerBridge(tracker)
        points = [_make_trigger_point(pid) for pid in [1, 2, 3]]
        detector = PointTriggerDetector(
            trigger_points  = points,
            trigger_confirm = 3,
            clear_confirm   = 4,
            on_trigger      = lambda pid, ts, pos: bridge.on_point_triggered(
                pid, ts, pos[0], pos[1]
            ),
            on_state_change = lambda pid, state: bridge.on_point_state_changed(
                pid, state.name
            ),
        )
        detector.start()

        for _cycle in range(2):
            detector.reset_cycle()
            bridge.reset()
            for pid in [1, 2, 3]:
                machine = detector.get_machine(pid)
                for _ in range(6):
                    machine.update(on_point=False, hand_pos=(200.0, 200.0))
                for _ in range(4):
                    machine.update(on_point=True, hand_pos=(100.0, 100.0))
                machine.update(on_point=False, hand_pos=(200.0, 200.0))

        assert len(completed) == 2, (
            f"ทำ 2 รอบ → ควรได้ 2 records แต่ได้ {len(completed)}"
        )

    def test_no_cycle_without_exit(self):
        """ถ้าไม่มี exit เลย → cycle ต้องไม่เสร็จ (regression guard)"""
        completed: list[CycleRecord] = []
        tracker = CycleTracker(
            zone_ids=[1, 2, 3],
            on_cycle_complete=completed.append,
        )
        bridge = PointTrackerBridge(tracker)

        # send only enters, no exits
        for pid in [1, 2, 3]:
            bridge.on_point_triggered(pid, float(pid), 100.0, 100.0)

        assert len(completed) == 0, "ต้องไม่มี cycle เสร็จถ้าไม่มี exit"


class TestBridgeResetBetweenSessions:
    """_prev_point_states ต้อง reset ระหว่าง session เพื่อกัน stale state"""

    def test_stale_active_state_does_not_leak(self):
        """
        ถ้าหยุด session ขณะ point อยู่ใน ACTIVE แล้วเริ่ม session ใหม่
        state เก่าต้องถูก reset ไม่งั้น COOLDOWN ใน session ใหม่จะยิง exit ผิดที่
        """
        completed: list[CycleRecord] = []
        tracker = CycleTracker(
            zone_ids=[1, 2, 3],
            on_cycle_complete=completed.append,
        )
        bridge = PointTrackerBridge(tracker)

        # Session 1: enter P1 but stop before exit
        bridge.on_point_triggered(1, 0.0, 100.0, 100.0)
        bridge.on_point_state_changed(1, "ACTIVE")
        # --- session stops here, bridge.reset() called ---
        bridge.reset()

        # Session 2: COOLDOWN arrives first (hand still on point from last session)
        bridge.on_point_state_changed(1, "COOLDOWN")
        # prev should be "" now (not "ACTIVE") so no spurious exit
        assert 1 not in tracker._zone_timings or \
               tracker._zone_timings.get(1, None) is None or \
               tracker._zone_timings[1].exit_time == 0, \
            "หลัง reset ต้องไม่ยิง exit จาก state เก่า"


if __name__ == "__main__":
    import unittest
    pytest.main([__file__, "-v"])
