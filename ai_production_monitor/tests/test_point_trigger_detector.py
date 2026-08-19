"""
tests/test_point_trigger_detector.py
Unit tests สำหรับ PointTriggerDetector และ PointStateMachine

วิธีรัน:
    python -m pytest tests/test_point_trigger_detector.py -v
    # หรือจากโฟลเดอร์ ai_production_monitor:
    python -m pytest tests/ -v

ไม่ต้องการกล้อง, MediaPipe, หรือ Qt — ใช้ _process_hand_position() hook
ที่ inject hand position โดยตรงเพื่อ test state machine logic เป็น isolation
"""

from __future__ import annotations

import time
import pytest

import sys
import os

# ──── path setup ────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.point_trigger_detector import (
    PointState,
    PointStateMachine,
    PointTriggerDetector,
    TriggerPoint,
    TrajectoryRecorder,
    TrajectoryPoint,
)


# ============================================================================
# Helpers / Fixtures
# ============================================================================

TRIGGER_CONFIRM = 3    # ใช้ค่าน้อยๆ เพื่อให้ test เร็ว
CLEAR_CONFIRM   = 4


def make_point(pid: int = 1, x: float = 100.0, y: float = 100.0,
               radius: float = 20.0) -> TriggerPoint:
    return TriggerPoint(point_id=pid, name=f"P{pid}", x=x, y=y, radius=radius)


def make_detector(
    pid: int = 1,
    trigger_confirm: int = TRIGGER_CONFIRM,
    clear_confirm:   int = CLEAR_CONFIRM,
) -> tuple[PointTriggerDetector, list[int]]:
    """สร้าง detector พร้อม trigger_log สำหรับเก็บ trigger events"""
    triggered_log: list[int] = []

    def on_trig(point_id, ts, pos):
        triggered_log.append(point_id)

    det = PointTriggerDetector(
        trigger_points   = [make_point(pid)],
        trigger_confirm  = trigger_confirm,
        clear_confirm    = clear_confirm,
        on_trigger       = on_trig,
    )
    # reset() เพื่อ WAITING_FOR_CLEAR (ไม่ start() ซึ่งจะโหลด MediaPipe)
    det.reset_all()
    return det, triggered_log


def feed(
    det:       PointTriggerDetector,
    sequence:  list[tuple[float, float, bool]],
    # list of (x, y, detected)
    ts_start:  float = 0.0,
    dt:        float = 1.0 / 30,
) -> list[bool]:
    """
    ป้อนลำดับ hand positions ให้ detector และคืน list ของ triggered flags
    ต่อ frame (True ถ้า trigger เกิดในเฟรมนั้น)
    """
    ts = ts_start
    results = []
    for x, y, detected in sequence:
        fr = det._process_hand_position(x=x, y=y, detected=detected, timestamp=ts)
        results.append(len(fr.triggered_points) > 0)
        ts += dt
    return results


# ── Position shortcuts สำหรับ Point (100, 100) radius 20 ──────────────────
ON_POINT  = (100.0, 100.0)   # distance = 0  → on_point
NEAR_EDGE = (118.0, 100.0)   # distance = 18 → on_point (18 < 20)
OUTSIDE   = (125.0, 100.0)   # distance = 25 → off_point (25 > 20)


# ============================================================================
# ── Test Case 1: Startup on-point → ต้องไม่ trigger ──────────────────────
# ============================================================================

class TestStartupOnPoint:
    """
    เคส: มือเริ่มอยู่บนจุดตั้งแต่ frame แรก
    Expected: ไม่ trigger จนกว่ามือจะออก → cleared → เข้าใหม่
    """

    def test_no_trigger_while_starting_on_point(self):
        """มืออยู่บน point ตั้งแต่เริ่ม — ห้าม trigger ระหว่างยัง WAITING_FOR_CLEAR"""
        det, log = make_detector()

        # ป้อน on_point 20 เฟรมต่อเนื่อง
        sequence = [ON_POINT] * 20
        feed(det, [(x, y, True) for x, y in sequence])

        assert len(log) == 0, "ห้าม trigger ขณะยังเป็น WAITING_FOR_CLEAR"
        assert det.get_state(1) == PointState.WAITING_FOR_CLEAR

    def test_clears_to_armed_after_leaving(self):
        """หลังมือออกครบ CLEAR_CONFIRM frames → state ต้องเป็น ARMED"""
        det, log = make_detector()

        # เริ่มบน point
        feed(det, [(*ON_POINT, True)] * 5)
        assert det.get_state(1) == PointState.WAITING_FOR_CLEAR

        # ออกจาก point CLEAR_CONFIRM frames
        feed(det, [(*OUTSIDE, True)] * CLEAR_CONFIRM)
        assert det.get_state(1) == PointState.ARMED, \
            "ต้องเป็น ARMED หลังออกจากจุดครบ clear frames"

    def test_triggers_after_leave_then_reenter(self):
        """
        ★ Core Case 1 ★
        มือเริ่มบนจุด → ออก (cleared) → เข้าใหม่ครบ confirm frames
        ต้อง trigger ได้ 1 ครั้ง
        """
        det, log = make_detector()

        # Phase 1: มือบนจุด (state = WAITING_FOR_CLEAR, ห้าม trigger)
        feed(det, [(*ON_POINT, True)] * 5)

        # Phase 2: มือออก (→ ARMED)
        feed(det, [(*OUTSIDE, True)] * CLEAR_CONFIRM)
        assert det.get_state(1) == PointState.ARMED

        # Phase 3: มือเข้าใหม่ ครบ TRIGGER_CONFIRM frames
        triggered = feed(det, [(*ON_POINT, True)] * TRIGGER_CONFIRM)

        assert len(log) == 1, "ต้อง trigger ได้ 1 ครั้ง"
        assert det.get_state(1) == PointState.ACTIVE

    def test_state_sequence_on_startup_on_point(self):
        """ตรวจ state sequence: WAITING_FOR_CLEAR → ARMED → TRIGGERED_PENDING → ACTIVE"""
        det, log = make_detector()
        states: list[PointState] = []

        def capture_state(pid, state):
            states.append(state)

        det._machines[1]._on_state = capture_state

        feed(det, [(*ON_POINT,  True)] * 3)       # WAITING_FOR_CLEAR (ไม่เปลี่ยน)
        feed(det, [(*OUTSIDE,   True)] * CLEAR_CONFIRM)  # → ARMED
        feed(det, [(*ON_POINT,  True)] * TRIGGER_CONFIRM)  # → TRIGGERED_PENDING → ACTIVE

        assert PointState.ARMED             in states
        assert PointState.TRIGGERED_PENDING in states
        assert PointState.ACTIVE            in states
        # ตรวจว่า WAITING_FOR_CLEAR ไม่ถูก skip ไปเป็น ARMED ตั้งแต่เฟรมแรก
        assert states.index(PointState.ARMED) > 0, \
            "ARMED ต้องไม่อยู่ตำแหน่งแรก (ต้องผ่าน WAITING_FOR_CLEAR ก่อน)"


# ============================================================================
# ── Test Case 2: Noise / Jitter บน edge — ห้าม false trigger ──────────────
# ============================================================================

class TestJitterNoise:
    """
    เคส: มือสั่นเข้าออกขอบรัศมีเร็วๆ (noise)
    Expected: ไม่เกิด multiple trigger หรือ false trigger
    """

    def test_single_frame_on_point_does_not_trigger(self):
        """เข้าออก 1 เฟรมสลับกัน — ห้ามเกิด trigger (ต้องการ TRIGGER_CONFIRM frames ต่อเนื่อง)"""
        det, log = make_detector()

        # Clear ให้เป็น ARMED ก่อน
        feed(det, [(*OUTSIDE, True)] * CLEAR_CONFIRM)

        # Jitter: on/off สลับกัน 20 เฟรม
        jitter = [(*ON_POINT, True), (*OUTSIDE, True)] * 10
        feed(det, jitter)

        assert len(log) == 0, "1-frame jitter ห้ามก่อ trigger"

    def test_short_burst_below_confirm_threshold_does_not_trigger(self):
        """อยู่บน point เพียง (TRIGGER_CONFIRM - 1) frames → ห้าม trigger"""
        det, log = make_detector()

        feed(det, [(*OUTSIDE, True)] * CLEAR_CONFIRM)  # → ARMED

        # ป้อน on_point น้อยกว่า threshold แล้วออก
        short_burst = (
            [(*ON_POINT, True)] * (TRIGGER_CONFIRM - 1)
            + [(*OUTSIDE, True)] * 3
        )
        feed(det, short_burst)

        assert len(log) == 0, "burst สั้นกว่า confirm threshold ห้าม trigger"
        assert det.get_state(1) == PointState.ARMED, \
            "หลัง burst สั้น ต้องกลับ ARMED"

    def test_rapid_oscillation_no_duplicate_trigger(self):
        """
        ★ Core Case 2 ★
        มือสั่นแกว่งเข้าออก edge รัวๆ ทำให้มี on/off หลาย burst
        แต่ไม่มี burst ไหนต่อเนื่องครบ TRIGGER_CONFIRM
        → trigger count = 0
        """
        det, log = make_detector()

        feed(det, [(*OUTSIDE, True)] * CLEAR_CONFIRM)  # → ARMED

        # Oscillation: on (2 frames) → off (2 frames) × 10 รอบ
        noise = ([(*ON_POINT, True)] * 2 + [(*OUTSIDE, True)] * 2) * 10
        feed(det, noise)

        assert len(log) == 0, "oscillation ที่ burst < confirm threshold ห้าม trigger"

    def test_partial_then_interrupted_then_full(self):
        """เริ่มนับ confirm แล้วถูก interrupt → ล้าง counter → ต้องนับใหม่ได้"""
        det, log = make_detector()

        feed(det, [(*OUTSIDE, True)] * CLEAR_CONFIRM)  # → ARMED

        # ป้อน on_point 2 frames (ต่ำกว่า threshold)
        feed(det, [(*ON_POINT, True)] * 2)
        assert det.get_state(1) == PointState.TRIGGERED_PENDING

        # interrupt ด้วย off_point
        feed(det, [(*OUTSIDE, True)] * 1)
        assert det.get_state(1) == PointState.ARMED, \
            "interrupt ต้องยกเลิก TRIGGERED_PENDING กลับ ARMED"

        # ป้อน on_point ครบ TRIGGER_CONFIRM ต่อเนื่อง
        feed(det, [(*ON_POINT, True)] * TRIGGER_CONFIRM)

        assert len(log) == 1, "ต้อง trigger ได้หลัง re-enter แบบครบ confirm"

    def test_wobble_on_edge_does_not_multi_trigger(self):
        """มืออยู่ขอบรัศมี (on/off สลับ) ระหว่าง COOLDOWN → ห้ามก่อ trigger ซ้ำ"""
        det, log = make_detector()

        # trigger ครั้งแรกสำเร็จ
        feed(det, [(*OUTSIDE, True)] * CLEAR_CONFIRM)
        feed(det, [(*ON_POINT, True)] * TRIGGER_CONFIRM)
        assert len(log) == 1
        assert det.get_state(1) == PointState.ACTIVE

        # มือออก → COOLDOWN แล้ว wobble บน edge ระหว่าง COOLDOWN
        feed(det, [(*OUTSIDE, True)] * 1)
        assert det.get_state(1) == PointState.COOLDOWN

        wobble = ([(*ON_POINT, True)] * 2 + [(*OUTSIDE, True)] * 2) * 5
        feed(det, wobble)

        assert len(log) == 1, "wobble ใน COOLDOWN ห้ามก่อ trigger ซ้ำ"
        assert det.get_state(1) == PointState.COOLDOWN, \
            "COOLDOWN ต้องคงอยู่ตราบ off_count < clear_confirm"


# ============================================================================
# ── Test Case 3: สองครั้ง trigger ถูกต้อง ────────────────────────────────
# ============================================================================

class TestDoubleValidTrigger:
    """
    เคส: เข้า-ออก-เข้า สองรอบ ครบ confirm/clear frames ทุกครั้ง
    Expected: trigger เกิดขึ้น 2 ครั้งพอดี
    """

    def test_two_valid_triggers(self):
        """
        ★ Core Case 3 ★
        Cycle: out(clear) → in(confirm) → ACTIVE → out(clear) → in(confirm) → ACTIVE
        Expected: trigger_count = 2
        """
        det, log = make_detector()

        # Initial clear
        feed(det, [(*OUTSIDE, True)] * CLEAR_CONFIRM)
        assert det.get_state(1) == PointState.ARMED

        # ── Trigger #1 ──────────────────────────────────────────────
        feed(det, [(*ON_POINT, True)] * TRIGGER_CONFIRM)
        assert det.get_state(1) == PointState.ACTIVE
        assert len(log) == 1

        # ออกจากจุด → COOLDOWN → ARMED
        feed(det, [(*OUTSIDE, True)] * CLEAR_CONFIRM)
        assert det.get_state(1) == PointState.ARMED

        # ── Trigger #2 ──────────────────────────────────────────────
        feed(det, [(*ON_POINT, True)] * TRIGGER_CONFIRM)
        assert det.get_state(1) == PointState.ACTIVE
        assert len(log) == 2, "ต้อง trigger ได้ 2 ครั้งพอดี"

    def test_two_triggers_with_hand_lost_between(self):
        """มือหายออกจากกล้อง (detected=False) ระหว่าง cycle → ต้อง handle ถูกต้อง"""
        det, log = make_detector()

        # Clear
        feed(det, [(*OUTSIDE, True)] * CLEAR_CONFIRM)

        # Trigger #1
        feed(det, [(*ON_POINT, True)] * TRIGGER_CONFIRM)
        assert len(log) == 1

        # มือหาย (detected=False) → treated เหมือน off_point ทุกจุด
        feed(det, [(0.0, 0.0, False)] * CLEAR_CONFIRM)
        # off_point → ออก ACTIVE → COOLDOWN → ARMED
        assert det.get_state(1) in (PointState.ARMED, PointState.COOLDOWN)

        # รอ clear ครบ (ถ้ายัง COOLDOWN)
        feed(det, [(0.0, 0.0, False)] * CLEAR_CONFIRM)
        assert det.get_state(1) == PointState.ARMED

        # Trigger #2
        feed(det, [(*ON_POINT, True)] * TRIGGER_CONFIRM)
        assert len(log) == 2

    def test_trigger_count_matches_enter_count(self):
        """ป้อน N รอบ enter ที่ถูกต้อง → trigger count ต้องเท่ากับ N"""
        det, log = make_detector()

        # Initial clear
        feed(det, [(*OUTSIDE, True)] * CLEAR_CONFIRM)

        N = 4
        for i in range(N):
            feed(det, [(*ON_POINT, True)] * TRIGGER_CONFIRM)   # trigger
            feed(det, [(*OUTSIDE, True)] * CLEAR_CONFIRM)       # clear

        assert len(log) == N, f"ป้อน {N} รอบ ต้อง trigger {N} ครั้ง"

    def test_only_one_signal_per_active_entry(self):
        """อยู่บนจุดนานกว่า confirm threshold มาก → trigger เพียง 1 ครั้งต่อ entry"""
        det, log = make_detector()

        feed(det, [(*OUTSIDE, True)] * CLEAR_CONFIRM)

        # อยู่บนจุดนาน 50 frames
        feed(det, [(*ON_POINT, True)] * 50)

        assert len(log) == 1, "ACTIVE ต้อง emit signal ครั้งเดียวต่อ entry"


# ============================================================================
# ── PointStateMachine unit tests (low-level) ─────────────────────────────
# ============================================================================

class TestPointStateMachine:
    """ทดสอบ PointStateMachine โดยตรง ไม่ผ่าน PointTriggerDetector"""

    def _make_machine(
        self,
        trigger: int = TRIGGER_CONFIRM,
        clear:   int = CLEAR_CONFIRM,
    ) -> tuple[PointStateMachine, list[int]]:
        log: list[int] = []
        m = PointStateMachine(
            point           = make_point(),
            trigger_confirm = trigger,
            clear_confirm   = clear,
            on_trigger_cb   = lambda pid, ts, pos: log.append(pid),
        )
        return m, log

    def test_initial_state_is_waiting_for_clear(self):
        m, _ = self._make_machine()
        assert m.state == PointState.WAITING_FOR_CLEAR

    def test_reset_returns_to_waiting_for_clear(self):
        m, _ = self._make_machine()
        # ทำให้ state เปลี่ยน
        for _ in range(CLEAR_CONFIRM):
            m.update(on_point=False)
        assert m.state == PointState.ARMED
        # reset แล้วกลับ WAITING
        m.reset()
        assert m.state == PointState.WAITING_FOR_CLEAR

    def test_waiting_for_clear_resets_off_count_on_touch(self):
        """ขณะ WAITING_FOR_CLEAR ถ้า on_point → off_count ต้อง reset ไปเป็น 0"""
        m, _ = self._make_machine()

        m.update(on_point=False)  # off_count = 1
        m.update(on_point=False)  # off_count = 2
        m.update(on_point=True)   # on_point → reset off_count
        assert m.off_count == 0
        assert m.state == PointState.WAITING_FOR_CLEAR

    def test_off_count_accumulates_while_waiting(self):
        m, _ = self._make_machine()
        for _ in range(CLEAR_CONFIRM - 1):
            m.update(on_point=False)
        assert m.state == PointState.WAITING_FOR_CLEAR
        assert m.off_count == CLEAR_CONFIRM - 1

    def test_transition_waiting_to_armed(self):
        m, _ = self._make_machine()
        for _ in range(CLEAR_CONFIRM):
            m.update(on_point=False)
        assert m.state == PointState.ARMED

    def test_confirm_progress(self):
        """confirm_progress ต้องขึ้นระหว่าง TRIGGERED_PENDING"""
        m, _ = self._make_machine(trigger=4)
        for _ in range(CLEAR_CONFIRM):
            m.update(on_point=False)

        m.update(on_point=True)   # → TRIGGERED_PENDING, on_count=1
        assert m.state == PointState.TRIGGERED_PENDING
        assert abs(m.confirm_progress - 1 / 4) < 0.01

        m.update(on_point=True)   # on_count=2
        assert abs(m.confirm_progress - 2 / 4) < 0.01

    def test_confirm_progress_is_1_when_active(self):
        m, _ = self._make_machine()
        for _ in range(CLEAR_CONFIRM):
            m.update(on_point=False)
        for _ in range(TRIGGER_CONFIRM):
            m.update(on_point=True)
        assert m.state == PointState.ACTIVE
        assert m.confirm_progress == 1.0

    def test_cooldown_resets_off_count_when_reentered(self):
        """ขณะ COOLDOWN ถ้ากลับเข้า → off_count reset (ยังคง COOLDOWN)"""
        m, _ = self._make_machine()
        # เข้า ACTIVE
        for _ in range(CLEAR_CONFIRM):
            m.update(on_point=False)
        for _ in range(TRIGGER_CONFIRM):
            m.update(on_point=True)
        assert m.state == PointState.ACTIVE
        # ออก → COOLDOWN
        m.update(on_point=False)
        assert m.state == PointState.COOLDOWN
        # นับ off 2 frames
        m.update(on_point=False)
        m.update(on_point=False)
        assert m.off_count == 3
        # กลับเข้า → reset
        m.update(on_point=True)
        assert m.off_count == 0
        assert m.state == PointState.COOLDOWN

    def test_triggered_result_flag(self):
        """PointUpdateResult.triggered ต้อง True เฉพาะเฟรมที่เข้า ACTIVE"""
        m, _ = self._make_machine()
        for _ in range(CLEAR_CONFIRM):
            m.update(on_point=False)
        results = [m.update(on_point=True) for _ in range(TRIGGER_CONFIRM)]
        triggered_frames = [r.triggered for r in results]
        # เฉพาะ frame สุดท้าย (TRIGGERED_CONFIRM-th) เท่านั้นที่ triggered=True
        assert triggered_frames.count(True) == 1
        assert triggered_frames[-1] is True


# ============================================================================
# ── TrajectoryRecorder tests ──────────────────────────────────────────────
# ============================================================================

class TestTrajectoryRecorder:

    def test_records_points_per_cycle(self):
        r = TrajectoryRecorder()
        r.start_cycle()
        r.record(10.0, 20.0)
        r.record(15.0, 25.0)
        pts = r.finish_cycle()
        assert len(pts) == 2
        assert pts[0].x == 10.0 and pts[0].y == 20.0

    def test_t_norm_normalised_0_to_1(self):
        r = TrajectoryRecorder()
        r.start_cycle()
        r.record(0.0, 0.0)
        time.sleep(0.02)
        r.record(1.0, 1.0)
        time.sleep(0.02)
        r.record(2.0, 2.0)
        pts = r.finish_cycle()
        assert pts[0].t_norm == pytest.approx(0.0, abs=0.01)
        assert 0.3 < pts[1].t_norm < 0.7, "กลาง cycle ควรอยู่ใกล้ 0.5"
        assert pts[-1].t_norm == pytest.approx(1.0, abs=0.01)

    def test_no_record_before_start(self):
        r = TrajectoryRecorder()
        r.record(5.0, 5.0)   # ไม่ได้ start ก่อน
        pts = r.finish_cycle()
        assert pts == []

    def test_finish_returns_empty_when_no_points(self):
        r = TrajectoryRecorder()
        r.start_cycle()
        pts = r.finish_cycle()
        assert pts == []

    def test_current_trajectory_before_finish(self):
        r = TrajectoryRecorder()
        r.start_cycle()
        r.record(1.0, 2.0)
        r.record(3.0, 4.0)
        current = r.current_trajectory
        assert len(current) == 2
        assert r.is_active  # ยังไม่ finish

    def test_respects_max_points(self):
        r = TrajectoryRecorder(max_points=5)
        r.start_cycle()
        for i in range(10):
            r.record(float(i), float(i))
        pts = r.finish_cycle()
        assert len(pts) == 5


# ============================================================================
# ── TriggerPoint tests ───────────────────────────────────────────────────
# ============================================================================

class TestTriggerPoint:

    def test_is_on_point_inside(self):
        p = make_point(x=100, y=100, radius=20)
        assert p.is_on_point(100, 100)       # center
        assert p.is_on_point(115, 100)       # edge-ish
        assert p.is_on_point(100, 119)       # near edge

    def test_is_on_point_exactly_on_radius(self):
        p = make_point(x=0, y=0, radius=10)
        assert p.is_on_point(10, 0)          # exactly on circumference (≤ radius)

    def test_is_on_point_outside(self):
        p = make_point(x=100, y=100, radius=20)
        assert not p.is_on_point(125, 100)   # 25 > 20

    def test_distance_to(self):
        p = make_point(x=0, y=0, radius=10)
        assert p.distance_to(3, 4) == pytest.approx(5.0)

    def test_from_dict_roundtrip(self):
        d = {"id": 7, "name": "Station A", "x": 320.0, "y": 240.0, "radius": 30.0}
        p = TriggerPoint.from_dict(d)
        assert p.point_id == 7
        assert p.name == "Station A"
        assert p.to_dict() == d


# ============================================================================
# ── Multi-point detector tests ───────────────────────────────────────────
# ============================================================================

class TestMultiPoint:
    """ตรวจสอบว่า detector ทำงานกับหลายจุดพร้อมกันได้อย่างอิสระ"""

    def make_multi(self):
        triggered_log: list[tuple[int, float]] = []

        def on_trig(pid, ts, pos):
            triggered_log.append((pid, ts))

        pts = [
            make_point(pid=1, x=100, y=100, radius=20),
            make_point(pid=2, x=300, y=300, radius=20),
        ]
        det = PointTriggerDetector(
            trigger_points  = pts,
            trigger_confirm = TRIGGER_CONFIRM,
            clear_confirm   = CLEAR_CONFIRM,
            on_trigger      = on_trig,
        )
        det.reset_all()
        return det, triggered_log

    def test_independent_state_machines(self):
        """จุด 1 และ 2 มี state แยกกัน การเคลื่อนที่ไปจุด 1 ไม่ควรกระทบจุด 2"""
        det, log = self.make_multi()

        # Clear จุด 1
        feed(det, [(150, 100, True)] * CLEAR_CONFIRM)  # outside จุด 1 (150>120), outside จุด 2
        assert det.get_state(1) == PointState.ARMED

        # จุด 2 ยังอยู่ WAITING_FOR_CLEAR (ยังไม่เห็น off จาก มุมมอง pt2)
        # (150, 100) อยู่ห่าง pt2=(300,300) มาก → off_point สำหรับ pt2 ด้วย
        # ดังนั้น pt2 ก็น่าจะ ARMED แล้วเช่นกัน
        assert det.get_state(2) == PointState.ARMED

    def test_trigger_only_the_correct_point(self):
        """เมื่อมือไปแตะจุด 1 → trigger จุด 1 เท่านั้น, จุด 2 ไม่ trigger"""
        det, log = self.make_multi()

        # Clear ทั้งสองจุด (อยู่ที่ตำแหน่งที่ off ทั้งคู่)
        feed(det, [(200, 200, True)] * CLEAR_CONFIRM)

        # แตะจุด 1
        feed(det, [(100, 100, True)] * TRIGGER_CONFIRM)

        assert len([e for e in log if e[0] == 1]) == 1, "จุด 1 ต้อง trigger 1 ครั้ง"
        assert len([e for e in log if e[0] == 2]) == 0, "จุด 2 ห้าม trigger"


if __name__ == "__main__":
    # รันโดยตรง (ไม่ผ่าน pytest) สำหรับ quick smoke test
    import unittest
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestStartupOnPoint,
        TestJitterNoise,
        TestDoubleValidTrigger,
        TestPointStateMachine,
        TestTrajectoryRecorder,
        TestTriggerPoint,
        TestMultiPoint,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
