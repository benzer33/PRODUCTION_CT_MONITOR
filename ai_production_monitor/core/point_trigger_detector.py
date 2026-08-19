"""
core/point_trigger_detector.py
Rising-edge state machine สำหรับตรวจจับการแตะ "จุด trigger" แทนโซนสี่เหลี่ยม

ทำไมต้องเปลี่ยนจาก Zone-based เป็น Point-based + Rising-Edge
═══════════════════════════════════════════════════════════════
ระบบ Zone-based เดิมใช้ "level-check" — เช็คแค่ว่ามืออยู่ในโซนไหม
ในเฟรมปัจจุบัน  ปัญหาคือถ้ามือค้างอยู่ในโซนตั้งแต่ก่อนเริ่มระบบ
(หรือข้ามมาจากรอบก่อน) จะไม่มี transition จาก "นอก→ใน" เลย
ทำให้ระบบไม่รู้ว่าควร fire trigger หรือไม่

แนวทางแก้ไข (Rising-Edge + WAITING_FOR_CLEAR)
───────────────────────────────────────────────
แต่ละจุดมี state machine เป็นของตัวเอง  trigger จะเกิดก็ต่อเมื่อ
state ผ่านลำดับ: ARMED → TRIGGERED_PENDING → ACTIVE เท่านั้น
ระบบเริ่มที่ WAITING_FOR_CLEAR เสมอ เพื่อบังคับให้เห็น "ออกจากจุด"
ก่อนจึงจะ arm ได้


State Transition Diagram
════════════════════════

 [init / reset_cycle()]
         │
         ▼
 ┌───────────────────┐   off_point ต่อเนื่อง
 │  WAITING_FOR_     │─────────────────────────────────────────────►┐
 │     CLEAR         │   ≥ CLEAR_CONFIRM_FRAMES                     │
 │                   │◄─ on_point: reset off_counter (ยังรออยู่)     │
 └───────────────────┘                                              │
                                                                    ▼
 ┌──────────────────────────────────────────────────────── ┌────────────────┐
 │                                                         │    ARMED       │
 │            ┌───────────────────────────────────────────►│  (พร้อม trigger)│
 │            │                                            └───────┬────────┘
 │            │  off_point ตรวจพบ                                  │ on_point ตรวจพบ
 │            │  (interrupt, ยกเลิก)                               ▼
 │            │                                            ┌────────────────┐
 │            │                                            │  TRIGGERED_    │
 │            └────────────────────────────────────────────│   PENDING      │
 │                                                         │ (นับ confirm   │
 │                                                         │  frames)       │
 │                                                         └───────┬────────┘
 │                                                                 │ on_point ต่อเนื่อง
 │                                                                 │ ≥ TRIGGER_CONFIRM_FRAMES
 │                                                                 ▼
 │                                                         ┌────────────────┐
 │                                                         │    ACTIVE      │◄──┐
 │                                               on_point: │  (trigger fire!) │  │ on_point
 │                                               stay here │  ← emit signal │  │ (ยังอยู่)
 │                                                         └───────┬────────┘  │
 │                                                                 │ off_point  │
 │                                                                 ▼           │
 │                                                         ┌────────────────┐  │
 │                                                         │   COOLDOWN     │──┘
 │   off_point ต่อเนื่อง                                   │  (รอ clear     │
 └─────────────────────────────────────────────────────────│   hysteresis)  │
     ≥ CLEAR_CONFIRM_FRAMES  →  ARMED                      └────────────────┘

กฎสำคัญ
────────
1. ทุก point เริ่มที่ WAITING_FOR_CLEAR (ไม่ใช่ ARMED) เมื่อ init และ reset_cycle()
2. TRIGGERED_PENDING ยกเลิกทันทีถ้าเห็น off_point แม้แต่เฟรมเดียว
3. COOLDOWN ต้องเห็น off_point ต่อเนื่อง ≥ CLEAR_CONFIRM_FRAMES ก่อนกลับ ARMED
   (ป้องกัน re-trigger ตอนมือสั่นอยู่ขอบรัศมี)
4. on_point ขณะ COOLDOWN → reset off_counter (stay COOLDOWN)
5. ACTIVE stay ตราบมือยังอยู่บนจุด — emit signal ครั้งเดียวตอน enter ACTIVE
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional


# ============================================================================
# Config dataclass สำหรับ 1 จุด trigger
# ============================================================================

@dataclass
class TriggerPoint:
    """
    นิยามจุด trigger หนึ่งจุด  โหลดมาจาก station_config.json
    รูปแบบ JSON:
      {"id": 1, "name": "Pick Part A", "x": 320, "y": 240, "radius": 30}
    """
    point_id: int
    name:     str
    x:        float   # pixel x ในพิกัด frame
    y:        float   # pixel y ในพิกัด frame
    radius:   float   # รัศมีตรวจจับ (pixel)

    @classmethod
    def from_dict(cls, d: dict) -> "TriggerPoint":
        return cls(
            point_id = d["id"],
            name     = d.get("name", f"Point {d['id']}"),
            x        = float(d["x"]),
            y        = float(d["y"]),
            radius   = float(d.get("radius", 30)),
        )

    def to_dict(self) -> dict:
        return {
            "id":     self.point_id,
            "name":   self.name,
            "x":      self.x,
            "y":      self.y,
            "radius": self.radius,
        }

    def distance_to(self, px: float, py: float) -> float:
        """คำนวณระยะ Euclidean จากจุด (px, py) มายังจุด trigger นี้"""
        return math.hypot(px - self.x, py - self.y)

    def is_on_point(self, px: float, py: float) -> bool:
        """True ถ้า (px, py) อยู่ภายในรัศมีของจุดนี้"""
        return self.distance_to(px, py) <= self.radius


# ============================================================================
# State Enum
# ============================================================================

class PointState(Enum):
    """
    State ของ state machine แต่ละจุด

    IDLE               — ระบบยังไม่เริ่ม / ถูก shutdown
    WAITING_FOR_CLEAR  — เริ่มต้น/reset ใหม่ รอเห็นมือออกจากจุดก่อน
    ARMED              — พร้อมรับ trigger รอบถัดไป
    TRIGGERED_PENDING  — เห็นมือเข้ามา กำลังนับ confirm frames
    ACTIVE             — trigger ยืนยันแล้ว (signal fired ครั้งเดียว)
    COOLDOWN           — หลัง trigger รอ hysteresis ก่อน armed ใหม่
    """
    IDLE              = auto()
    WAITING_FOR_CLEAR = auto()
    ARMED             = auto()
    TRIGGERED_PENDING = auto()
    ACTIVE            = auto()
    COOLDOWN          = auto()


# ============================================================================
# ผลลัพธ์จาก update() ต่อ 1 frame ต่อ 1 จุด
# ============================================================================

@dataclass
class PointUpdateResult:
    """ผลลัพธ์จาก PointStateMachine.update() ใน 1 เฟรม"""
    point_id:      int
    prev_state:    PointState
    new_state:     PointState
    triggered:     bool    # True ในเฟรมที่ state เปลี่ยนเป็น ACTIVE
    state_changed: bool

    @property
    def name(self) -> str:
        return self.new_state.name


# ============================================================================
# PointStateMachine — core logic ต่อ 1 จุด
# ============================================================================

class PointStateMachine:
    """
    State machine Rising-Edge สำหรับจุด trigger 1 จุด

    ออกแบบให้ testable โดยไม่ต้องมี Qt หรือ MediaPipe —
    รับแค่ on_point: bool ต่อ frame

    Parameters
    ──────────
    point           : TriggerPoint config
    trigger_confirm : จำนวนเฟรม on_point ต่อเนื่องก่อน trigger จริง
    clear_confirm   : จำนวนเฟรม off_point ต่อเนื่องก่อน armed ใหม่
    on_trigger_cb   : callback(point_id, timestamp) เมื่อ state เข้า ACTIVE
    on_state_cb     : callback(point_id, new_state) เมื่อ state เปลี่ยน
    """

    def __init__(
        self,
        point:           TriggerPoint,
        trigger_confirm: int = 5,
        clear_confirm:   int = 8,
        on_trigger_cb:   Optional[Callable[[int, float, tuple], None]] = None,
        on_state_cb:     Optional[Callable[[int, PointState], None]]   = None,
    ) -> None:
        self.point           = point
        self._trigger_frames = trigger_confirm
        self._clear_frames   = clear_confirm
        self._on_trigger     = on_trigger_cb
        self._on_state       = on_state_cb

        # runtime counters
        self._state:      PointState = PointState.IDLE
        self._on_count:   int = 0   # เฟรม on_point ต่อเนื่อง (TRIGGERED_PENDING)
        self._off_count:  int = 0   # เฟรม off_point ต่อเนื่อง (COOLDOWN / WAITING)

        # เริ่มต้น state จริงผ่าน reset()
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset กลับสู่ WAITING_FOR_CLEAR เสมอ (ไม่ใช่ ARMED)
        เรียกตอน init และตอนเริ่ม cycle ใหม่
        ป้องกัน false trigger ถ้ามืออยู่บนจุดตั้งแต่ก่อนเริ่ม
        """
        self._set_state(PointState.WAITING_FOR_CLEAR)
        self._on_count  = 0
        self._off_count = 0

    def update(
        self,
        on_point:    bool,
        hand_pos:    tuple[float, float] = (0.0, 0.0),
        timestamp:   float | None = None,
    ) -> PointUpdateResult:
        """
        อัปเดต state machine 1 เฟรม

        Parameters
        ──────────
        on_point  : True ถ้ามืออยู่ภายในรัศมีของจุดนี้ในเฟรมนี้
        hand_pos  : (x, y) pixel ของมือ ใช้ส่งใน callback
        timestamp : เวลา (monotonic) ถ้า None ใช้ time.monotonic()

        Returns
        ───────
        PointUpdateResult — สถานะหลัง update รวมถึง triggered flag
        """
        ts        = timestamp if timestamp is not None else time.monotonic()
        prev      = self._state
        triggered = False

        # ── transition logic ──────────────────────────────────────────

        if self._state == PointState.WAITING_FOR_CLEAR:
            # รอเห็น off_point ต่อเนื่องก่อน armed
            if on_point:
                self._off_count = 0   # มือยังอยู่บนจุด → รีเซ็ต ยังรออยู่
            else:
                self._off_count += 1
                if self._off_count >= self._clear_frames:
                    # เห็น off_point ต่อเนื่องครบ → armed แล้ว
                    self._set_state(PointState.ARMED)
                    self._off_count = 0

        elif self._state == PointState.ARMED:
            if on_point:
                # เห็นมือเข้ามา → เริ่มนับ confirm frames
                self._on_count = 1
                self._set_state(PointState.TRIGGERED_PENDING)

        elif self._state == PointState.TRIGGERED_PENDING:
            if on_point:
                self._on_count += 1
                if self._on_count >= self._trigger_frames:
                    # ★ TRIGGER CONFIRMED ★
                    self._set_state(PointState.ACTIVE)
                    triggered = True
                    self._on_count  = 0
                    self._off_count = 0
                    # เรียก callback (จะถูกเรียกครั้งเดียว)
                    if self._on_trigger:
                        self._on_trigger(self.point.point_id, ts, hand_pos)
            else:
                # มือออกก่อนครบ confirm → ยกเลิก กลับ ARMED
                self._on_count = 0
                self._set_state(PointState.ARMED)

        elif self._state == PointState.ACTIVE:
            if not on_point:
                # มือออกจากจุดหลัง trigger → เข้า COOLDOWN
                self._off_count = 1
                self._set_state(PointState.COOLDOWN)
            # ถ้า on_point ยังคง ACTIVE (ไม่ re-trigger, signal fire แล้ว)

        elif self._state == PointState.COOLDOWN:
            if on_point:
                # มือกลับเข้ามา → reset off_counter (ยังอยู่ใน COOLDOWN)
                self._off_count = 0
            else:
                self._off_count += 1
                if self._off_count >= self._clear_frames:
                    # ออกจากจุดต่อเนื่องครบ → armed ใหม่
                    self._set_state(PointState.ARMED)
                    self._off_count = 0

        # IDLE ไม่มี transition (ต้อง reset() ก่อน)

        return PointUpdateResult(
            point_id      = self.point.point_id,
            prev_state    = prev,
            new_state     = self._state,
            triggered     = triggered,
            state_changed = (self._state != prev),
        )

    @property
    def state(self) -> PointState:
        return self._state

    @property
    def on_count(self) -> int:
        """เฟรม on_point ต่อเนื่องปัจจุบัน (ใช้ debug/visualise)"""
        return self._on_count

    @property
    def off_count(self) -> int:
        """เฟรม off_point ต่อเนื่องปัจจุบัน (ใช้ debug/visualise)"""
        return self._off_count

    @property
    def confirm_progress(self) -> float:
        """
        ความคืบหน้าการ confirm เป็น % [0.0–1.0]
        ใช้วาด progress arc รอบจุดบน overlay
        """
        if self._state == PointState.TRIGGERED_PENDING and self._trigger_frames > 0:
            return min(self._on_count / self._trigger_frames, 1.0)
        if self._state == PointState.ACTIVE:
            return 1.0
        return 0.0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_state(self, new: PointState) -> None:
        """เปลี่ยน state + เรียก on_state_cb ถ้ามี"""
        if new != self._state:
            self._state = new
            if self._on_state:
                self._on_state(self.point.point_id, new)


# ============================================================================
# TrajectoryRecorder — บันทึก path ของมือต่อ cycle
# ============================================================================

@dataclass
class TrajectoryPoint:
    x:         float
    y:         float
    timestamp: float       # time.monotonic()
    t_norm:    float = 0.0  # [0–1] ปกติหลัง cycle เสร็จ

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y,
                "timestamp": self.timestamp, "t_norm": self.t_norm}


class TrajectoryRecorder:
    """
    บันทึก path ของมือในระหว่าง cycle
    ไม่ผูกกับ point state machine โดยตรง — CycleTracker เรียกใช้แยกต่างหาก

    Usage:
        recorder.start_cycle()
        # ทุกเฟรม:
        if hand_detected:
            recorder.record(x, y)
        trajectory = recorder.finish_cycle()  # normalise t_norm แล้วคืนค่า
    """

    def __init__(self, max_points: int = 10_000) -> None:
        self._max    = max_points
        self._points: list[TrajectoryPoint] = []
        self._start:  float = 0.0
        self._active: bool  = False

    def start_cycle(self) -> None:
        self._points = []
        self._start  = time.monotonic()
        self._active = True

    def record(self, x: float, y: float) -> None:
        """เพิ่มจุดพิกัดมือ ณ เฟรมปัจจุบัน (เรียกทุกเฟรมที่ hand detected)"""
        if not self._active:
            return
        if len(self._points) >= self._max:
            return  # ป้องกัน memory หากนาน
        self._points.append(
            TrajectoryPoint(x=x, y=y, timestamp=time.monotonic())
        )

    def finish_cycle(self) -> list[TrajectoryPoint]:
        """
        หยุดบันทึก + normalise t_norm ให้อยู่ใน [0, 1] ตาม duration
        คืน list ของ TrajectoryPoint
        """
        self._active = False
        if not self._points:
            return []
        total = self._points[-1].timestamp - self._start
        if total > 0:
            for pt in self._points:
                pt.t_norm = (pt.timestamp - self._start) / total
        return list(self._points)

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def current_trajectory(self) -> list[TrajectoryPoint]:
        """ดึง trajectory ปัจจุบันโดยไม่หยุดบันทึก (ใช้ real-time preview)"""
        return list(self._points)


# ============================================================================
# PointTriggerDetector — orchestrator หลัก (pure Python, ไม่มี Qt)
# ============================================================================

@dataclass
class FrameResult:
    """ผลลัพธ์จากการ process 1 frame"""
    hand_detected:    bool
    hand_x:           float
    hand_y:           float
    triggered_points: list[int]                    # point_ids ที่ trigger ในเฟรมนี้
    state_changes:    list[PointUpdateResult]       # การเปลี่ยน state ทุกจุด
    point_states:     dict[int, PointState]         # สถานะปัจจุบันทุกจุด
    timestamp:        float


class PointTriggerDetector:
    """
    Orchestrator ที่รวม:
    - MediaPipe hand tracking (ผ่าน HandTracker)
    - State machine แต่ละจุด (ผ่าน PointStateMachine)
    - Trajectory recording (ผ่าน TrajectoryRecorder)

    ออกแบบเป็น pure Python (ไม่มี Qt) เพื่อ testability
    Qt signals อยู่ใน PointTrackerThread wrapper (ดู vision/point_tracker_thread.py)

    Callbacks (สำหรับส่งข้อมูลออก):
    ─────────────────────────────────
    on_trigger(point_id, timestamp, (x, y))  — เมื่อ trigger ยืนยัน
    on_state_change(point_id, new_state)     — เมื่อ state เปลี่ยน
    on_hand_position(x, y)                  — ทุกเฟรมที่มีมือ
    """

    # Palm centroid landmark indices (MediaPipe 21-point hand)
    # ใช้ centroid ฝ่ามือแทนแค่ wrist เพราะ stable กว่าตอนมือตั้ง/เอียง
    _PALM_LANDMARKS = [0, 5, 9, 13, 17]   # wrist + 4 MCP joints

    def __init__(
        self,
        trigger_points:   list[TriggerPoint],
        trigger_confirm:  int = 5,
        clear_confirm:    int = 8,
        use_palm_centroid: bool = True,
        on_trigger:       Optional[Callable[[int, float, tuple], None]] = None,
        on_state_change:  Optional[Callable[[int, PointState], None]]   = None,
        on_hand_position: Optional[Callable[[float, float], None]]      = None,
    ) -> None:
        self._points          = {p.point_id: p for p in trigger_points}
        self._trigger_confirm = trigger_confirm
        self._clear_confirm   = clear_confirm
        self._use_palm        = use_palm_centroid
        self._on_trigger      = on_trigger
        self._on_state_change = on_state_change
        self._on_hand_pos     = on_hand_position

        # สร้าง state machine ต่อจุด
        self._machines: dict[int, PointStateMachine] = {}
        for p in trigger_points:
            self._machines[p.point_id] = PointStateMachine(
                point           = p,
                trigger_confirm = trigger_confirm,
                clear_confirm   = clear_confirm,
                on_trigger_cb   = self._on_trigger,
                on_state_cb     = self._on_state_change,
            )

        self._trajectory = TrajectoryRecorder()
        self._hand_tracker = None   # lazy init เพื่อไม่โหลด MediaPipe จนกว่าจะใช้

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        เริ่มต้น detector — โหลด MediaPipe และ reset state ทุกจุด
        เรียกหนึ่งครั้งก่อนเริ่ม process_frame loop
        """
        if self._hand_tracker is None:
            from vision.hand_tracker import HandTracker
            self._hand_tracker = HandTracker()
        self.reset_all()

    def reset_all(self) -> None:
        """Reset state ทุกจุดกลับ WAITING_FOR_CLEAR พร้อมเริ่มรอบใหม่"""
        for machine in self._machines.values():
            machine.reset()

    def reset_cycle(self) -> None:
        """
        เรียกตอนเริ่ม production cycle ใหม่
        - Reset state ทุกจุด → WAITING_FOR_CLEAR
        - เริ่ม trajectory recording รอบใหม่
        """
        self.reset_all()
        self._trajectory.start_cycle()

    def close(self) -> None:
        """ปิด MediaPipe resources"""
        if self._hand_tracker:
            self._hand_tracker.close()
            self._hand_tracker = None

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process_frame(self, frame) -> FrameResult:
        """
        Process 1 เฟรม:
        1. Run MediaPipe hand detection
        2. คำนวณ hand reference position (wrist หรือ palm centroid)
        3. อัปเดต state machine ทุกจุด
        4. บันทึก trajectory
        5. เรียก callbacks

        Parameters
        ──────────
        frame : np.ndarray BGR (จาก OpenCV)

        Returns
        ───────
        FrameResult
        """
        if self._hand_tracker is None:
            self.start()   # auto-init ถ้ายังไม่ได้เรียก start()

        ts = time.monotonic()

        # ── Hand detection ──────────────────────────────────────────
        hand_result = self._hand_tracker.process(frame)

        hx, hy = 0.0, 0.0
        if hand_result.detected:
            if self._use_palm and hand_result.landmarks:
                hx, hy = self._palm_centroid(
                    hand_result.landmarks,
                    frame.shape[1],
                    frame.shape[0],
                )
            else:
                hx, hy = hand_result.x, hand_result.y

            # trajectory recording
            self._trajectory.record(hx, hy)

            # hand position callback
            if self._on_hand_pos:
                self._on_hand_pos(hx, hy)

        # ── อัปเดต state machine ทุกจุด ─────────────────────────────
        triggered_points: list[int]         = []
        state_changes:    list[PointUpdateResult] = []

        for pid, machine in self._machines.items():
            on_pt = (
                hand_result.detected
                and machine.point.is_on_point(hx, hy)
            )
            result = machine.update(
                on_point  = on_pt,
                hand_pos  = (hx, hy),
                timestamp = ts,
            )
            if result.triggered:
                triggered_points.append(pid)
            if result.state_changed:
                state_changes.append(result)

        return FrameResult(
            hand_detected    = hand_result.detected,
            hand_x           = hx,
            hand_y           = hy,
            triggered_points = triggered_points,
            state_changes    = state_changes,
            point_states     = {pid: m.state for pid, m in self._machines.items()},
            timestamp        = ts,
        )

    def _process_hand_position(
        self,
        x: float,
        y: float,
        detected: bool,
        timestamp: float | None = None,
    ) -> FrameResult:
        """
        ★ สำหรับ Unit Test เท่านั้น ★
        Bypass MediaPipe — inject hand position โดยตรง
        ทำให้ test state machine logic ได้โดยไม่ต้องมีกล้องหรือ MediaPipe

        เรียกใช้โดยตรงบน PointTriggerDetector object ใน test:
            detector._process_hand_position(x=100, y=100, detected=True)
        """
        ts = timestamp if timestamp is not None else time.monotonic()
        hx, hy = (x, y) if detected else (0.0, 0.0)

        if detected:
            self._trajectory.record(hx, hy)
            if self._on_hand_pos:
                self._on_hand_pos(hx, hy)

        triggered_points: list[int]               = []
        state_changes:    list[PointUpdateResult] = []

        for pid, machine in self._machines.items():
            on_pt = detected and machine.point.is_on_point(hx, hy)
            result = machine.update(
                on_point  = on_pt,
                hand_pos  = (hx, hy),
                timestamp = ts,
            )
            if result.triggered:
                triggered_points.append(pid)
            if result.state_changed:
                state_changes.append(result)

        return FrameResult(
            hand_detected    = detected,
            hand_x           = hx,
            hand_y           = hy,
            triggered_points = triggered_points,
            state_changes    = state_changes,
            point_states     = {pid: m.state for pid, m in self._machines.items()},
            timestamp        = ts,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_state(self, point_id: int) -> PointState:
        m = self._machines.get(point_id)
        return m.state if m else PointState.IDLE

    def get_all_states(self) -> dict[int, PointState]:
        return {pid: m.state for pid, m in self._machines.items()}

    def get_machine(self, point_id: int) -> Optional[PointStateMachine]:
        return self._machines.get(point_id)

    def get_trajectory(self) -> list[TrajectoryPoint]:
        """ดึง trajectory ที่กำลัง record อยู่ (ไม่หยุดบันทึก)"""
        return self._trajectory.current_trajectory

    def finish_trajectory(self) -> list[TrajectoryPoint]:
        """หยุดบันทึก + normalise แล้วคืน trajectory ของ cycle ล่าสุด"""
        return self._trajectory.finish_cycle()

    @property
    def trigger_points(self) -> list[TriggerPoint]:
        return list(self._points.values())

    # ------------------------------------------------------------------
    # Palm centroid helper
    # ------------------------------------------------------------------

    def _palm_centroid(
        self,
        landmarks: list,
        frame_w: int,
        frame_h: int,
    ) -> tuple[float, float]:
        """
        คำนวณ centroid ของ palm จาก landmark 0, 5, 9, 13, 17
        (wrist + 4 MCP joints) ในพิกัด pixel

        ใช้แทน wrist เพราะ stable กว่าตอนมือหงาย/คว่ำ
        """
        xs, ys = [], []
        for idx in self._PALM_LANDMARKS:
            if idx < len(landmarks):
                lm = landmarks[idx]
                xs.append(lm.x * frame_w)
                ys.append(lm.y * frame_h)
        if not xs:
            return 0.0, 0.0
        return sum(xs) / len(xs), sum(ys) / len(ys)
