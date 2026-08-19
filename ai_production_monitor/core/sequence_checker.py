"""
core/sequence_checker.py
========================
ตรวจจับการละเมิดลำดับการทำงานใน Production Cycle แบบ real-time

ประเภทของ Violation
───────────────────
SKIP
    ข้ามขั้นตอน — เข้า zone ที่สูงกว่า index ถัดไปในลำดับ
    ตัวอย่าง: ลำดับมาตรฐาน [1, 2, 3] เข้า 1 แล้วข้ามไป 3 = skip zone 2

OUT_OF_ORDER
    ทำสลับลำดับ — เข้า zone ที่ควรผ่านไปแล้วในลำดับ
    ตัวอย่าง: เข้า 1→2→3 แล้วกลับไปแตะ 2 อีก (ไม่ใช่ repeat ตรงๆ)
    หรือเริ่มด้วย zone ที่ไม่ใช่ zone แรก (เช่น เริ่มที่ 2)

REPEAT
    ทำ zone เดิมซ้ำติดต่อกัน — เข้า zone ที่เพิ่งออกมาโดยไม่ผ่าน zone อื่น
    ตัวอย่าง: เข้า 1→ออก 1→เข้า 1 ซ้ำ (โดยไม่ผ่าน 2 ก่อน)

State Machine
─────────────
SequenceChecker เก็บ state:
    expected_idx  — index ใน expected_sequence ที่รอ zone ถัดไป
    visited       — list ของ zone_id ที่เข้ามาแล้วใน cycle นี้ (ตามลำดับ)
    last_zone     — zone_id สุดท้ายที่เข้า

เรียก observe_enter(zone_id) ทุกครั้งที่มีการเข้า zone
คืน SequenceViolation ถ้าผิด หรือ None ถ้าถูกต้อง

เรียก reset() เมื่อ cycle ใหม่เริ่มต้น
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional


# ============================================================================
# Enums / Data containers
# ============================================================================

class ViolationType(Enum):
    SKIP          = auto()   # ข้ามขั้นตอน
    OUT_OF_ORDER  = auto()   # ทำสลับลำดับ / เริ่มผิด zone
    REPEAT        = auto()   # ทำซ้ำ zone เดิมโดยไม่ผ่าน zone ถัดไปก่อน


@dataclass
class SequenceViolation:
    """ข้อมูล violation 1 เหตุการณ์"""

    violation_type:  ViolationType
    actual_zone:     int             # zone ที่เข้าจริง
    expected_zone:   int | None      # zone ที่คาดว่าจะเข้า (None สำหรับ REPEAT)
    skipped_zones:   list[int]       # zone ที่ถูก skip ไป (เฉพาะ SKIP type)
    sequence_so_far: list[int]       # ลำดับ zone ที่เข้ามาแล้วก่อน violation นี้
    message:         str
    timestamp:       float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict:
        return {
            "violation_type":  self.violation_type.name,
            "actual_zone":     self.actual_zone,
            "expected_zone":   self.expected_zone,
            "skipped_zones":   self.skipped_zones,
            "sequence_so_far": self.sequence_so_far,
            "message":         self.message,
            "timestamp":       self.timestamp,
        }

    def __str__(self) -> str:
        return f"[{self.violation_type.name}] {self.message}"


# ============================================================================
# SequenceChecker
# ============================================================================

class SequenceChecker:
    """
    ตรวจจับ sequence violation แบบ real-time

    Parameters
    ──────────
    expected_sequence : ลำดับ zone มาตรฐาน เช่น [1, 2, 3]
    allow_repeat      : ถ้า True จะไม่ detect REPEAT violation
                        (ใช้เมื่อบางสถานีต้องทำ zone ซ้ำได้)
    on_violation      : callback(SequenceViolation) เรียกทุกครั้งที่ตรวจพบ

    Usage
    ─────
        checker = SequenceChecker([1, 2, 3], on_violation=my_cb)
        checker.reset()

        # ทุกครั้งที่ CycleTracker emit zone enter:
        v = checker.observe_enter(zone_id)
        if v:
            print(v)

        # เมื่อ cycle ใหม่เริ่ม:
        checker.reset()
    """

    def __init__(
        self,
        expected_sequence: list[int],
        allow_repeat:      bool = False,
        on_violation:      Callable[[SequenceViolation], None] | None = None,
    ) -> None:
        if not expected_sequence:
            raise ValueError("expected_sequence ต้องมีอย่างน้อย 1 zone")

        self._sequence     = list(expected_sequence)
        self._allow_repeat = allow_repeat
        self._on_violation = on_violation

        # Runtime state — reset ด้วย reset()
        self._expected_idx: int       = 0
        self._visited:      list[int] = []
        self._last_zone:    int | None = None
        self._violations:   list[SequenceViolation] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """เริ่ม cycle ใหม่ — ล้าง state ทั้งหมด"""
        self._expected_idx = 0
        self._visited      = []
        self._last_zone    = None

    def reset_violations(self) -> None:
        """ล้าง history ของ violations (ใช้ตอนเริ่ม session ใหม่)"""
        self._violations.clear()

    def observe_enter(self, zone_id: int) -> Optional[SequenceViolation]:
        """
        แจ้งว่ามือเพิ่งเข้า zone_id

        Returns
        ───────
        SequenceViolation ถ้าผิดลำดับ, None ถ้าถูก
        """
        violation: Optional[SequenceViolation] = None
        seq_snapshot = list(self._visited)  # snapshot ก่อน append

        # ── ตรวจ REPEAT ─────────────────────────────────────────────
        # เข้า zone เดิมซ้ำทันทีโดยไม่ผ่าน zone อื่น
        if (
            not self._allow_repeat
            and self._last_zone is not None
            and zone_id == self._last_zone
        ):
            violation = SequenceViolation(
                violation_type  = ViolationType.REPEAT,
                actual_zone     = zone_id,
                expected_zone   = self._next_expected(),
                skipped_zones   = [],
                sequence_so_far = seq_snapshot,
                message         = (
                    f"REPEAT: เข้า Zone {zone_id} ซ้ำโดยไม่ผ่าน zone ถัดไปก่อน "
                    f"(ลำดับที่ถูก: {self._sequence})"
                ),
            )

        # ── ตรวจ SKIP / OUT_OF_ORDER (ถ้ายังอยู่ใน sequence) ────────
        elif self._expected_idx < len(self._sequence):
            expected = self._sequence[self._expected_idx]

            if zone_id == expected:
                # ถูกต้อง — advance pointer
                pass

            elif zone_id in self._sequence:
                zone_pos = self._sequence.index(zone_id)

                if zone_pos > self._expected_idx:
                    # SKIP — กระโดดข้ามไปข้างหน้า
                    skipped = self._sequence[self._expected_idx:zone_pos]
                    violation = SequenceViolation(
                        violation_type  = ViolationType.SKIP,
                        actual_zone     = zone_id,
                        expected_zone   = expected,
                        skipped_zones   = skipped,
                        sequence_so_far = seq_snapshot,
                        message         = (
                            f"SKIP: คาดว่าจะเข้า Zone {expected} "
                            f"แต่เข้า Zone {zone_id} "
                            f"(ข้าม Zone {skipped})"
                        ),
                    )
                else:
                    # OUT_OF_ORDER — ย้อนกลับไป zone ที่ผ่านมาแล้ว
                    violation = SequenceViolation(
                        violation_type  = ViolationType.OUT_OF_ORDER,
                        actual_zone     = zone_id,
                        expected_zone   = expected,
                        skipped_zones   = [],
                        sequence_so_far = seq_snapshot,
                        message         = (
                            f"OUT_OF_ORDER: คาดว่าจะเข้า Zone {expected} "
                            f"แต่ย้อนกลับไป Zone {zone_id} "
                            f"ที่ผ่านไปแล้ว"
                        ),
                    )

            else:
                # Zone ที่ไม่อยู่ใน sequence เลย → OUT_OF_ORDER
                violation = SequenceViolation(
                    violation_type  = ViolationType.OUT_OF_ORDER,
                    actual_zone     = zone_id,
                    expected_zone   = expected,
                    skipped_zones   = [],
                    sequence_so_far = seq_snapshot,
                    message         = (
                        f"OUT_OF_ORDER: Zone {zone_id} "
                        f"ไม่อยู่ในลำดับมาตรฐาน {self._sequence}"
                    ),
                )

        else:
            # เลยจุดสุดท้ายของ sequence แล้ว — zone เพิ่มเติมไม่คาดไว้
            violation = SequenceViolation(
                violation_type  = ViolationType.OUT_OF_ORDER,
                actual_zone     = zone_id,
                expected_zone   = None,
                skipped_zones   = [],
                sequence_so_far = seq_snapshot,
                message         = (
                    f"OUT_OF_ORDER: Cycle ควรจบแล้ว "
                    f"แต่เข้า Zone {zone_id} เพิ่มเติม"
                ),
            )

        # ── อัปเดต state ─────────────────────────────────────────────
        self._visited.append(zone_id)
        self._last_zone = zone_id

        # advance expected pointer ถ้า zone นี้ตรงกับที่คาด (ไม่ว่าจะ violation หรือเปล่า)
        if self._expected_idx < len(self._sequence):
            if zone_id == self._sequence[self._expected_idx]:
                self._expected_idx += 1
            elif violation and violation.violation_type == ViolationType.SKIP:
                # SKIP: advance ไปยัง index ถัดจาก zone ที่ skip ถึง
                zone_pos = self._sequence.index(zone_id)
                self._expected_idx = zone_pos + 1

        # ── dispatch ─────────────────────────────────────────────────
        if violation:
            self._violations.append(violation)
            if self._on_violation:
                self._on_violation(violation)

        return violation

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def expected_sequence(self) -> list[int]:
        return list(self._sequence)

    @property
    def visited_zones(self) -> list[int]:
        """ลำดับ zone ที่เข้ามาแล้วใน cycle ปัจจุบัน"""
        return list(self._visited)

    @property
    def next_expected_zone(self) -> int | None:
        """zone ถัดไปที่คาดว่าจะเข้า (None ถ้า cycle ควรจบแล้ว)"""
        return self._next_expected()

    @property
    def is_complete(self) -> bool:
        """True ถ้าเข้าครบทุก zone ตามลำดับแล้ว"""
        return self._expected_idx >= len(self._sequence)

    @property
    def progress_index(self) -> int:
        """index ของ zone ถัดไปใน expected_sequence (0-based)"""
        return self._expected_idx

    @property
    def violation_count(self) -> int:
        return len(self._violations)

    @property
    def all_violations(self) -> list[SequenceViolation]:
        """violations ทั้งหมดที่เก็บไว้ (ทุก cycle จนกว่าจะ reset_violations)"""
        return list(self._violations)

    @property
    def cycle_violations(self) -> list[SequenceViolation]:
        """violations ของ cycle ปัจจุบัน (นับจาก visited เริ่มต้น)"""
        # violations ที่ sequence_so_far มีความยาวเท่ากับ visited ในช่วงปัจจุบัน
        # ใช้วิธีง่าย: นับจาก all_violations ที่ sequence_so_far subset ของ visited
        return [
            v for v in self._violations
            if self._is_current_cycle(v)
        ]

    def set_callback(self, cb: Callable[[SequenceViolation], None]) -> None:
        self._on_violation = cb

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_expected(self) -> int | None:
        if self._expected_idx < len(self._sequence):
            return self._sequence[self._expected_idx]
        return None

    def _is_current_cycle(self, v: SequenceViolation) -> bool:
        """ตรวจว่า violation นี้เกิดใน cycle ปัจจุบัน"""
        # ถ้า sequence_so_far เป็น prefix ของ visited ปัจจุบัน
        visited = self._visited
        sf      = v.sequence_so_far
        if len(sf) > len(visited):
            return False
        return visited[:len(sf)] == sf
