"""
core/alert_manager.py
=====================
ระบบแจ้งเตือน real-time สำหรับ Live Monitor mode

ระบบที่ 1: Threshold Alert
──────────────────────────
ตั้งค่า threshold ต่อ zone แยกได้:
    WARNING  — เกิน standard X% (default 20%)
    CRITICAL — เกิน standard Y% (default 50%)

เรียก check_realtime() ทุกเฟรมขณะมืออยู่ใน zone
→ emit ทันทีเมื่อเกิน threshold (ไม่ต้องรอออก zone ก่อน)

ระบบที่ 2: Alert Counting & Summary
────────────────────────────────────
AlertCounter เก็บจำนวน alert ต่อ (zone_id, level) แบบ in-memory
เข้าถึงได้ทุกเวลาโดยไม่ต้อง query DB

Backward Compatibility
──────────────────────
ยังมี trigger_threshold() และ trigger_sequence_violation() เหมือนเดิม
เพื่อไม่ให้ vision_thread.py เดิมพัง

DB Persistence
──────────────
AlertManager รับ on_alert_with_db callback (cycle_id, AlertEvent) เพื่อ
ให้ vision_thread เรียก db.log_alert() โดยไม่ต้องนำ db เข้ามาใน module นี้
(รักษา separation of concerns)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional


# ============================================================================
# Enums
# ============================================================================

class AlertLevel(Enum):
    WARNING  = auto()   # สีเหลือง — ใกล้เกิน threshold
    CRITICAL = auto()   # สีแดง   — เกิน threshold มาก


class AlertType(Enum):
    THRESHOLD_EXCEEDED = auto()
    SEQUENCE_VIOLATION = auto()
    CYCLE_TIMEOUT      = auto()


# ============================================================================
# Data containers
# ============================================================================

@dataclass
class AlertEvent:
    """1 alert event"""

    alert_type:   AlertType
    alert_level:  AlertLevel
    zone_id:      int | None
    message:      str
    elapsed_sec:  float | None = None
    standard_sec: float | None = None
    over_pct:     float | None = None
    timestamp:    float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict:
        return {
            "type":        self.alert_type.name,
            "level":       self.alert_level.name,
            "zone_id":     self.zone_id,
            "message":     self.message,
            "elapsed_sec": self.elapsed_sec,
            "standard_sec": self.standard_sec,
            "over_pct":    self.over_pct,
            "timestamp":   self.timestamp,
        }

    def __str__(self) -> str:
        icon = "⚠" if self.alert_level == AlertLevel.WARNING else "🔴"
        zone = f" [Zone {self.zone_id}]" if self.zone_id is not None else ""
        pct  = f" (+{self.over_pct:.0f}%)" if self.over_pct is not None else ""
        return f"{icon}{zone}{pct}: {self.message}"


@dataclass
class ZoneThreshold:
    """
    Threshold config สำหรับ 1 zone

    Attributes
    ──────────
    warning_pct  : % เกิน standard ที่ trigger WARNING (0 = ปิด)
    critical_pct : % เกิน standard ที่ trigger CRITICAL (0 = ปิด)
    """
    warning_pct:  float = 20.0
    critical_pct: float = 50.0

    def level_for(self, over_pct: float) -> Optional[AlertLevel]:
        """
        คืน AlertLevel ที่เหมาะสมตาม over_pct
        คืน None ถ้ายังไม่เกิน threshold ไหนเลย
        """
        if self.critical_pct > 0 and over_pct >= self.critical_pct:
            return AlertLevel.CRITICAL
        if self.warning_pct > 0 and over_pct >= self.warning_pct:
            return AlertLevel.WARNING
        return None


class ThresholdConfig:
    """
    ตัวจัดการ per-zone threshold พร้อม global fallback

    Usage
    ─────
        cfg = ThresholdConfig(global_warning=20, global_critical=50)
        cfg.set_zone(zone_id=2, warning_pct=10, critical_pct=30)
        level = cfg.get_threshold(zone_id=2).level_for(over_pct=35)
    """

    def __init__(
        self,
        global_warning:  float = 20.0,
        global_critical: float = 50.0,
    ) -> None:
        self._global = ZoneThreshold(
            warning_pct  = global_warning,
            critical_pct = global_critical,
        )
        self._per_zone: dict[int, ZoneThreshold] = {}

    def set_zone(
        self,
        zone_id:      int,
        warning_pct:  float,
        critical_pct: float,
    ) -> None:
        """ตั้ง threshold เฉพาะ zone นั้น (override global)"""
        self._per_zone[zone_id] = ZoneThreshold(
            warning_pct  = warning_pct,
            critical_pct = critical_pct,
        )

    def set_global(self, warning_pct: float, critical_pct: float) -> None:
        self._global = ZoneThreshold(
            warning_pct  = warning_pct,
            critical_pct = critical_pct,
        )

    def get_threshold(self, zone_id: int) -> ZoneThreshold:
        """คืน threshold ของ zone นั้น (per-zone ถ้ามี ไม่ก็ global)"""
        return self._per_zone.get(zone_id, self._global)

    def remove_zone(self, zone_id: int) -> None:
        self._per_zone.pop(zone_id, None)

    @property
    def global_threshold(self) -> ZoneThreshold:
        return self._global


@dataclass
class AlertCounter:
    """
    นับจำนวน alert แบบ in-memory (ไม่ต้อง query DB)
    เก็บ per-zone per-level
    """
    _counts: dict[tuple[int | None, str], int] = field(default_factory=dict)
    _total:  int = 0

    def increment(self, zone_id: int | None, level: AlertLevel) -> None:
        key = (zone_id, level.name)
        self._counts[key] = self._counts.get(key, 0) + 1
        self._total += 1

    def count(self, zone_id: int | None = None, level: AlertLevel | None = None) -> int:
        """
        นับ alert ที่ตรงกับ zone_id และ/หรือ level
        ถ้าไม่ระบุ → คืน total
        """
        if zone_id is None and level is None:
            return self._total
        result = 0
        for (z, lv), cnt in self._counts.items():
            if zone_id is not None and z != zone_id:
                continue
            if level is not None and lv != level.name:
                continue
            result += cnt
        return result

    def reset(self) -> None:
        self._counts.clear()
        self._total = 0

    def summary(self) -> dict:
        """คืน {(zone_id, level_name): count}"""
        return dict(self._counts)

    @property
    def total(self) -> int:
        return self._total


# ============================================================================
# AlertManager
# ============================================================================

class AlertManager:
    """
    ระบบแจ้งเตือน real-time สำหรับ Production Monitor

    Parameters
    ──────────
    on_alert         : callback(AlertEvent) — consumer หลัก (vision thread relay ไป GUI)
    on_alert_with_db : callback(AlertEvent, cycle_id) — สำหรับ DB persistence
                       (เรียกพร้อมกับ on_alert ถ้ามี)
    threshold_config : ThresholdConfig object
                       (ถ้าไม่ระบุ จะสร้าง default จาก warning_pct/critical_pct)
    cooldown_sec     : เวลา minimum ระหว่าง alert ของ (type, zone) คู่เดิม
    audio_enabled    : เล่น beep ทุกครั้งที่ CRITICAL
    warning_pct      : global warning % (ใช้ถ้าไม่ระบุ threshold_config)
    critical_pct     : global critical % (ใช้ถ้าไม่ระบุ threshold_config)

    Real-time Usage
    ───────────────
        # เรียกทุกเฟรมขณะ monitor
        alert_mgr.check_realtime(
            zone_id=1,
            elapsed_sec=4.5,
            standard_sec=3.0,
            cycle_id=current_cycle_id,
        )
    """

    def __init__(
        self,
        on_alert:           Callable[[AlertEvent], None] | None = None,
        on_alert_with_db:   Callable[[AlertEvent, int | None], None] | None = None,
        threshold_config:   ThresholdConfig | None = None,
        cooldown_sec:       float = 3.0,
        audio_enabled:      bool  = True,
        # legacy / convenience params → สร้าง ThresholdConfig อัตโนมัติ
        warning_pct:        float = 20.0,
        critical_pct:       float = 50.0,
    ) -> None:
        self._on_alert         = on_alert
        self._on_alert_with_db = on_alert_with_db
        self._cooldown         = cooldown_sec
        self._audio_enabled    = audio_enabled

        # Threshold config (per-zone หรือ global)
        self._thresholds = threshold_config or ThresholdConfig(
            global_warning  = warning_pct,
            global_critical = critical_pct,
        )

        # Deduplication: {(alert_type, zone_id, level): last_fire_time}
        self._last_fired: dict[tuple, float]  = {}

        # In-memory history
        self._alert_history: list[AlertEvent] = []
        self._counter = AlertCounter()

        # Current cycle_id สำหรับ DB logging
        self._current_cycle_id: int | None = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_current_cycle(self, cycle_id: int | None) -> None:
        """อัปเดต cycle_id ปัจจุบัน — เรียกทุกครั้งที่ cycle ใหม่เริ่ม"""
        self._current_cycle_id = cycle_id

    def set_callback(self, cb: Callable[[AlertEvent], None]) -> None:
        self._on_alert = cb

    def set_db_callback(
        self,
        cb: Callable[[AlertEvent, int | None], None],
    ) -> None:
        """
        ตั้ง callback สำหรับ DB persistence
        signature: cb(event: AlertEvent, cycle_id: int | None)
        """
        self._on_alert_with_db = cb

    def set_thresholds(self, warning_pct: float, critical_pct: float) -> None:
        """ตั้ง global threshold (ไม่กระทบ per-zone ที่ตั้งไว้แล้ว)"""
        self._thresholds.set_global(warning_pct, critical_pct)

    def set_zone_threshold(
        self,
        zone_id:      int,
        warning_pct:  float,
        critical_pct: float,
    ) -> None:
        """ตั้ง threshold เฉพาะ zone — override global สำหรับ zone นั้น"""
        self._thresholds.set_zone(zone_id, warning_pct, critical_pct)

    def set_audio(self, enabled: bool) -> None:
        self._audio_enabled = enabled

    # ------------------------------------------------------------------
    # Real-time threshold checking (เรียกทุกเฟรม)
    # ------------------------------------------------------------------

    def check_realtime(
        self,
        zone_id:      int,
        elapsed_sec:  float,
        standard_sec: float,
        cycle_id:     int | None = None,
    ) -> Optional[AlertEvent]:
        """
        ตรวจสอบ threshold แบบ real-time ทุกเฟรม

        เรียกขณะมืออยู่ใน zone ที่ยังไม่ออก
        emit alert ทันทีเมื่อ elapsed เกิน threshold
        (ไม่ต้องรอออกจาก zone ก่อน)

        Parameters
        ──────────
        zone_id      : zone ที่กำลังตรวจ
        elapsed_sec  : เวลาที่ผ่านไปใน zone นี้แล้ว (วินาที)
        standard_sec : เวลามาตรฐานของ zone นี้ (วินาที)
        cycle_id     : cycle ปัจจุบัน (ถ้าไม่ระบุ ใช้ current_cycle_id)

        Returns
        ───────
        AlertEvent ถ้า emit, None ถ้าไม่เกิน threshold หรืออยู่ใน cooldown
        """
        if standard_sec <= 0:
            return None

        over_pct = (elapsed_sec - standard_sec) / standard_sec * 100.0
        thr      = self._thresholds.get_threshold(zone_id)
        level    = thr.level_for(over_pct)

        if level is None:
            return None

        msg = (
            f"Zone {zone_id}: {elapsed_sec:.1f}s "
            f"vs standard {standard_sec:.1f}s "
            f"(+{over_pct:.0f}% — {level.name})"
        )

        event = AlertEvent(
            alert_type   = AlertType.THRESHOLD_EXCEEDED,
            alert_level  = level,
            zone_id      = zone_id,
            message      = msg,
            elapsed_sec  = elapsed_sec,
            standard_sec = standard_sec,
            over_pct     = over_pct,
        )

        cid = cycle_id if cycle_id is not None else self._current_cycle_id
        return self._dispatch(event, cid)

    # ------------------------------------------------------------------
    # Named trigger methods (backward compat + sequence violations)
    # ------------------------------------------------------------------

    def trigger_threshold(
        self,
        zone_id:      int,
        elapsed_sec:  float,
        standard_sec: float,
        cycle_id:     int | None = None,
    ) -> Optional[AlertEvent]:
        """Backward-compat wrapper สำหรับ vision_thread.py เดิม"""
        return self.check_realtime(zone_id, elapsed_sec, standard_sec, cycle_id)

    def trigger_sequence_violation(
        self,
        expected_zone: int,
        actual_zone:   int,
        cycle_id:      int | None = None,
        detail:        str = "",
    ) -> AlertEvent:
        """
        Emit sequence violation alert

        Parameters
        ──────────
        expected_zone : zone ที่คาดไว้
        actual_zone   : zone ที่เข้าจริง
        detail        : คำอธิบายเพิ่มเติม (เช่น SKIP, OUT_OF_ORDER)
        """
        msg = (
            f"Sequence error — expected Zone {expected_zone}, "
            f"entered Zone {actual_zone}"
        )
        if detail:
            msg = f"{detail} | {msg}"

        event = AlertEvent(
            alert_type  = AlertType.SEQUENCE_VIOLATION,
            alert_level = AlertLevel.CRITICAL,
            zone_id     = actual_zone,
            message     = msg,
        )
        cid = cycle_id if cycle_id is not None else self._current_cycle_id
        result = self._dispatch(event, cid, force=True)  # sequence errors ไม่ cooldown
        return result or event

    def trigger_cycle_timeout(
        self,
        elapsed_sec:  float,
        standard_sec: float,
        cycle_id:     int | None = None,
    ) -> AlertEvent:
        msg = (
            f"Cycle timeout: {elapsed_sec:.0f}s elapsed "
            f"vs standard {standard_sec:.0f}s"
        )
        event = AlertEvent(
            alert_type   = AlertType.CYCLE_TIMEOUT,
            alert_level  = AlertLevel.CRITICAL,
            zone_id      = None,
            message      = msg,
            elapsed_sec  = elapsed_sec,
            standard_sec = standard_sec,
        )
        cid = cycle_id if cycle_id is not None else self._current_cycle_id
        result = self._dispatch(event, cid)
        return result or event

    # ------------------------------------------------------------------
    # Internal dispatch with deduplication
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        event:    AlertEvent,
        cycle_id: int | None = None,
        force:    bool = False,
    ) -> Optional[AlertEvent]:
        """
        Dedup + emit alert

        force=True ข้ามการ cooldown (ใช้กับ sequence violations)
        """
        key  = (event.alert_type, event.zone_id, event.alert_level)
        now  = time.monotonic()
        last = self._last_fired.get(key, 0.0)

        if not force and (now - last) < self._cooldown:
            return None   # cooldown active — ไม่ emit ซ้ำ

        self._last_fired[key] = now
        self._alert_history.append(event)
        self._counter.increment(event.zone_id, event.alert_level)

        # Audio beep เฉพาะ CRITICAL
        if self._audio_enabled and event.alert_level == AlertLevel.CRITICAL:
            self._beep()

        # Primary callback (→ GUI via Qt signal)
        if self._on_alert:
            self._on_alert(event)

        # DB persistence callback
        if self._on_alert_with_db:
            self._on_alert_with_db(event, cycle_id)

        return event

    @staticmethod
    def _beep() -> None:
        try:
            import sys
            sys.stdout.write("\a")
            sys.stdout.flush()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def counter(self) -> AlertCounter:
        """AlertCounter สำหรับ query จำนวน alert แบบ real-time"""
        return self._counter

    @property
    def alert_history(self) -> list[AlertEvent]:
        return list(self._alert_history)

    @property
    def threshold_config(self) -> ThresholdConfig:
        return self._thresholds

    def recent_alerts(self, n: int = 10) -> list[AlertEvent]:
        return self._alert_history[-n:]

    def clear_history(self) -> None:
        self._alert_history.clear()
        self._last_fired.clear()
        self._counter.reset()

    def reset_for_session(self) -> None:
        """เรียกตอนเริ่ม session ใหม่"""
        self.clear_history()
        self._current_cycle_id = None
