"""
data/models.py
SQLAlchemy ORM models for the Production Cycle Monitor.

Tables
------
- Session              : one recording session (start → stop)
- CycleLog             : one completed work cycle within a session
- ZoneEvent            : individual zone entry/exit event within a cycle
- GoldenCycle          : stored golden-cycle standard per station
- AlertLog             : threshold alert events (WARNING / CRITICAL) per cycle
- SequenceViolationLog : sequence violation events (SKIP / OUT_OF_ORDER / REPEAT)
"""

from __future__ import annotations

import datetime
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, Text, JSON
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    __allow_unmapped__ = True


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class Session(Base):
    """Top-level container for a monitoring session."""

    __tablename__ = "sessions"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    station_id    = Column(String(64), nullable=False, default="station_01")
    operator_name = Column(String(128), nullable=True)
    started_at    = Column(DateTime, default=datetime.datetime.utcnow)
    ended_at      = Column(DateTime, nullable=True)
    total_cycles  = Column(Integer, default=0)
    pass_cycles   = Column(Integer, default=0)
    fail_cycles   = Column(Integer, default=0)
    notes         = Column(Text, nullable=True)

    cycles: list[CycleLog] = relationship(
        "CycleLog", back_populates="session",
        cascade="all, delete-orphan"
    )

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None

    def __repr__(self) -> str:
        return f"<Session id={self.id} station={self.station_id}>"


# ---------------------------------------------------------------------------
# CycleLog
# ---------------------------------------------------------------------------

class CycleLog(Base):
    """Records one complete work cycle (Zone1 → Zone2 → Zone3 → store)."""

    __tablename__ = "cycle_logs"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    session_id       = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    cycle_number     = Column(Integer, nullable=False)
    started_at       = Column(DateTime, nullable=False)
    ended_at         = Column(DateTime, nullable=True)
    cycle_time_sec   = Column(Float, nullable=True)    # total duration
    standard_time_sec = Column(Float, nullable=True)   # golden-cycle median
    deviation_pct    = Column(Float, nullable=True)    # % over/under standard
    status           = Column(String(32), default="in_progress")
    # status values: in_progress | pass | fail | sequence_error | timeout
    sequence_errors  = Column(JSON, default=list)      # list of error descriptions
    zone_times       = Column(JSON, default=dict)      # {zone_id: elapsed_sec}
    dtw_score        = Column(Float, nullable=True)    # similarity vs golden

    session: Session = relationship("Session", back_populates="cycles")
    zone_events: list[ZoneEvent] = relationship(
        "ZoneEvent", back_populates="cycle",
        cascade="all, delete-orphan",
        order_by="ZoneEvent.timestamp"
    )

    def __repr__(self) -> str:
        return (
            f"<CycleLog id={self.id} cycle={self.cycle_number} "
            f"status={self.status} time={self.cycle_time_sec:.2f}s>"
        )


# ---------------------------------------------------------------------------
# ZoneEvent
# ---------------------------------------------------------------------------

class ZoneEvent(Base):
    """One zone entry or exit event captured during a cycle."""

    __tablename__ = "zone_events"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    cycle_id    = Column(Integer, ForeignKey("cycle_logs.id"), nullable=False)
    zone_id     = Column(Integer, nullable=False)
    zone_name   = Column(String(64), nullable=True)
    event_type  = Column(String(8), nullable=False)  # "enter" | "exit"
    timestamp   = Column(DateTime, nullable=False)
    hand_x      = Column(Float, nullable=True)       # normalized [0,1]
    hand_y      = Column(Float, nullable=True)

    cycle: CycleLog = relationship("CycleLog", back_populates="zone_events")

    def __repr__(self) -> str:
        return (
            f"<ZoneEvent zone={self.zone_id} {self.event_type} "
            f"@ {self.timestamp.isoformat()}>"
        )


# ---------------------------------------------------------------------------
# GoldenCycle
# ---------------------------------------------------------------------------

class GoldenCycle(Base):
    """Persisted golden-cycle standard for a station."""

    __tablename__ = "golden_cycles"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    station_id          = Column(String(64), nullable=False, unique=True)
    recorded_at         = Column(DateTime, default=datetime.datetime.utcnow)
    num_source_cycles   = Column(Integer, nullable=False)
    standard_total_sec  = Column(Float, nullable=False)
    zone_standard_times = Column(JSON, nullable=False)   # {zone_id: median_sec}
    trajectory_points   = Column(JSON, default=list)     # [{x,y,zone,t_norm}]
    raw_cycle_times     = Column(JSON, default=list)     # list of total times

    def __repr__(self) -> str:
        return (
            f"<GoldenCycle station={self.station_id} "
            f"total={self.standard_total_sec:.2f}s "
            f"from {self.num_source_cycles} cycles>"
        )


# ---------------------------------------------------------------------------
# AlertLog
# ---------------------------------------------------------------------------

class AlertLog(Base):
    """
    บันทึก threshold alert event ทุกครั้งที่เกิดขึ้นระหว่าง cycle

    Columns
    -------
    cycle_id      : FK → cycle_logs.id  (None ถ้า alert เกิดนอก cycle)
    zone_id       : zone ที่ trigger alert (None สำหรับ cycle-level alert)
    alert_type    : "THRESHOLD_EXCEEDED" | "CYCLE_TIMEOUT"
    alert_level   : "WARNING" | "CRITICAL"
    elapsed_sec   : เวลาที่ใช้ไปใน zone ณ จุดที่ alert
    standard_sec  : standard time ของ zone นั้น
    over_pct      : (elapsed - standard) / standard × 100
    message       : human-readable message
    occurred_at   : UTC timestamp ที่ alert เกิด
    """

    __tablename__ = "alert_logs"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    cycle_id     = Column(Integer, ForeignKey("cycle_logs.id"), nullable=True)
    zone_id      = Column(Integer, nullable=True)
    alert_type   = Column(String(32), nullable=False, default="THRESHOLD_EXCEEDED")
    alert_level  = Column(String(16), nullable=False)   # "WARNING" | "CRITICAL"
    elapsed_sec  = Column(Float, nullable=True)
    standard_sec = Column(Float, nullable=True)
    over_pct     = Column(Float, nullable=True)
    message      = Column(Text, nullable=False)
    occurred_at  = Column(DateTime, default=datetime.datetime.utcnow)

    cycle: "CycleLog | None" = relationship(
        "CycleLog",
        foreign_keys=[cycle_id],
        backref="alert_logs",
    )

    def __repr__(self) -> str:
        return (
            f"<AlertLog [{self.alert_level}] zone={self.zone_id} "
            f"+{self.over_pct:.1f}% @ {self.occurred_at}>"
        )


# ---------------------------------------------------------------------------
# SequenceViolationLog
# ---------------------------------------------------------------------------

class SequenceViolationLog(Base):
    """
    บันทึก sequence violation event ทุกครั้งที่ตรวจพบ

    Columns
    -------
    cycle_id         : FK → cycle_logs.id
    violation_type   : "SKIP" | "OUT_OF_ORDER" | "REPEAT"
    expected_zone    : zone ที่คาดว่าจะเข้าตามลำดับมาตรฐาน
    actual_zone      : zone ที่เข้าจริง
    skipped_zones    : JSON list ของ zone ที่ถูก skip (เฉพาะ SKIP type)
    sequence_so_far  : JSON list ของ zone ที่เข้ามาแล้วก่อน violation นี้
    message          : human-readable description
    occurred_at      : UTC timestamp
    """

    __tablename__ = "sequence_violation_logs"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    cycle_id         = Column(Integer, ForeignKey("cycle_logs.id"), nullable=True)
    violation_type   = Column(String(16), nullable=False)   # SKIP|OUT_OF_ORDER|REPEAT
    expected_zone    = Column(Integer, nullable=True)
    actual_zone      = Column(Integer, nullable=False)
    skipped_zones    = Column(JSON, default=list)
    sequence_so_far  = Column(JSON, default=list)
    message          = Column(Text, nullable=False)
    occurred_at      = Column(DateTime, default=datetime.datetime.utcnow)

    cycle: "CycleLog | None" = relationship(
        "CycleLog",
        foreign_keys=[cycle_id],
        backref="sequence_violations",
    )

    def __repr__(self) -> str:
        return (
            f"<SequenceViolationLog [{self.violation_type}] "
            f"expected={self.expected_zone} actual={self.actual_zone}>"
        )
