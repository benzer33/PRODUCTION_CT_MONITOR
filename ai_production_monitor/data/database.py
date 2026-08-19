"""
data/database.py
Database connection management and CRUD helpers.

Usage
-----
    from data.database import DatabaseManager
    db = DatabaseManager("logs/production.db")
    session_id = db.start_session("station_01", operator_name="Alice")
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session as OrmSession, sessionmaker

from data.models import (
    AlertLog, Base, CycleLog, GoldenCycle,
    SequenceViolationLog, Session, ZoneEvent,
)


class DatabaseManager:
    """Thread-safe (per-call session) SQLite helper via SQLAlchemy."""

    def __init__(self, db_path: str = "logs/production.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            echo=False,
        )
        Base.metadata.create_all(self._engine)
        self._SessionFactory = sessionmaker(bind=self._engine, expire_on_commit=False)

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------

    def _session(self) -> OrmSession:
        return self._SessionFactory()

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    def start_session(
        self,
        station_id: str = "station_01",
        operator_name: str | None = None,
    ) -> int:
        with self._session() as s:
            sess = Session(
                station_id=station_id,
                operator_name=operator_name,
                started_at=datetime.datetime.utcnow(),
            )
            s.add(sess)
            s.commit()
            return sess.id

    def end_session(self, session_id: int) -> None:
        with self._session() as s:
            sess = s.get(Session, session_id)
            if sess:
                sess.ended_at = datetime.datetime.utcnow()
                cycles = s.execute(
                    select(CycleLog).where(CycleLog.session_id == session_id)
                ).scalars().all()
                sess.total_cycles = len(cycles)
                sess.pass_cycles  = sum(1 for c in cycles if c.status == "pass")
                sess.fail_cycles  = sum(1 for c in cycles if c.status != "pass"
                                        and c.status != "in_progress")
                s.commit()

    def get_session(self, session_id: int) -> Session | None:
        with self._session() as s:
            return s.get(Session, session_id)

    # ------------------------------------------------------------------
    # Cycle CRUD
    # ------------------------------------------------------------------

    def start_cycle(self, session_id: int, cycle_number: int) -> int:
        with self._session() as s:
            cycle = CycleLog(
                session_id=session_id,
                cycle_number=cycle_number,
                started_at=datetime.datetime.utcnow(),
                status="in_progress",
            )
            s.add(cycle)
            s.commit()
            return cycle.id

    def complete_cycle(
        self,
        cycle_id: int,
        status: str,
        cycle_time_sec: float,
        standard_time_sec: float | None = None,
        deviation_pct: float | None = None,
        zone_times: dict | None = None,
        sequence_errors: list | None = None,
        dtw_score: float | None = None,
    ) -> None:
        with self._session() as s:
            cycle = s.get(CycleLog, cycle_id)
            if cycle:
                cycle.ended_at         = datetime.datetime.utcnow()
                cycle.cycle_time_sec   = cycle_time_sec
                cycle.standard_time_sec = standard_time_sec
                cycle.deviation_pct    = deviation_pct
                cycle.zone_times       = zone_times or {}
                cycle.sequence_errors  = sequence_errors or []
                cycle.dtw_score        = dtw_score
                cycle.status           = status
                s.commit()

    def get_session_cycles(self, session_id: int) -> list[CycleLog]:
        with self._session() as s:
            return s.execute(
                select(CycleLog)
                .where(CycleLog.session_id == session_id)
                .order_by(CycleLog.cycle_number)
            ).scalars().all()

    # ------------------------------------------------------------------
    # Zone Event CRUD
    # ------------------------------------------------------------------

    def log_zone_event(
        self,
        cycle_id: int,
        zone_id: int,
        zone_name: str,
        event_type: str,   # "enter" | "exit"
        hand_x: float | None = None,
        hand_y: float | None = None,
    ) -> int:
        with self._session() as s:
            evt = ZoneEvent(
                cycle_id=cycle_id,
                zone_id=zone_id,
                zone_name=zone_name,
                event_type=event_type,
                timestamp=datetime.datetime.utcnow(),
                hand_x=hand_x,
                hand_y=hand_y,
            )
            s.add(evt)
            s.commit()
            return evt.id

    def get_cycle_events(self, cycle_id: int) -> list[ZoneEvent]:
        with self._session() as s:
            return s.execute(
                select(ZoneEvent)
                .where(ZoneEvent.cycle_id == cycle_id)
                .order_by(ZoneEvent.timestamp)
            ).scalars().all()

    # ------------------------------------------------------------------
    # Golden Cycle CRUD
    # ------------------------------------------------------------------

    def save_golden_cycle(
        self,
        station_id: str,
        num_source_cycles: int,
        standard_total_sec: float,
        zone_standard_times: dict,
        trajectory_points: list,
        raw_cycle_times: list,
    ) -> int:
        with self._session() as s:
            existing = s.execute(
                select(GoldenCycle).where(GoldenCycle.station_id == station_id)
            ).scalar_one_or_none()

            if existing:
                existing.recorded_at         = datetime.datetime.utcnow()
                existing.num_source_cycles   = num_source_cycles
                existing.standard_total_sec  = standard_total_sec
                existing.zone_standard_times = zone_standard_times
                existing.trajectory_points   = trajectory_points
                existing.raw_cycle_times     = raw_cycle_times
                s.commit()
                return existing.id
            else:
                gc = GoldenCycle(
                    station_id=station_id,
                    num_source_cycles=num_source_cycles,
                    standard_total_sec=standard_total_sec,
                    zone_standard_times=zone_standard_times,
                    trajectory_points=trajectory_points,
                    raw_cycle_times=raw_cycle_times,
                )
                s.add(gc)
                s.commit()
                return gc.id

    def load_golden_cycle(self, station_id: str) -> GoldenCycle | None:
        with self._session() as s:
            return s.execute(
                select(GoldenCycle).where(GoldenCycle.station_id == station_id)
            ).scalar_one_or_none()

    # ------------------------------------------------------------------
    # Summary statistics helpers
    # ------------------------------------------------------------------

    def get_session_stats(self, session_id: int) -> dict[str, Any]:
        """Return aggregate stats for a completed session."""
        cycles = self.get_session_cycles(session_id)
        completed = [c for c in cycles if c.status != "in_progress"]
        if not completed:
            return {}

        times = [c.cycle_time_sec for c in completed if c.cycle_time_sec]
        deviations = [c.deviation_pct for c in completed if c.deviation_pct is not None]
        zone_time_totals: dict[str, list[float]] = {}
        for c in completed:
            if c.zone_times:
                for z_id, t in c.zone_times.items():
                    zone_time_totals.setdefault(z_id, []).append(t)

        import statistics
        return {
            "total_cycles":    len(completed),
            "pass_cycles":     sum(1 for c in completed if c.status == "pass"),
            "fail_cycles":     sum(1 for c in completed if c.status == "fail"),
            "seq_error_cycles": sum(1 for c in completed if c.status == "sequence_error"),
            "avg_cycle_time":  statistics.mean(times) if times else 0,
            "min_cycle_time":  min(times) if times else 0,
            "max_cycle_time":  max(times) if times else 0,
            "stdev_cycle_time": statistics.stdev(times) if len(times) > 1 else 0,
            "avg_deviation_pct": statistics.mean(deviations) if deviations else 0,
            "cycle_times":     times,
            "zone_avg_times":  {
                z: statistics.mean(ts) for z, ts in zone_time_totals.items()
            },
            "sequence_errors": [
                e for c in completed
                for e in (c.sequence_errors or [])
            ],
        }

    # ------------------------------------------------------------------
    # Alert Log CRUD
    # ------------------------------------------------------------------

    def get_session_full_data(self, session_id: int) -> dict:
        """
        ดึงข้อมูลทั้งหมดของ session สำหรับ AI Summary Screen ใน 1 call

        Returns
        -------
        {
          "session"       : Session ORM row,
          "cycles"        : list[CycleLog],       # ทุก cycle เรียงลำดับ
          "stats"         : dict,                  # จาก get_session_stats()
          "alert_summary" : dict,                  # จาก get_session_alert_summary()
          "violation_summary": dict,               # จาก get_session_violation_summary()
          "per_cycle_alerts": {cycle_id: [AlertLog]},
          "per_cycle_violations": {cycle_id: [SequenceViolationLog]},
        }
        """
        session  = self.get_session(session_id)
        cycles   = self.get_session_cycles(session_id)
        stats    = self.get_session_stats(session_id)
        a_summ   = self.get_session_alert_summary(session_id)
        v_summ   = self.get_session_violation_summary(session_id)

        per_cycle_alerts:     dict[int, list] = {}
        per_cycle_violations: dict[int, list] = {}
        for c in cycles:
            per_cycle_alerts[c.id]     = self.get_cycle_alerts(c.id)
            per_cycle_violations[c.id] = self.get_cycle_violations(c.id)

        return {
            "session":               session,
            "cycles":                cycles,
            "stats":                 stats,
            "alert_summary":         a_summ,
            "violation_summary":     v_summ,
            "per_cycle_alerts":      per_cycle_alerts,
            "per_cycle_violations":  per_cycle_violations,
        }

    # ------------------------------------------------------------------
    # Alert Log CRUD
    # ------------------------------------------------------------------

    def log_alert(
        self,
        alert_level:  str,           # "WARNING" | "CRITICAL"
        message:      str,
        alert_type:   str    = "THRESHOLD_EXCEEDED",
        cycle_id:     int | None = None,
        zone_id:      int | None = None,
        elapsed_sec:  float | None = None,
        standard_sec: float | None = None,
        over_pct:     float | None = None,
    ) -> int:
        """
        บันทึก alert event ลง alert_logs table

        Returns
        -------
        id ของ AlertLog row ที่เพิ่งสร้าง
        """
        with self._session() as s:
            row = AlertLog(
                cycle_id     = cycle_id,
                zone_id      = zone_id,
                alert_type   = alert_type,
                alert_level  = alert_level,
                elapsed_sec  = elapsed_sec,
                standard_sec = standard_sec,
                over_pct     = over_pct,
                message      = message,
                occurred_at  = datetime.datetime.utcnow(),
            )
            s.add(row)
            s.commit()
            return row.id

    def get_cycle_alerts(
        self,
        cycle_id: int,
        level:    str | None = None,   # กรอง "WARNING" | "CRITICAL" ถ้าระบุ
    ) -> list[AlertLog]:
        """ดึง alert ทั้งหมดของ cycle นั้น เรียงตาม occurred_at"""
        with self._session() as s:
            stmt = (
                select(AlertLog)
                .where(AlertLog.cycle_id == cycle_id)
                .order_by(AlertLog.occurred_at)
            )
            if level:
                stmt = stmt.where(AlertLog.alert_level == level)
            return s.execute(stmt).scalars().all()

    def get_session_alert_summary(
        self,
        session_id: int,
    ) -> dict:
        """
        สรุป alert ของ session ทั้งหมด

        Returns
        -------
        {
          "total": int,
          "warnings": int,
          "criticals": int,
          "by_zone": {zone_id: count},
          "by_type":  {alert_type: count},
        }
        """
        with self._session() as s:
            cycle_ids = [
                row[0] for row in s.execute(
                    select(CycleLog.id)
                    .where(CycleLog.session_id == session_id)
                ).all()
            ]
            if not cycle_ids:
                return {"total": 0, "warnings": 0, "criticals": 0,
                        "by_zone": {}, "by_type": {}}

            alerts = s.execute(
                select(AlertLog).where(AlertLog.cycle_id.in_(cycle_ids))
            ).scalars().all()

            by_zone: dict[int | None, int] = {}
            by_type: dict[str, int]         = {}
            for a in alerts:
                by_zone[a.zone_id] = by_zone.get(a.zone_id, 0) + 1
                by_type[a.alert_type] = by_type.get(a.alert_type, 0) + 1

            return {
                "total":    len(alerts),
                "warnings":  sum(1 for a in alerts if a.alert_level == "WARNING"),
                "criticals": sum(1 for a in alerts if a.alert_level == "CRITICAL"),
                "by_zone":  by_zone,
                "by_type":  by_type,
            }

    # ------------------------------------------------------------------
    # Sequence Violation Log CRUD
    # ------------------------------------------------------------------

    def log_sequence_violation(
        self,
        violation_type:  str,           # "SKIP" | "OUT_OF_ORDER" | "REPEAT"
        actual_zone:     int,
        message:         str,
        cycle_id:        int | None  = None,
        expected_zone:   int | None  = None,
        skipped_zones:   list | None = None,
        sequence_so_far: list | None = None,
    ) -> int:
        """
        บันทึก sequence violation event

        Returns
        -------
        id ของ SequenceViolationLog row ที่เพิ่งสร้าง
        """
        with self._session() as s:
            row = SequenceViolationLog(
                cycle_id        = cycle_id,
                violation_type  = violation_type,
                expected_zone   = expected_zone,
                actual_zone     = actual_zone,
                skipped_zones   = skipped_zones or [],
                sequence_so_far = sequence_so_far or [],
                message         = message,
                occurred_at     = datetime.datetime.utcnow(),
            )
            s.add(row)
            s.commit()
            return row.id

    def get_cycle_violations(
        self,
        cycle_id:       int,
        violation_type: str | None = None,  # กรอง SKIP|OUT_OF_ORDER|REPEAT
    ) -> list[SequenceViolationLog]:
        """ดึง violation ทั้งหมดของ cycle นั้น เรียงตาม occurred_at"""
        with self._session() as s:
            stmt = (
                select(SequenceViolationLog)
                .where(SequenceViolationLog.cycle_id == cycle_id)
                .order_by(SequenceViolationLog.occurred_at)
            )
            if violation_type:
                stmt = stmt.where(
                    SequenceViolationLog.violation_type == violation_type
                )
            return s.execute(stmt).scalars().all()

    def get_session_violation_summary(
        self,
        session_id: int,
    ) -> dict:
        """
        สรุป sequence violation ของ session ทั้งหมด

        Returns
        -------
        {
          "total": int,
          "skip_count": int,
          "out_of_order_count": int,
          "repeat_count": int,
          "affected_cycles": int,
          "details": [{violation_type, expected, actual, message}, ...]
        }
        """
        with self._session() as s:
            cycle_ids = [
                row[0] for row in s.execute(
                    select(CycleLog.id)
                    .where(CycleLog.session_id == session_id)
                ).all()
            ]
            if not cycle_ids:
                return {"total": 0, "skip_count": 0,
                        "out_of_order_count": 0, "repeat_count": 0,
                        "affected_cycles": 0, "details": []}

            viols = s.execute(
                select(SequenceViolationLog)
                .where(SequenceViolationLog.cycle_id.in_(cycle_ids))
                .order_by(SequenceViolationLog.occurred_at)
            ).scalars().all()

            affected = len({v.cycle_id for v in viols})

            return {
                "total":              len(viols),
                "skip_count":         sum(1 for v in viols if v.violation_type == "SKIP"),
                "out_of_order_count": sum(1 for v in viols if v.violation_type == "OUT_OF_ORDER"),
                "repeat_count":       sum(1 for v in viols if v.violation_type == "REPEAT"),
                "affected_cycles":    affected,
                "details": [
                    {
                        "violation_type": v.violation_type,
                        "expected_zone":  v.expected_zone,
                        "actual_zone":    v.actual_zone,
                        "message":        v.message,
                    }
                    for v in viols
                ],
            }
