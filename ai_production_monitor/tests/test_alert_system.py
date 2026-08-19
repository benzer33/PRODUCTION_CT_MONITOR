"""
tests/test_alert_system.py
==========================
Unit tests สำหรับระบบแจ้งเตือน:
  1. ThresholdConfig / ZoneThreshold — per-zone thresholds
  2. AlertCounter — นับ alert แบบ in-memory
  3. AlertManager.check_realtime() — real-time threshold detection
  4. AlertManager cooldown / deduplication
  5. SequenceChecker — SKIP / OUT_OF_ORDER / REPEAT detection
  6. DB integration mock tests — log_alert / log_sequence_violation

ไม่ต้องการกล้อง, MediaPipe, Qt, หรือ SQLite จริง

วิธีรัน:
    python -m pytest tests/test_alert_system.py -v
"""

from __future__ import annotations

import sys
import os
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.alert_manager import (
    AlertCounter,
    AlertEvent,
    AlertLevel,
    AlertManager,
    AlertType,
    ThresholdConfig,
    ZoneThreshold,
)
from core.sequence_checker import (
    SequenceChecker,
    SequenceViolation,
    ViolationType,
)


# ============================================================================
# ── ZoneThreshold tests ───────────────────────────────────────────────────────
# ============================================================================

class TestZoneThreshold:

    def test_no_alert_below_warning(self):
        t = ZoneThreshold(warning_pct=20, critical_pct=50)
        assert t.level_for(10.0) is None
        assert t.level_for(19.9) is None

    def test_warning_at_threshold(self):
        t = ZoneThreshold(warning_pct=20, critical_pct=50)
        assert t.level_for(20.0) == AlertLevel.WARNING
        assert t.level_for(30.0) == AlertLevel.WARNING

    def test_critical_at_threshold(self):
        t = ZoneThreshold(warning_pct=20, critical_pct=50)
        assert t.level_for(50.0) == AlertLevel.CRITICAL
        assert t.level_for(100.0) == AlertLevel.CRITICAL

    def test_critical_takes_priority_over_warning(self):
        """เมื่อเกิน critical → ต้องคืน CRITICAL ไม่ใช่ WARNING"""
        t = ZoneThreshold(warning_pct=20, critical_pct=50)
        assert t.level_for(60.0) == AlertLevel.CRITICAL

    def test_zero_warning_disables_warning(self):
        t = ZoneThreshold(warning_pct=0, critical_pct=50)
        assert t.level_for(30.0) is None   # 30% ไม่มี warning
        assert t.level_for(50.0) == AlertLevel.CRITICAL

    def test_zero_critical_disables_critical(self):
        t = ZoneThreshold(warning_pct=20, critical_pct=0)
        assert t.level_for(100.0) == AlertLevel.WARNING   # ไม่มี critical


# ============================================================================
# ── ThresholdConfig tests ─────────────────────────────────────────────────────
# ============================================================================

class TestThresholdConfig:

    def test_global_fallback(self):
        cfg = ThresholdConfig(global_warning=15, global_critical=40)
        thr = cfg.get_threshold(zone_id=99)
        assert thr.warning_pct  == 15
        assert thr.critical_pct == 40

    def test_per_zone_overrides_global(self):
        cfg = ThresholdConfig(global_warning=20, global_critical=50)
        cfg.set_zone(zone_id=2, warning_pct=10, critical_pct=25)
        thr2 = cfg.get_threshold(2)
        thr1 = cfg.get_threshold(1)
        assert thr2.warning_pct  == 10
        assert thr2.critical_pct == 25
        assert thr1.warning_pct  == 20   # zone 1 ยัง global

    def test_remove_per_zone_restores_global(self):
        cfg = ThresholdConfig(global_warning=20, global_critical=50)
        cfg.set_zone(zone_id=1, warning_pct=5, critical_pct=10)
        cfg.remove_zone(zone_id=1)
        thr = cfg.get_threshold(1)
        assert thr.warning_pct == 20

    def test_set_global_updates_fallback(self):
        cfg = ThresholdConfig(global_warning=20, global_critical=50)
        cfg.set_global(warning_pct=30, critical_pct=70)
        thr = cfg.get_threshold(99)
        assert thr.warning_pct  == 30
        assert thr.critical_pct == 70

    def test_per_zone_not_affected_by_global_update(self):
        cfg = ThresholdConfig(global_warning=20, global_critical=50)
        cfg.set_zone(2, 5, 15)
        cfg.set_global(30, 70)
        thr2 = cfg.get_threshold(2)
        assert thr2.warning_pct == 5   # per-zone ไม่โดน override


# ============================================================================
# ── AlertCounter tests ────────────────────────────────────────────────────────
# ============================================================================

class TestAlertCounter:

    def test_initial_zero(self):
        c = AlertCounter()
        assert c.total == 0
        assert c.count() == 0

    def test_increment_and_total(self):
        c = AlertCounter()
        c.increment(zone_id=1, level=AlertLevel.WARNING)
        c.increment(zone_id=1, level=AlertLevel.CRITICAL)
        c.increment(zone_id=2, level=AlertLevel.WARNING)
        assert c.total == 3

    def test_count_by_zone(self):
        c = AlertCounter()
        c.increment(1, AlertLevel.WARNING)
        c.increment(1, AlertLevel.CRITICAL)
        c.increment(2, AlertLevel.WARNING)
        assert c.count(zone_id=1) == 2
        assert c.count(zone_id=2) == 1
        assert c.count(zone_id=3) == 0

    def test_count_by_level(self):
        c = AlertCounter()
        c.increment(1, AlertLevel.WARNING)
        c.increment(2, AlertLevel.WARNING)
        c.increment(1, AlertLevel.CRITICAL)
        assert c.count(level=AlertLevel.WARNING)  == 2
        assert c.count(level=AlertLevel.CRITICAL) == 1

    def test_count_by_zone_and_level(self):
        c = AlertCounter()
        c.increment(1, AlertLevel.WARNING)
        c.increment(1, AlertLevel.CRITICAL)
        c.increment(2, AlertLevel.WARNING)
        assert c.count(zone_id=1, level=AlertLevel.WARNING)  == 1
        assert c.count(zone_id=1, level=AlertLevel.CRITICAL) == 1
        assert c.count(zone_id=2, level=AlertLevel.CRITICAL) == 0

    def test_reset(self):
        c = AlertCounter()
        c.increment(1, AlertLevel.WARNING)
        c.increment(2, AlertLevel.CRITICAL)
        c.reset()
        assert c.total == 0
        assert c.count(zone_id=1) == 0

    def test_summary_keys(self):
        c = AlertCounter()
        c.increment(1, AlertLevel.WARNING)
        c.increment(1, AlertLevel.CRITICAL)
        summary = c.summary()
        assert (1, "WARNING")  in summary
        assert (1, "CRITICAL") in summary


# ============================================================================
# ── AlertManager.check_realtime tests ─────────────────────────────────────────
# ============================================================================

class TestAlertManagerRealtime:

    def _make_manager(self, warning=20, critical=50, cooldown=0.0):
        log = []
        mgr = AlertManager(
            on_alert     = lambda e: log.append(e),
            warning_pct  = warning,
            critical_pct = critical,
            cooldown_sec = cooldown,
            audio_enabled = False,
        )
        return mgr, log

    # ── threshold detection ────────────────────────────────────

    def test_no_alert_below_warning(self):
        mgr, log = self._make_manager(warning=20, critical=50, cooldown=0)
        result = mgr.check_realtime(1, elapsed_sec=2.0, standard_sec=2.0)
        assert result is None
        assert len(log) == 0

    def test_warning_alert_emitted(self):
        mgr, log = self._make_manager(warning=20, critical=50, cooldown=0)
        result = mgr.check_realtime(1, elapsed_sec=2.5, standard_sec=2.0)
        # over_pct = 25% → WARNING
        assert result is not None
        assert result.alert_level == AlertLevel.WARNING
        assert len(log) == 1

    def test_critical_alert_emitted(self):
        mgr, log = self._make_manager(warning=20, critical=50, cooldown=0)
        result = mgr.check_realtime(1, elapsed_sec=3.1, standard_sec=2.0)
        # over_pct = 55% → CRITICAL
        assert result is not None
        assert result.alert_level == AlertLevel.CRITICAL

    def test_alert_event_fields(self):
        mgr, log = self._make_manager(warning=20, critical=50, cooldown=0)
        mgr.check_realtime(1, elapsed_sec=3.0, standard_sec=2.0, cycle_id=42)
        e = log[0]
        assert e.zone_id      == 1
        assert e.elapsed_sec  == pytest.approx(3.0)
        assert e.standard_sec == pytest.approx(2.0)
        assert e.over_pct     == pytest.approx(50.0, abs=0.1)
        assert e.alert_type   == AlertType.THRESHOLD_EXCEEDED

    def test_zero_standard_returns_none(self):
        mgr, log = self._make_manager()
        result = mgr.check_realtime(1, elapsed_sec=10.0, standard_sec=0.0)
        assert result is None

    # ── per-zone threshold ────────────────────────────────────

    def test_per_zone_threshold_override(self):
        mgr, log = self._make_manager(warning=30, critical=60, cooldown=0)
        mgr.set_zone_threshold(zone_id=2, warning_pct=5, critical_pct=15)

        # Zone 1: global threshold, 20% over → ไม่ถึง 30%
        r1 = mgr.check_realtime(1, elapsed_sec=2.4, standard_sec=2.0)  # 20%
        assert r1 is None

        # Zone 2: per-zone threshold, 20% over → เกิน 15% CRITICAL
        r2 = mgr.check_realtime(2, elapsed_sec=2.4, standard_sec=2.0)
        assert r2 is not None
        assert r2.alert_level == AlertLevel.CRITICAL

    def test_per_zone_critical_lower_than_global(self):
        mgr, log = self._make_manager(warning=20, critical=50, cooldown=0)
        mgr.set_zone_threshold(3, warning_pct=10, critical_pct=20)
        # Zone 3: 25% over → CRITICAL (เกิน 20% per-zone critical)
        r = mgr.check_realtime(3, elapsed_sec=2.5, standard_sec=2.0)
        assert r is not None
        assert r.alert_level == AlertLevel.CRITICAL

    # ── cooldown / deduplication ──────────────────────────────

    def test_cooldown_prevents_repeat_within_window(self):
        mgr, log = self._make_manager(warning=10, critical=30, cooldown=60.0)
        mgr.check_realtime(1, elapsed_sec=2.5, standard_sec=2.0)  # emit
        mgr.check_realtime(1, elapsed_sec=2.6, standard_sec=2.0)  # cooldown
        mgr.check_realtime(1, elapsed_sec=2.7, standard_sec=2.0)  # cooldown
        assert len(log) == 1, "cooldown ต้อง block การ emit ซ้ำ"

    def test_zero_cooldown_allows_every_frame(self):
        mgr, log = self._make_manager(warning=10, critical=30, cooldown=0.0)
        for _ in range(5):
            mgr.check_realtime(1, elapsed_sec=2.5, standard_sec=2.0)
        assert len(log) == 5

    def test_different_zones_have_independent_cooldown(self):
        mgr, log = self._make_manager(warning=10, critical=30, cooldown=60.0)
        mgr.check_realtime(1, elapsed_sec=2.5, standard_sec=2.0)  # zone 1 emit
        mgr.check_realtime(2, elapsed_sec=2.5, standard_sec=2.0)  # zone 2 emit
        assert len(log) == 2, "zone ต่างกันต้อง emit ได้อิสระ"

    # ── counter integration ───────────────────────────────────

    def test_counter_increments_on_emit(self):
        mgr, _ = self._make_manager(warning=10, critical=50, cooldown=0)
        mgr.check_realtime(1, elapsed_sec=2.5, standard_sec=2.0)
        mgr.check_realtime(1, elapsed_sec=2.5, standard_sec=2.0)
        assert mgr.counter.count(zone_id=1) == 2

    def test_counter_separates_warning_and_critical(self):
        mgr, _ = self._make_manager(warning=10, critical=50, cooldown=0)
        mgr.check_realtime(1, elapsed_sec=2.2, standard_sec=2.0)   # 10% → WARNING
        mgr.check_realtime(1, elapsed_sec=3.1, standard_sec=2.0)   # 55% → CRITICAL
        assert mgr.counter.count(zone_id=1, level=AlertLevel.WARNING)  == 1
        assert mgr.counter.count(zone_id=1, level=AlertLevel.CRITICAL) == 1

    # ── DB callback ───────────────────────────────────────────

    def test_db_callback_called_with_cycle_id(self):
        db_log = []
        mgr = AlertManager(
            on_alert_with_db = lambda e, cid: db_log.append((e, cid)),
            warning_pct  = 10,
            critical_pct = 30,
            cooldown_sec = 0,
            audio_enabled = False,
        )
        mgr.check_realtime(1, elapsed_sec=2.5, standard_sec=2.0, cycle_id=99)
        assert len(db_log) == 1
        event, cid = db_log[0]
        assert cid == 99
        assert isinstance(event, AlertEvent)

    def test_db_callback_uses_current_cycle_if_not_specified(self):
        db_log = []
        mgr = AlertManager(
            on_alert_with_db = lambda e, cid: db_log.append(cid),
            warning_pct  = 10,
            cooldown_sec = 0,
            audio_enabled = False,
        )
        mgr.set_current_cycle(cycle_id=77)
        mgr.check_realtime(1, elapsed_sec=2.5, standard_sec=2.0)
        assert db_log[0] == 77

    # ── backward compat: trigger_threshold ───────────────────

    def test_trigger_threshold_backward_compat(self):
        mgr, log = self._make_manager(warning=20, critical=50, cooldown=0)
        mgr.trigger_threshold(1, elapsed_sec=2.5, standard_sec=2.0)
        assert len(log) == 1

    # ── sequence violation ────────────────────────────────────

    def test_sequence_violation_always_emits(self):
        """sequence violation ห้าม cooldown block"""
        mgr, log = self._make_manager(cooldown=60.0)
        mgr.trigger_sequence_violation(expected_zone=2, actual_zone=3)
        mgr.trigger_sequence_violation(expected_zone=2, actual_zone=3)
        seq_alerts = [e for e in log if e.alert_type == AlertType.SEQUENCE_VIOLATION]
        assert len(seq_alerts) == 2

    def test_sequence_violation_is_critical(self):
        mgr, log = self._make_manager(cooldown=0)
        mgr.trigger_sequence_violation(2, 3)
        assert log[-1].alert_level == AlertLevel.CRITICAL

    # ── history / clear ───────────────────────────────────────

    def test_alert_history_accumulates(self):
        mgr, _ = self._make_manager(warning=10, cooldown=0)
        mgr.check_realtime(1, elapsed_sec=2.5, standard_sec=2.0)
        mgr.check_realtime(2, elapsed_sec=2.5, standard_sec=2.0)
        assert len(mgr.alert_history) == 2

    def test_clear_history_resets_all(self):
        mgr, _ = self._make_manager(warning=10, cooldown=0)
        mgr.check_realtime(1, elapsed_sec=2.5, standard_sec=2.0)
        mgr.clear_history()
        assert len(mgr.alert_history) == 0
        assert mgr.counter.total == 0

    def test_recent_alerts_returns_last_n(self):
        mgr, _ = self._make_manager(warning=10, cooldown=0)
        for i in range(5):
            mgr.check_realtime(i + 1, elapsed_sec=2.5, standard_sec=2.0)
        recent = mgr.recent_alerts(3)
        assert len(recent) == 3


# ============================================================================
# ── SequenceChecker tests ─────────────────────────────────────────────────────
# ============================================================================

class TestSequenceChecker:

    def _make(self, seq=(1, 2, 3), allow_repeat=False):
        log = []
        checker = SequenceChecker(
            expected_sequence = list(seq),
            allow_repeat      = allow_repeat,
            on_violation      = lambda v: log.append(v),
        )
        checker.reset()
        return checker, log

    # ── ปกติ — ไม่มี violation ────────────────────────────────

    def test_correct_sequence_no_violation(self):
        checker, log = self._make()
        for zone in [1, 2, 3]:
            v = checker.observe_enter(zone)
            assert v is None
        assert len(log) == 0

    def test_is_complete_after_all_zones(self):
        checker, _ = self._make()
        for z in [1, 2, 3]:
            checker.observe_enter(z)
        assert checker.is_complete

    def test_visited_zones_recorded(self):
        checker, _ = self._make()
        checker.observe_enter(1)
        checker.observe_enter(2)
        assert checker.visited_zones == [1, 2]

    def test_next_expected_advances(self):
        checker, _ = self._make()
        assert checker.next_expected_zone == 1
        checker.observe_enter(1)
        assert checker.next_expected_zone == 2
        checker.observe_enter(2)
        assert checker.next_expected_zone == 3

    def test_reset_clears_state(self):
        checker, _ = self._make()
        checker.observe_enter(1)
        checker.observe_enter(2)
        checker.reset()
        assert checker.visited_zones     == []
        assert checker.next_expected_zone == 1
        assert not checker.is_complete

    # ── SKIP ─────────────────────────────────────────────────

    def test_skip_one_zone_detected(self):
        """
        ★ Core SKIP ★
        ลำดับ [1,2,3] เข้า 1 แล้วข้ามไป 3 = skip zone 2
        """
        checker, log = self._make()
        checker.observe_enter(1)
        v = checker.observe_enter(3)
        assert v is not None
        assert v.violation_type == ViolationType.SKIP
        assert v.skipped_zones  == [2]
        assert v.actual_zone    == 3
        assert v.expected_zone  == 2

    def test_skip_two_zones_detected(self):
        """ลำดับ [1,2,3,4] เข้า 1 แล้วข้ามไป 4 = skip [2,3]"""
        checker, log = self._make(seq=(1, 2, 3, 4))
        checker.observe_enter(1)
        v = checker.observe_enter(4)
        assert v is not None
        assert v.violation_type == ViolationType.SKIP
        assert 2 in v.skipped_zones
        assert 3 in v.skipped_zones

    def test_skip_from_start(self):
        """เริ่มด้วย zone 2 แทน zone 1 = skip zone 1"""
        checker, log = self._make()
        v = checker.observe_enter(2)
        assert v is not None
        assert v.violation_type == ViolationType.SKIP
        assert 1 in v.skipped_zones

    def test_skip_advances_expected_pointer(self):
        """หลัง SKIP ถึง zone 3 → expected pointer ต้องอยู่หลัง zone 3"""
        checker, _ = self._make()
        checker.observe_enter(1)
        checker.observe_enter(3)  # skip 2
        assert checker.next_expected_zone is None or checker.is_complete

    def test_callback_fired_on_skip(self):
        checker, log = self._make()
        checker.observe_enter(1)
        checker.observe_enter(3)
        assert len(log) == 1
        assert log[0].violation_type == ViolationType.SKIP

    # ── OUT_OF_ORDER ─────────────────────────────────────────

    def test_out_of_order_backward_detected(self):
        """
        ★ Core OUT_OF_ORDER ★
        เข้า 1→2→3 แล้วย้อนกลับไป 1
        """
        checker, log = self._make()
        checker.observe_enter(1)
        checker.observe_enter(2)
        checker.observe_enter(3)
        v = checker.observe_enter(1)
        assert v is not None
        assert v.violation_type == ViolationType.OUT_OF_ORDER
        assert v.actual_zone == 1

    def test_out_of_order_zone_not_in_sequence(self):
        """เข้า zone ที่ไม่อยู่ใน sequence เลย → OUT_OF_ORDER"""
        checker, log = self._make(seq=(1, 2, 3))
        checker.observe_enter(1)
        v = checker.observe_enter(99)
        assert v is not None
        assert v.violation_type == ViolationType.OUT_OF_ORDER

    def test_out_of_order_after_complete(self):
        """Cycle เสร็จแล้ว เข้า zone เพิ่ม → OUT_OF_ORDER"""
        checker, _ = self._make()
        for z in [1, 2, 3]:
            checker.observe_enter(z)
        v = checker.observe_enter(1)
        assert v is not None
        assert v.violation_type == ViolationType.OUT_OF_ORDER

    def test_message_contains_zone_ids(self):
        checker, _ = self._make()
        checker.observe_enter(1)
        v = checker.observe_enter(3)
        assert "3" in v.message
        assert "2" in v.message   # skipped

    # ── REPEAT ───────────────────────────────────────────────

    def test_repeat_detected(self):
        """
        ★ Core REPEAT ★
        เข้า zone 1 แล้วเข้า zone 1 ซ้ำทันที
        """
        checker, log = self._make()
        checker.observe_enter(1)
        v = checker.observe_enter(1)
        assert v is not None
        assert v.violation_type == ViolationType.REPEAT

    def test_repeat_not_detected_after_different_zone(self):
        """เข้า 1→2→1 ไม่ใช่ REPEAT (มี zone อื่นคั่น) แต่เป็น OUT_OF_ORDER"""
        checker, log = self._make()
        checker.observe_enter(1)
        checker.observe_enter(2)
        v = checker.observe_enter(1)
        # ไม่ใช่ REPEAT เพราะ last_zone = 2
        assert v is not None
        assert v.violation_type != ViolationType.REPEAT

    def test_allow_repeat_disables_repeat_detection(self):
        checker, log = self._make(allow_repeat=True)
        checker.observe_enter(1)
        v = checker.observe_enter(1)
        # ไม่ detect REPEAT แต่อาจ detect OUT_OF_ORDER
        assert v is None or v.violation_type != ViolationType.REPEAT

    def test_repeat_middle_of_sequence(self):
        """เข้า 1→2→2 = REPEAT ที่ zone 2"""
        checker, _ = self._make()
        checker.observe_enter(1)
        checker.observe_enter(2)
        v = checker.observe_enter(2)
        assert v is not None
        assert v.violation_type == ViolationType.REPEAT
        assert v.actual_zone == 2

    # ── Multiple violations in one cycle ─────────────────────

    def test_multiple_violations_recorded(self):
        checker, log = self._make()
        checker.observe_enter(1)
        checker.observe_enter(1)   # REPEAT
        checker.observe_enter(3)   # SKIP zone 2
        assert len(log) == 2

    def test_violation_count_property(self):
        checker, _ = self._make()
        checker.observe_enter(1)
        checker.observe_enter(3)   # SKIP
        assert checker.violation_count == 1

    # ── sequence_so_far ───────────────────────────────────────

    def test_sequence_so_far_at_violation(self):
        """sequence_so_far ควรเป็น snapshot ก่อน zone ปัจจุบัน append"""
        checker, _ = self._make()
        checker.observe_enter(1)
        checker.observe_enter(2)
        v = checker.observe_enter(2)   # REPEAT
        # snapshot ก่อน append = [1, 2]
        assert v.sequence_so_far == [1, 2]

    # ── Custom zone IDs ───────────────────────────────────────

    def test_custom_zone_ids(self):
        """ทำงานกับ zone ID ที่ไม่ใช่ 1,2,3"""
        checker, log = self._make(seq=(10, 20, 30))
        checker.observe_enter(10)
        v = checker.observe_enter(30)   # skip 20
        assert v is not None
        assert v.violation_type == ViolationType.SKIP
        assert 20 in v.skipped_zones

    def test_two_zone_sequence(self):
        checker, log = self._make(seq=(1, 2))
        checker.observe_enter(1)
        v = checker.observe_enter(2)
        assert v is None
        assert checker.is_complete

    # ── reset_violations ──────────────────────────────────────

    def test_reset_violations_clears_history(self):
        checker, log = self._make()
        checker.observe_enter(1)
        checker.observe_enter(3)   # SKIP
        checker.reset_violations()
        assert checker.violation_count == 0
        assert checker.all_violations  == []


# ============================================================================
# ── AlertEvent serialisation tests ───────────────────────────────────────────
# ============================================================================

class TestAlertEvent:

    def test_to_dict_keys(self):
        e = AlertEvent(
            alert_type   = AlertType.THRESHOLD_EXCEEDED,
            alert_level  = AlertLevel.WARNING,
            zone_id      = 2,
            message      = "test",
            elapsed_sec  = 3.0,
            standard_sec = 2.0,
            over_pct     = 50.0,
        )
        d = e.to_dict()
        assert "type"        in d
        assert "level"       in d
        assert "zone_id"     in d
        assert "over_pct"    in d
        assert d["type"]  == "THRESHOLD_EXCEEDED"
        assert d["level"] == "WARNING"

    def test_str_representation_warning(self):
        e = AlertEvent(
            alert_type  = AlertType.THRESHOLD_EXCEEDED,
            alert_level = AlertLevel.WARNING,
            zone_id     = 1,
            message     = "test warning",
            over_pct    = 25.0,
        )
        s = str(e)
        assert "Zone 1" in s
        assert "25" in s

    def test_str_representation_critical(self):
        e = AlertEvent(
            alert_type  = AlertType.THRESHOLD_EXCEEDED,
            alert_level = AlertLevel.CRITICAL,
            zone_id     = 2,
            message     = "critical!",
            over_pct    = 60.0,
        )
        s = str(e)
        assert "Zone 2" in s


# ============================================================================
# ── SequenceViolation serialisation tests ────────────────────────────────────
# ============================================================================

class TestSequenceViolation:

    def test_to_dict_keys(self):
        v = SequenceViolation(
            violation_type  = ViolationType.SKIP,
            actual_zone     = 3,
            expected_zone   = 2,
            skipped_zones   = [2],
            sequence_so_far = [1],
            message         = "SKIP: ...",
        )
        d = v.to_dict()
        assert "violation_type"  in d
        assert "actual_zone"     in d
        assert "skipped_zones"   in d
        assert "sequence_so_far" in d
        assert d["violation_type"] == "SKIP"

    def test_str_representation(self):
        v = SequenceViolation(
            violation_type  = ViolationType.OUT_OF_ORDER,
            actual_zone     = 1,
            expected_zone   = 3,
            skipped_zones   = [],
            sequence_so_far = [1, 2, 3],
            message         = "OUT_OF_ORDER",
        )
        assert "OUT_OF_ORDER" in str(v)


# ============================================================================
# ── Integration: AlertManager + SequenceChecker together ─────────────────────
# ============================================================================

class TestAlertSystemIntegration:
    """
    จำลอง flow จริง: checker detect violation → alert_mgr emit + DB log
    ไม่ใช้ DB จริง — mock callback แทน
    """

    def test_skip_triggers_sequence_alert(self):
        """SKIP → SequenceChecker → AlertManager emit sequence alert"""
        alert_log = []
        db_log    = []

        mgr = AlertManager(
            on_alert         = lambda e: alert_log.append(e),
            on_alert_with_db = lambda e, cid: db_log.append((e, cid)),
            warning_pct      = 20,
            critical_pct     = 50,
            cooldown_sec     = 0,
            audio_enabled    = False,
        )
        mgr.set_current_cycle(cycle_id=5)

        checker = SequenceChecker(
            expected_sequence = [1, 2, 3],
            on_violation      = lambda v: mgr.trigger_sequence_violation(
                expected_zone = v.expected_zone or 0,
                actual_zone   = v.actual_zone,
                cycle_id      = 5,
                detail        = v.violation_type.name,
            ),
        )
        checker.reset()
        checker.observe_enter(1)
        checker.observe_enter(3)   # SKIP zone 2

        # ต้องมี 1 sequence alert
        seq_alerts = [e for e in alert_log if e.alert_type == AlertType.SEQUENCE_VIOLATION]
        assert len(seq_alerts) == 1
        assert seq_alerts[0].alert_level == AlertLevel.CRITICAL

        # DB callback ต้องถูกเรียกพร้อม cycle_id ถูกต้อง
        assert len(db_log) == 1
        assert db_log[0][1] == 5

    def test_threshold_and_sequence_independent(self):
        """threshold alert และ sequence alert ทำงานอิสระต่อกัน"""
        alert_log = []
        mgr = AlertManager(
            on_alert     = lambda e: alert_log.append(e),
            warning_pct  = 10,
            critical_pct = 30,
            cooldown_sec = 0,
            audio_enabled = False,
        )

        # Threshold alert
        mgr.check_realtime(1, elapsed_sec=2.5, standard_sec=2.0)

        # Sequence alert
        mgr.trigger_sequence_violation(2, 3)

        assert len(alert_log) == 2
        types = {e.alert_type for e in alert_log}
        assert AlertType.THRESHOLD_EXCEEDED in types
        assert AlertType.SEQUENCE_VIOLATION in types

    def test_realtime_threshold_emits_at_exact_boundary(self):
        """
        standard=2.0, warning=20% → alert เมื่อ elapsed = 2.4s พอดี
        ก่อนหน้านั้น (elapsed=2.39) ไม่ควร emit
        """
        mgr, log = TestAlertManagerRealtime()._make_manager(
            warning=20, critical=50, cooldown=0
        )
        mgr.check_realtime(1, elapsed_sec=2.39, standard_sec=2.0)  # 19.5% → no
        mgr.check_realtime(1, elapsed_sec=2.40, standard_sec=2.0)  # 20.0% → WARNING
        warnings = [e for e in log if e.alert_level == AlertLevel.WARNING]
        assert len(warnings) == 1

    def test_counter_tracks_across_multiple_zones(self):
        mgr, _ = TestAlertManagerRealtime()._make_manager(warning=10, cooldown=0)
        mgr.check_realtime(1, elapsed_sec=2.5, standard_sec=2.0)
        mgr.check_realtime(2, elapsed_sec=3.0, standard_sec=2.0)
        mgr.check_realtime(1, elapsed_sec=2.6, standard_sec=2.0)
        assert mgr.counter.count(zone_id=1) == 2
        assert mgr.counter.count(zone_id=2) == 1
        assert mgr.counter.total == 3


if __name__ == "__main__":
    import unittest
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestZoneThreshold,
        TestThresholdConfig,
        TestAlertCounter,
        TestAlertManagerRealtime,
        TestSequenceChecker,
        TestAlertEvent,
        TestSequenceViolation,
        TestAlertSystemIntegration,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
