"""
tests/test_cycle_comparator.py
==============================
Unit tests สำหรับ core/dtw_comparator.py

ทดสอบด้วยข้อมูลจำลอง (mock CycleRecord) — ไม่ต้องการกล้อง, MediaPipe, หรือ Qt

วิธีรัน:
    python -m pytest tests/test_cycle_comparator.py -v
    # หรือรันโดยตรง:
    python tests/test_cycle_comparator.py
"""

from __future__ import annotations

import math
import sys
import os
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.dtw_comparator import (
    AlignmentPath,
    ComparisonResult,
    DTWComparator,
    GoldenCycleProcessor,
    GoldenReference,
    ZoneDeviation,
    _dtw_compute,
    _normalise,
    _numpy_dtw,
    _resample,
    _to_array,
)
from core.cycle_tracker import CycleRecord, ZoneTiming


# ============================================================================
# ── Mock data builders ────────────────────────────────────────────────────────
# ============================================================================

def _make_zone_timing(
    zone_id: int,
    duration: float,
    enter_at: float = 0.0,
) -> ZoneTiming:
    """สร้าง ZoneTiming จำลองที่มี duration ตามที่ระบุ"""
    zt           = ZoneTiming(zone_id=zone_id)
    zt.enter_time = enter_at
    zt.exit_time  = enter_at + duration
    zt.elapsed    = duration
    return zt


def _make_trajectory(
    n_points: int = 50,
    x_start: float = 0.0,
    x_end:   float = 100.0,
    y_start: float = 0.0,
    y_end:   float = 50.0,
    noise:   float = 0.0,
) -> list[dict]:
    """
    สร้าง trajectory จำลอง (เส้นตรงจาก start ไป end + noise)
    คืน list[{x, y, zone_id, t_abs, t_norm}]
    """
    xs = np.linspace(x_start, x_end, n_points)
    ys = np.linspace(y_start, y_end, n_points)
    if noise > 0:
        rng = np.random.default_rng(seed=42)
        xs += rng.normal(0, noise, n_points)
        ys += rng.normal(0, noise, n_points)
    return [
        {
            "x": float(xs[i]),
            "y": float(ys[i]),
            "zone_id": None,
            "t_abs":  float(i),
            "t_norm": float(i) / (n_points - 1),
        }
        for i in range(n_points)
    ]


def _make_record(
    cycle_number: int = 1,
    z1_dur:   float = 3.0,
    z2_dur:   float = 2.0,
    z3_dur:   float = 1.5,
    n_traj:   int   = 50,
    traj_noise: float = 0.0,
    start_time: float = 0.0,
) -> CycleRecord:
    """
    สร้าง CycleRecord จำลองพร้อม zone timings และ trajectory
    """
    total = z1_dur + z2_dur + z3_dur
    rec = CycleRecord(
        cycle_number = cycle_number,
        start_time   = start_time,
        end_time     = start_time + total,
    )
    rec.zone_timings = {
        1: _make_zone_timing(1, z1_dur, enter_at=start_time),
        2: _make_zone_timing(2, z2_dur, enter_at=start_time + z1_dur),
        3: _make_zone_timing(3, z3_dur, enter_at=start_time + z1_dur + z2_dur),
    }
    rec.trajectory = _make_trajectory(
        n_points = n_traj,
        noise    = traj_noise,
    )
    return rec


# ── Golden records ชุด "สม่ำเสมอ" ──────────────────────────────────────────

def _golden_records_consistent() -> list[CycleRecord]:
    """5 cycle ที่มีเวลาใกล้เคียงกัน — median ควรชัดเจน"""
    return [
        _make_record(i + 1, z1_dur=3.0 + i * 0.1,
                             z2_dur=2.0 + i * 0.05,
                             z3_dur=1.5)
        for i in range(5)
    ]


def _golden_records_with_outlier() -> list[CycleRecord]:
    """3 cycle ปกติ + 1 outlier (ช้ามากผิดปกติ)"""
    records = [_make_record(i + 1, z1_dur=3.0, z2_dur=2.0, z3_dur=1.5)
               for i in range(3)]
    records.append(_make_record(4, z1_dur=30.0, z2_dur=20.0, z3_dur=15.0))
    return records


# ============================================================================
# ── Helper functions tests ────────────────────────────────────────────────────
# ============================================================================

class TestHelpers:

    def test_to_array_basic(self):
        traj = [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]
        arr  = _to_array(traj)
        assert arr.shape == (2, 2)
        assert arr[0, 0] == 1.0
        assert arr[1, 1] == 4.0

    def test_to_array_empty(self):
        arr = _to_array([])
        assert arr.shape == (1, 2)

    def test_resample_output_shape(self):
        arr = _to_array(_make_trajectory(20))
        res = _resample(arr, 100)
        assert res.shape == (100, 2)

    def test_resample_preserves_endpoints(self):
        arr = np.array([[0.0, 0.0], [10.0, 20.0]])
        res = _resample(arr, 50)
        assert res[0, 0]  == pytest.approx(0.0,  abs=0.1)
        assert res[-1, 0] == pytest.approx(10.0, abs=0.1)
        assert res[-1, 1] == pytest.approx(20.0, abs=0.1)

    def test_resample_single_point(self):
        arr = np.array([[5.0, 7.0]])
        res = _resample(arr, 10)
        assert res.shape == (10, 2)
        assert np.allclose(res, 5.0, atol=0.1)  # ทุก point ควรเหมือนกัน

    def test_normalise_range(self):
        arr = np.array([[0.0, 0.0], [100.0, 50.0], [50.0, 25.0]])
        norm = _normalise(arr)
        assert norm.min() == pytest.approx(0.0)
        assert norm.max() == pytest.approx(1.0)

    def test_normalise_constant_dimension(self):
        """กัน division-by-zero เมื่อ x หรือ y ค่าคงที่"""
        arr  = np.array([[5.0, 3.0], [5.0, 3.0], [5.0, 3.0]])
        norm = _normalise(arr)
        assert not np.any(np.isnan(norm))


# ============================================================================
# ── DTW computation tests ─────────────────────────────────────────────────────
# ============================================================================

class TestDTWCompute:

    def test_identical_trajectories_zero_distance(self):
        """trajectory เหมือนกันทุกประการ → distance = 0"""
        arr = _normalise(_resample(_to_array(_make_trajectory(50)), 200))
        dist, path = _dtw_compute(arr, arr)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_distance_positive_for_different_trajectories(self):
        """trajectory ต่างกัน → distance > 0"""
        a1 = _normalise(_resample(_to_array(_make_trajectory(50, x_end=100)), 200))
        a2 = _normalise(_resample(_to_array(_make_trajectory(50, x_end=200, y_end=200)), 200))
        dist, _ = _dtw_compute(a1, a2)
        assert dist > 0.0

    def test_alignment_path_covers_full_range(self):
        """alignment path ต้องครอบคลุมตั้งแต่ index 0 ถึง N-1 ของทั้งสอง series"""
        arr1 = _normalise(_resample(_to_array(_make_trajectory(30)), 200))
        arr2 = _normalise(_resample(_to_array(_make_trajectory(30, y_end=100)), 200))
        _, path = _dtw_compute(arr1, arr2)
        live_indices   = [p[0] for p in path]
        golden_indices = [p[1] for p in path]
        assert min(live_indices)   == 0
        assert min(golden_indices) == 0
        assert max(live_indices)   == 199
        assert max(golden_indices) == 199

    def test_alignment_path_monotonic(self):
        """alignment path ต้องเดินไปข้างหน้าเสมอ (monotonically non-decreasing)"""
        arr1 = _normalise(_resample(_to_array(_make_trajectory(40)), 200))
        arr2 = _normalise(_resample(_to_array(_make_trajectory(40, y_end=80)), 200))
        _, path = _dtw_compute(arr1, arr2)
        live_prev = golden_prev = -1
        for li, gi in path:
            assert li >= live_prev,   "live index ต้องไม่ถอยหลัง"
            assert gi >= golden_prev, "golden index ต้องไม่ถอยหลัง"
            live_prev, golden_prev = li, gi

    def test_numpy_dtw_matches_fastdtw_approx(self):
        """
        pure-NumPy DTW ควรให้ผลใกล้เคียงกับ fastdtw (ถ้ามี)
        ทดสอบเฉพาะ numpy_dtw โดยตรงว่า path ถูกต้อง
        """
        a1 = np.array([[float(i), float(i)] for i in range(20)])
        a2 = np.array([[float(i) + 0.5, float(i) - 0.5] for i in range(20)])
        dist, path = _numpy_dtw(a1, a2)
        assert dist > 0
        assert len(path) >= max(len(a1), len(a2))  # path ต้องยาวพอ

    def test_symmetry_approx(self):
        """
        DTW ไม่จำเป็นต้อง symmetric อย่างสมบูรณ์เสมอไป
        แต่ควรใกล้เคียง (ต่างกันไม่เกิน 5%)
        """
        a1 = _normalise(_resample(_to_array(_make_trajectory(30)), 100))
        a2 = _normalise(_resample(_to_array(_make_trajectory(30, y_end=80)), 100))
        d12, _ = _dtw_compute(a1, a2)
        d21, _ = _dtw_compute(a2, a1)
        if d12 > 0:
            ratio = abs(d12 - d21) / d12
            assert ratio < 0.05, f"asymmetry {ratio:.3f} เกิน 5%"


# ============================================================================
# ── GoldenCycleProcessor tests ────────────────────────────────────────────────
# ============================================================================

class TestGoldenCycleProcessor:

    def test_empty_records_raises(self):
        with pytest.raises(ValueError, match="อย่างน้อย"):
            GoldenCycleProcessor.process([])

    def test_single_record_warns_but_works(self):
        rec = _make_record()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            golden = GoldenCycleProcessor.process([rec])
        assert any("อย่างน้อย" in str(warning.message) for warning in w)
        assert isinstance(golden, GoldenReference)

    def test_standard_times_are_median(self):
        """
        5 cycle: z1 = 3.0, 3.1, 3.2, 3.3, 3.4
        median z1 ต้องเป็น 3.2
        """
        records = [
            _make_record(i + 1, z1_dur=3.0 + i * 0.1)
            for i in range(5)
        ]
        golden = GoldenCycleProcessor.process(records)
        assert golden.standard_times[1] == pytest.approx(3.2, abs=0.01)

    def test_median_vs_mean_with_outlier(self):
        """
        ด้วย outlier ที่ช้ามาก:
        median z1 ต้องใกล้ 3.0 (ไม่ถูกดึงขึ้นโดย outlier)
        mean z1 จะเป็น (3+3+3+30)/4 = 9.75 — ห้ามใช้ mean
        """
        records = _golden_records_with_outlier()
        golden  = GoldenCycleProcessor.process(records)
        # median ของ [3.0, 3.0, 3.0, 30.0] = 3.0
        assert golden.standard_times[1] == pytest.approx(3.0, abs=0.01)
        assert golden.standard_times[1] < 5.0, "median ห้ามถูก outlier ดึงขึ้น"

    def test_representative_cycle_is_closest_to_median(self):
        """
        representative cycle ต้องเป็น cycle ที่ total_time ใกล้ median ที่สุด
        """
        records = [
            _make_record(1, z1_dur=3.0, z2_dur=2.0, z3_dur=1.5),  # total=6.5
            _make_record(2, z1_dur=3.5, z2_dur=2.2, z3_dur=1.8),  # total=7.5
            _make_record(3, z1_dur=4.0, z2_dur=2.5, z3_dur=2.0),  # total=8.5
        ]
        # median total = 7.5 → rep ควรเป็น record 2
        golden = GoldenCycleProcessor.process(records)
        assert golden.representative_record is not None
        rep_total = golden.representative_record.total_time
        assert rep_total == pytest.approx(7.5, abs=0.01)

    def test_total_standard_time_is_median(self):
        records = [
            _make_record(1, z1_dur=3.0, z2_dur=2.0, z3_dur=1.5),  # 6.5
            _make_record(2, z1_dur=3.0, z2_dur=2.0, z3_dur=1.5),  # 6.5
            _make_record(3, z1_dur=4.0, z2_dur=3.0, z3_dur=2.5),  # 9.5
            _make_record(4, z1_dur=3.0, z2_dur=2.0, z3_dur=1.5),  # 6.5
            _make_record(5, z1_dur=3.0, z2_dur=2.0, z3_dur=1.5),  # 6.5
        ]
        golden = GoldenCycleProcessor.process(records)
        # median ของ [6.5, 6.5, 9.5, 6.5, 6.5] = 6.5
        assert golden.total_standard_time == pytest.approx(6.5, abs=0.01)

    def test_trajectory_from_representative(self):
        """trajectory ใน GoldenReference ต้องมาจาก representative cycle"""
        records = _golden_records_consistent()
        golden  = GoldenCycleProcessor.process(records)
        assert golden.trajectory.shape[0] == GoldenCycleProcessor.N_RESAMPLE
        assert golden.trajectory.shape[1] == 2

    def test_all_zones_covered(self):
        records = _golden_records_consistent()
        golden  = GoldenCycleProcessor.process(records)
        for z in [1, 2, 3]:
            assert z in golden.standard_times
            assert golden.standard_times[z] > 0

    def test_n_cycles_recorded(self):
        records = _golden_records_consistent()
        golden  = GoldenCycleProcessor.process(records)
        assert golden.n_cycles_recorded == len(records)

    def test_custom_zone_order(self):
        """ทดสอบ zone_order แบบกำหนดเอง"""
        rec = _make_record()
        rec.zone_timings = {
            10: _make_zone_timing(10, 2.0),
            20: _make_zone_timing(20, 1.5),
        }
        rec.trajectory = _make_trajectory(20)
        golden = GoldenCycleProcessor.process([rec], zone_order=[10, 20])
        assert 10 in golden.standard_times
        assert 20 in golden.standard_times


# ============================================================================
# ── GoldenReference tests ─────────────────────────────────────────────────────
# ============================================================================

class TestGoldenReference:

    def _make_golden(self) -> GoldenReference:
        return GoldenCycleProcessor.process(_golden_records_consistent())

    def test_ghost_position_at_0_pct(self):
        golden = self._make_golden()
        x, y = golden.ghost_position_at_progress(0.0)
        # trajectory เริ่มที่ (0, 0) (ตาม _make_trajectory default)
        assert abs(x) < 5.0
        assert abs(y) < 5.0

    def test_ghost_position_at_100_pct(self):
        golden = self._make_golden()
        x, y = golden.ghost_position_at_progress(100.0)
        # trajectory จบที่ (100, 50) (ตาม _make_trajectory default)
        assert abs(x - 100.0) < 5.0
        assert abs(y - 50.0)  < 5.0

    def test_ghost_position_at_50_pct(self):
        golden = self._make_golden()
        x, y = golden.ghost_position_at_progress(50.0)
        # กลาง trajectory ควรอยู่แถว x=50, y=25
        assert 30.0 < x < 70.0
        assert 10.0 < y < 40.0

    def test_ghost_position_at_frame(self):
        golden = self._make_golden()
        x0, y0 = golden.ghost_position_at_frame(0)
        xN, yN = golden.ghost_position_at_frame(199)
        assert x0 != xN or y0 != yN, "frame 0 และ frame 199 ควรต่างกัน"

    def test_ghost_frame_out_of_range_clamped(self):
        golden = self._make_golden()
        x_neg, _  = golden.ghost_position_at_frame(-10)
        x_pos, _  = golden.ghost_position_at_frame(9999)
        x_0,   _  = golden.ghost_position_at_frame(0)
        x_last, _ = golden.ghost_position_at_frame(199)
        assert x_neg == x_0
        assert x_pos == x_last

    def test_repr_contains_n(self):
        records = _golden_records_consistent()
        golden  = GoldenCycleProcessor.process(records)
        r = repr(golden)
        assert str(len(records)) in r


# ============================================================================
# ── DTWComparator.compare() tests ─────────────────────────────────────────────
# ============================================================================

class TestDTWComparatorCompare:

    def _make_comparator(
        self,
        alert_threshold: int = 30,
        records: list | None = None,
    ) -> DTWComparator:
        comp = DTWComparator(alert_threshold_pct=alert_threshold)
        recs = records or _golden_records_consistent()
        comp.set_golden(GoldenCycleProcessor.process(recs))
        return comp

    # ── No golden ──────────────────────────────────────────────

    def test_compare_without_golden_returns_empty_result(self):
        comp   = DTWComparator()
        live   = _make_record()
        result = comp.compare(live)
        assert result.has_golden is False
        assert result.similarity_score == 0.0
        assert result.deviation_per_zone == {}

    # ── deviation_per_zone ──────────────────────────────────────

    def test_deviation_per_zone_keys(self):
        comp   = self._make_comparator()
        live   = _make_record(z1_dur=3.0, z2_dur=2.0, z3_dur=1.5)
        result = comp.compare(live)
        assert set(result.deviation_per_zone.keys()) == {1, 2, 3}

    def test_deviation_zero_when_matching_golden(self):
        """ถ้า live ตรงกับ standard เป๊ะ → diff ควรใกล้ 0"""
        records = [_make_record(i, z1_dur=3.0, z2_dur=2.0, z3_dur=1.5)
                   for i in range(3)]
        comp   = DTWComparator()
        comp.set_golden(GoldenCycleProcessor.process(records))
        live   = _make_record(z1_dur=3.0, z2_dur=2.0, z3_dur=1.5)
        result = comp.compare(live)
        assert result.deviation_per_zone[1].diff_seconds == pytest.approx(0.0, abs=0.01)
        assert result.deviation_per_zone[2].diff_seconds == pytest.approx(0.0, abs=0.01)
        assert result.deviation_per_zone[3].diff_seconds == pytest.approx(0.0, abs=0.01)

    def test_deviation_positive_when_live_slower(self):
        """live ช้ากว่า standard → diff_seconds > 0"""
        records = [_make_record(i, z1_dur=3.0) for i in range(3)]
        comp    = DTWComparator()
        comp.set_golden(GoldenCycleProcessor.process(records))
        live   = _make_record(z1_dur=6.0)   # ช้ากว่า standard 3 วินาที
        result = comp.compare(live)
        assert result.deviation_per_zone[1].diff_seconds == pytest.approx(3.0, abs=0.1)

    def test_deviation_negative_when_live_faster(self):
        """live เร็วกว่า standard → diff_seconds < 0"""
        records = [_make_record(i, z1_dur=3.0) for i in range(3)]
        comp    = DTWComparator()
        comp.set_golden(GoldenCycleProcessor.process(records))
        live   = _make_record(z1_dur=1.5)   # เร็วกว่า standard
        result = comp.compare(live)
        assert result.deviation_per_zone[1].diff_seconds < 0

    def test_deviation_pct_calculation(self):
        """
        standard = 4.0s, actual = 6.0s
        diff = 2.0s, diff_pct = 50.0%
        """
        records = [_make_record(i, z1_dur=4.0) for i in range(3)]
        comp    = DTWComparator()
        comp.set_golden(GoldenCycleProcessor.process(records))
        live   = _make_record(z1_dur=6.0)
        result = comp.compare(live)
        assert result.deviation_per_zone[1].diff_pct == pytest.approx(50.0, abs=1.0)

    def test_over_threshold_flag(self):
        """diff_pct > alert_threshold → is_over_threshold = True"""
        records = [_make_record(i, z1_dur=2.0) for i in range(3)]
        comp    = DTWComparator(alert_threshold_pct=30)
        comp.set_golden(GoldenCycleProcessor.process(records))
        live   = _make_record(z1_dur=3.0)   # 50% over → เกิน 30%
        result = comp.compare(live)
        assert result.deviation_per_zone[1].is_over_threshold is True
        assert result.is_over_threshold is True
        assert 1 in result.over_threshold_zones

    def test_under_threshold_flag(self):
        """diff_pct ≤ alert_threshold → is_over_threshold = False"""
        records = [_make_record(i, z1_dur=3.0) for i in range(3)]
        comp    = DTWComparator(alert_threshold_pct=30)
        comp.set_golden(GoldenCycleProcessor.process(records))
        live   = _make_record(z1_dur=3.2)   # ~6.7% over → ต่ำกว่า 30%
        result = comp.compare(live)
        assert result.deviation_per_zone[1].is_over_threshold is False

    # ── total_cycle_time_diff ───────────────────────────────────

    def test_total_cycle_time_diff_value(self):
        """total_cycle_time_diff = actual_total − standard_total"""
        records = [_make_record(i, z1_dur=3.0, z2_dur=2.0, z3_dur=1.5)
                   for i in range(3)]
        comp   = DTWComparator()
        comp.set_golden(GoldenCycleProcessor.process(records))
        live   = _make_record(z1_dur=4.0, z2_dur=2.0, z3_dur=1.5)  # total +1
        result = comp.compare(live)
        assert result.total_cycle_time_diff == pytest.approx(1.0, abs=0.1)

    def test_total_diff_pct(self):
        """
        standard_total = 6.5, live_total = 9.5
        diff = 3.0, diff_pct = 3.0/6.5 × 100 ≈ 46.2%
        """
        records = [_make_record(i, z1_dur=3.0, z2_dur=2.0, z3_dur=1.5)
                   for i in range(3)]
        comp   = DTWComparator()
        comp.set_golden(GoldenCycleProcessor.process(records))
        live   = _make_record(z1_dur=6.0, z2_dur=2.0, z3_dur=1.5)  # +3s
        result = comp.compare(live)
        expected_pct = 3.0 / 6.5 * 100
        assert result.total_diff_pct == pytest.approx(expected_pct, abs=1.0)

    # ── similarity_score ────────────────────────────────────────

    def test_similarity_high_for_similar_trajectory(self):
        """trajectory เหมือน golden → similarity ควร > 70"""
        records = [_make_record(i) for i in range(3)]
        comp    = DTWComparator()
        comp.set_golden(GoldenCycleProcessor.process(records))
        live   = _make_record(traj_noise=2.0)   # noise เบาๆ
        result = comp.compare(live)
        assert result.similarity_score > 70.0, \
            f"trajectory ที่คล้ายกันควรได้ score > 70 แต่ได้ {result.similarity_score}"

    def test_similarity_lower_for_different_trajectory(self):
        """trajectory ต่างกันมาก → similarity ควรต่ำกว่า trajectory ที่คล้ายกัน"""
        records = [_make_record(i) for i in range(3)]
        comp    = DTWComparator()
        comp.set_golden(GoldenCycleProcessor.process(records))
        live_similar  = _make_record(traj_noise=1.0)
        live_different = _make_record(
            n_traj=50,
            traj_noise=0.0,
        )
        # ปรับ trajectory ให้ต่างมาก
        live_different.trajectory = _make_trajectory(
            50, x_start=500, x_end=600, y_start=400, y_end=500
        )
        r_similar   = comp.compare(live_similar)
        r_different = comp.compare(live_different)
        assert r_similar.similarity_score > r_different.similarity_score, \
            "trajectory ที่คล้ายกันควรได้ score สูงกว่า"

    def test_similarity_score_range(self):
        """similarity_score ต้องอยู่ใน [0, 100]"""
        comp = self._make_comparator()
        for _ in range(5):
            live   = _make_record(traj_noise=5.0)
            result = comp.compare(live)
            assert 0.0 <= result.similarity_score <= 100.0

    def test_similarity_100_for_identical_trajectory(self):
        """
        ถ้า live trajectory เหมือน representative golden เป๊ะ
        → similarity ต้องสูงมาก (> 95)
        """
        records = [_make_record(i, z1_dur=3.0, z2_dur=2.0, z3_dur=1.5)
                   for i in range(3)]
        golden = GoldenCycleProcessor.process(records)
        comp   = DTWComparator()
        comp.set_golden(golden)

        # ใช้ trajectory เดียวกับ representative
        live = _make_record(z1_dur=3.0, z2_dur=2.0, z3_dur=1.5)
        live.trajectory = list(golden.raw_trajectory)
        result = comp.compare(live)
        assert result.similarity_score > 95.0, \
            f"trajectory เหมือนกันเป๊ะ ควรได้ score > 95 แต่ได้ {result.similarity_score}"

    # ── alignment_path ──────────────────────────────────────────

    def test_alignment_path_not_empty(self):
        comp   = self._make_comparator()
        live   = _make_record()
        result = comp.compare(live)
        assert len(result.alignment_path) > 0

    def test_alignment_path_type(self):
        """alignment_path ต้องเป็น list ของ tuple(int, int)"""
        comp   = self._make_comparator()
        live   = _make_record()
        result = comp.compare(live)
        for item in result.alignment_path[:5]:
            assert len(item) == 2
            assert isinstance(item[0], int)
            assert isinstance(item[1], int)

    def test_alignment_path_starts_at_zero(self):
        comp   = self._make_comparator()
        live   = _make_record()
        result = comp.compare(live)
        assert result.alignment_path[0] == (0, 0)

    def test_alignment_path_ends_at_last_index(self):
        comp   = self._make_comparator()
        live   = _make_record()
        result = comp.compare(live)
        li, gi = result.alignment_path[-1]
        assert li == 199
        assert gi == 199

    # ── empty trajectory handling ───────────────────────────────

    def test_compare_no_trajectory_returns_valid_result(self):
        comp = self._make_comparator()
        live = _make_record()
        live.trajectory = []
        result = comp.compare(live)
        assert result.has_golden is True
        assert result.similarity_score == 0.0
        # alignment path fallback linear
        assert len(result.alignment_path) == 200

    # ── last_result ─────────────────────────────────────────────

    def test_last_result_updated_after_compare(self):
        comp = self._make_comparator()
        assert comp.last_result is None
        comp.compare(_make_record())
        assert comp.last_result is not None

    def test_last_result_replaced_on_next_compare(self):
        comp = self._make_comparator()
        comp.compare(_make_record(z1_dur=3.0))
        r1 = comp.last_result
        comp.compare(_make_record(z1_dur=5.0))
        r2 = comp.last_result
        assert r1 is not r2


# ============================================================================
# ── ComparisonResult.ghost_golden_idx tests ───────────────────────────────────
# ============================================================================

class TestComparisonResultGhostIdx:

    def _make_result_with_path(self) -> ComparisonResult:
        comp   = DTWComparator()
        comp.set_golden(GoldenCycleProcessor.process(_golden_records_consistent()))
        live   = _make_record()
        return comp.compare(live)

    def test_ghost_golden_idx_0(self):
        result = self._make_result_with_path()
        idx = result.ghost_golden_idx(0)
        assert idx == 0

    def test_ghost_golden_idx_last(self):
        result = self._make_result_with_path()
        idx = result.ghost_golden_idx(199)
        assert idx == 199

    def test_ghost_golden_idx_monotonic(self):
        """ghost index ต้องไม่ลด เมื่อ live_frame_idx เพิ่มขึ้น"""
        result = self._make_result_with_path()
        prev = -1
        for i in range(0, 200, 10):
            idx = result.ghost_golden_idx(i)
            assert idx >= prev
            prev = idx

    def test_ghost_golden_idx_no_path_fallback(self):
        """ถ้า alignment_path ว่าง → fallback linear ต้องไม่ error"""
        result = ComparisonResult(alignment_path=[])
        idx = result.ghost_golden_idx(100)
        assert isinstance(idx, int)
        assert idx >= 0


# ============================================================================
# ── DTWComparator.ghost_at_live_frame tests ───────────────────────────────────
# ============================================================================

class TestGhostAtLiveFrame:

    def _make_comp_with_result(self) -> tuple[DTWComparator, ComparisonResult]:
        comp = DTWComparator()
        comp.set_golden(GoldenCycleProcessor.process(_golden_records_consistent()))
        live   = _make_record()
        result = comp.compare(live)
        return comp, result

    def test_ghost_at_frame_0(self):
        comp, result = self._make_comp_with_result()
        x, y = comp.ghost_at_live_frame(0, result)
        assert abs(x) < 10.0
        assert abs(y) < 10.0

    def test_ghost_at_frame_last(self):
        comp, result = self._make_comp_with_result()
        x, y = comp.ghost_at_live_frame(199, result)
        assert abs(x - 100.0) < 10.0
        assert abs(y - 50.0)  < 10.0

    def test_ghost_without_explicit_result_uses_last(self):
        comp, _ = self._make_comp_with_result()
        x, y = comp.ghost_at_live_frame(50)   # ใช้ last_result
        assert isinstance(x, float)
        assert isinstance(y, float)

    def test_ghost_no_golden_returns_origin(self):
        comp = DTWComparator()
        x, y = comp.ghost_at_live_frame(50)
        assert x == 0.0
        assert y == 0.0

    def test_ghost_position_progresses_across_frames(self):
        """ghost position ต้องเปลี่ยนไปตาม frame (ไม่ค้างอยู่ที่เดิม)"""
        comp, result = self._make_comp_with_result()
        positions = [comp.ghost_at_live_frame(i, result) for i in range(0, 200, 20)]
        unique_x = len(set(p[0] for p in positions))
        assert unique_x > 3, "ghost ต้องเคลื่อนที่ไปหลายตำแหน่ง"


# ============================================================================
# ── ZoneDeviation tests ───────────────────────────────────────────────────────
# ============================================================================

class TestZoneDeviation:

    def test_repr_contains_zone_id(self):
        zd = ZoneDeviation(
            zone_id=2,
            actual_seconds=3.5,
            standard_seconds=3.0,
            diff_seconds=0.5,
            diff_pct=16.7,
        )
        assert "Zone2" in repr(zd)
        assert "+0.50" in repr(zd)

    def test_repr_negative_diff(self):
        zd = ZoneDeviation(
            zone_id=1,
            actual_seconds=2.0,
            standard_seconds=3.0,
            diff_seconds=-1.0,
            diff_pct=-33.3,
        )
        r = repr(zd)
        assert "-1.00" in r

    def test_over_threshold_default_false(self):
        zd = ZoneDeviation(1, 1.0, 1.0, 0.0, 0.0)
        assert zd.is_over_threshold is False


# ============================================================================
# ── Integration: end-to-end golden recording + comparison ─────────────────────
# ============================================================================

class TestEndToEnd:
    """
    ทดสอบ flow ทั้งหมด:
    บันทึก golden → process → set → compare live → ตรวจผล
    """

    def test_full_flow_consistent_operator(self):
        """
        ผู้ปฏิบัติงานทำงานสม่ำเสมอ: golden 5 cycle ปกติ
        → live cycle ที่ใกล้เคียง → similarity สูง, deviation น้อย
        """
        golden_recs = [
            _make_record(i, z1_dur=3.0, z2_dur=2.0, z3_dur=1.5)
            for i in range(5)
        ]
        golden = GoldenCycleProcessor.process(golden_recs)
        comp   = DTWComparator(alert_threshold_pct=30)
        comp.set_golden(golden)

        live   = _make_record(z1_dur=3.1, z2_dur=2.05, z3_dur=1.45)
        result = comp.compare(live)

        assert result.has_golden
        assert result.similarity_score > 60.0
        assert not result.is_over_threshold
        assert abs(result.total_cycle_time_diff) < 0.5

    def test_full_flow_slow_operator_triggers_alert(self):
        """
        live cycle ช้ากว่า golden เกิน threshold → is_over_threshold = True
        """
        golden_recs = [
            _make_record(i, z1_dur=3.0, z2_dur=2.0, z3_dur=1.5)
            for i in range(3)
        ]
        golden = GoldenCycleProcessor.process(golden_recs)
        comp   = DTWComparator(alert_threshold_pct=30)
        comp.set_golden(golden)

        # zone 1 ช้ากว่า standard 60% (3.0 → 4.8)
        live   = _make_record(z1_dur=4.8, z2_dur=2.0, z3_dur=1.5)
        result = comp.compare(live)

        assert result.is_over_threshold
        assert 1 in result.over_threshold_zones

    def test_full_flow_with_outlier_in_golden(self):
        """
        golden มี outlier → ด้วย median, standard time ยังคงสมเหตุสมผล
        live ปกติ → ไม่ถูก alert เพราะ standard ไม่ถูก outlier ดึง
        """
        golden_recs = _golden_records_with_outlier()
        golden = GoldenCycleProcessor.process(golden_recs)
        comp   = DTWComparator(alert_threshold_pct=30)
        comp.set_golden(golden)

        live   = _make_record(z1_dur=3.1, z2_dur=2.1, z3_dur=1.6)
        result = comp.compare(live)

        # standard_z1 = median([3,3,3,30]) = 3.0 → live 3.1 ≈ 3.3% over → ไม่ alert
        assert not result.deviation_per_zone[1].is_over_threshold, \
            "ด้วย median standard ปกติ live ที่ใกล้เคียงไม่ควร alert"

    def test_alignment_path_usable_for_ghost_sync(self):
        """
        ทดสอบว่าใช้ alignment_path เพื่อ sync ghost overlay ได้จริง
        แต่ละ live frame idx ต้อง map ไปยัง golden frame idx ที่ valid
        """
        golden_recs = [_make_record(i) for i in range(3)]
        golden = GoldenCycleProcessor.process(golden_recs)
        comp   = DTWComparator()
        comp.set_golden(golden)
        live   = _make_record()
        result = comp.compare(live)

        for live_frame in [0, 50, 100, 150, 199]:
            gold_idx = result.ghost_golden_idx(live_frame)
            assert 0 <= gold_idx <= 199, \
                f"golden_idx {gold_idx} ต้องอยู่ใน [0, 199]"
            x, y = golden.ghost_position_at_frame(gold_idx)
            assert isinstance(x, float)
            assert isinstance(y, float)

    def test_summary_line_format(self):
        comp = DTWComparator()
        comp.set_golden(GoldenCycleProcessor.process(
            [_make_record(i) for i in range(3)]
        ))
        result = comp.compare(_make_record())
        line   = result.summary_line()
        assert "Total:"    in line
        assert "std"       in line
        assert "DTW"       in line


if __name__ == "__main__":
    import unittest
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestHelpers,
        TestDTWCompute,
        TestGoldenCycleProcessor,
        TestGoldenReference,
        TestDTWComparatorCompare,
        TestComparisonResultGhostIdx,
        TestGhostAtLiveFrame,
        TestZoneDeviation,
        TestEndToEnd,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
