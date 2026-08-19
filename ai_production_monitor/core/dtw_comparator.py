"""
core/dtw_comparator.py
======================
เปรียบเทียบ cycle จริงกับ Golden Cycle ด้วย Dynamic Time Warping (DTW)

ทำไมต้องใช้ DTW?
─────────────────
ผู้ปฏิบัติงานแต่ละครั้งทำงานด้วยความเร็วต่างกัน  frame-by-frame diff
จะรายงานว่า "ช้า" แม้ว่าเส้นทางการเคลื่อนที่จะเหมือนกันทุกประการ
DTW บิด (warp) แกนเวลาของทั้งสองลำดับเพื่อหา alignment ที่มีต้นทุนน้อยที่สุด
ทำให้ค่าเบี่ยงเบนสะท้อนความต่างของ "วิธีทำ" ไม่ใช่แค่ "ความเร็ว"

Public API
──────────
    # สร้าง golden reference จาก 3-5 cycle ที่บันทึกไว้
    golden = GoldenCycleProcessor.process(list_of_cycle_records)

    # เปรียบเทียบ cycle จริง
    comparator = DTWComparator()
    comparator.set_golden(golden)
    result = comparator.compare(live_cycle_record)

    # ใช้ alignment path สำหรับ ghost overlay
    ghost_x, ghost_y = comparator.ghost_at_live_frame(live_frame_idx, result)

DTW Backend Priority
────────────────────
1. dtaidistance  (เร็วกว่า, รองรับ multivariate)
2. fastdtw       (fallback)
3. Pure-NumPy Sakoe-Chiba band DTW  (fallback สุดท้าย, ไม่ต้องติดตั้งอะไร)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── DTW backend detection ──────────────────────────────────────────────────
_BACKEND = "numpy"  # จะถูก override ด้านล่าง

try:
    import dtaidistance.dtw as _dtai_dtw  # type: ignore
    import dtaidistance.dtw_ndim as _dtai_nd  # type: ignore
    _BACKEND = "dtaidistance"
except ImportError:
    try:
        from fastdtw import fastdtw as _fastdtw  # type: ignore
        from scipy.spatial.distance import euclidean as _euclidean  # type: ignore
        _BACKEND = "fastdtw"
    except ImportError:
        pass  # ใช้ pure-NumPy fallback


# ============================================================================
# ── Trajectory helpers ──────────────────────────────────────────────────────
# ============================================================================

def _to_array(trajectory: list[dict]) -> np.ndarray:
    """
    แปลง trajectory list[{x, y, ...}] → ndarray shape (N, 2)
    รองรับ key ทั้ง 'x'/'y' และ dict ที่มี field เพิ่มเติม
    """
    if not trajectory:
        return np.zeros((1, 2), dtype=np.float64)
    return np.array(
        [[float(pt["x"]), float(pt["y"])] for pt in trajectory],
        dtype=np.float64,
    )


def _resample(arr: np.ndarray, n: int = 200) -> np.ndarray:
    """
    Resample trajectory (N, 2) → (n, 2) ด้วย arc-length parameterisation
    เพื่อให้ความยาวของทั้งสอง trajectory เท่ากันก่อน DTW
    """
    if len(arr) < 2:
        return np.tile(arr[0] if len(arr) else np.zeros(2), (n, 1))

    diffs   = np.diff(arr, axis=0)
    dists   = np.sqrt((diffs ** 2).sum(axis=1))
    cumdist = np.concatenate([[0.0], np.cumsum(dists)])
    total   = cumdist[-1]

    if total == 0.0:
        return np.tile(arr[0], (n, 1))

    t_new = np.linspace(0.0, total, n)
    return np.column_stack([
        np.interp(t_new, cumdist, arr[:, 0]),
        np.interp(t_new, cumdist, arr[:, 1]),
    ])


def _normalise(arr: np.ndarray) -> np.ndarray:
    """
    Normalise (N, 2) trajectory ให้อยู่ใน [0, 1] × [0, 1]
    เพื่อให้ DTW ไม่ขึ้นกับ resolution กล้อง
    ใช้เฉพาะตอนเปรียบเทียบ shape — ไม่แตะ pixel-space ของ ghost overlay
    """
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    rng  = maxs - mins
    rng[rng == 0] = 1.0   # กัน division-by-zero
    return (arr - mins) / rng


# ============================================================================
# ── DTW computation — ใช้ backend ที่ดีที่สุดที่มี ──────────────────────────
# ============================================================================

AlignmentPath = list[tuple[int, int]]
"""list of (live_idx, golden_idx) pairs จาก DTW alignment"""


def _dtw_compute(
    live_arr:   np.ndarray,
    golden_arr: np.ndarray,
) -> tuple[float, AlignmentPath]:
    """
    คำนวณ DTW distance และ alignment path ระหว่าง 2 trajectory (N×2)

    คืนค่า
    ──────
    (distance, path)
    path = [(live_i, golden_j), ...]  — ใช้สำหรับ ghost overlay sync
    """
    # ── 1. dtaidistance (เร็วสุด) ───────────────────────────────────
    if _BACKEND == "dtaidistance":
        # dtaidistance.dtw_ndim รองรับ multivariate โดยตรง
        dist = float(_dtai_nd.distance(live_arr, golden_arr))
        # warp path จาก 1-D wrapper (normalised ด้วย x เป็น proxy)
        try:
            path_raw = _dtai_nd.warping_path(live_arr, golden_arr)
            path: AlignmentPath = [(int(a), int(b)) for a, b in path_raw]
        except Exception:
            path = _fallback_path(live_arr, golden_arr)
        return dist, path

    # ── 2. fastdtw ──────────────────────────────────────────────────
    if _BACKEND == "fastdtw":
        dist_raw, path_raw = _fastdtw(live_arr, golden_arr, dist=_euclidean)
        path = [(int(a), int(b)) for a, b in path_raw]
        return float(dist_raw), path

    # ── 3. Pure-NumPy Sakoe-Chiba DTW (fallback) ────────────────────
    return _numpy_dtw(live_arr, golden_arr, band_pct=0.15)


def _fallback_path(a: np.ndarray, b: np.ndarray) -> AlignmentPath:
    """
    สร้าง path แบบ linear interpolation (ใช้เมื่อ library ไม่คืน path)
    """
    n, m = len(a), len(b)
    path = []
    for i in range(max(n, m)):
        li = min(int(i * n / max(n, m)), n - 1)
        gi = min(int(i * m / max(n, m)), m - 1)
        path.append((li, gi))
    return path


def _numpy_dtw(
    a: np.ndarray,
    b: np.ndarray,
    band_pct: float = 0.15,
) -> tuple[float, AlignmentPath]:
    """
    Sakoe-Chiba Band DTW ด้วย pure NumPy
    band_pct = ความกว้าง band เป็น % ของ max(N, M)

    Complexity: O(N × band_width)  แทน O(N²)
    """
    n, m   = len(a), len(b)
    band_w = max(1, int(band_pct * max(n, m)))

    INF = np.inf
    # cost matrix (distance ระหว่าง point คู่)
    # คำนวณแบบ lazy เพื่อประหยัด memory
    dtw_mat = np.full((n + 1, m + 1), INF)
    dtw_mat[0, 0] = 0.0

    for i in range(1, n + 1):
        j_lo = max(1, i - band_w)
        j_hi = min(m, i + band_w)
        for j in range(j_lo, j_hi + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            dtw_mat[i, j] = cost + min(
                dtw_mat[i - 1, j],      # insertion
                dtw_mat[i, j - 1],      # deletion
                dtw_mat[i - 1, j - 1],  # match
            )

    distance = float(dtw_mat[n, m])

    # traceback เพื่อได้ alignment path
    path: AlignmentPath = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        choices = {
            (i - 1, j - 1): dtw_mat[i - 1, j - 1],
            (i - 1, j):     dtw_mat[i - 1, j],
            (i,     j - 1): dtw_mat[i,     j - 1],
        }
        (i, j) = min(choices, key=choices.get)  # type: ignore[arg-type]
    path.reverse()

    return distance, path


# ============================================================================
# ── Data containers ─────────────────────────────────────────────────────────
# ============================================================================

@dataclass
class ZoneDeviation:
    """ค่าเบี่ยงเบนของ 1 zone เทียบกับ golden"""

    zone_id:          int
    actual_seconds:   float    # เวลาจริงของ zone นี้ใน cycle ปัจจุบัน
    standard_seconds: float    # เวลา median ของ golden (0 ถ้าไม่มี golden)
    diff_seconds:     float    # actual − standard  (+ = ช้ากว่า golden)
    diff_pct:         float    # diff / standard × 100  (0 ถ้าไม่มี standard)
    is_over_threshold: bool = False  # True ถ้าเกิน alert threshold %

    def __repr__(self) -> str:
        sign = "+" if self.diff_seconds >= 0 else ""
        return (
            f"Zone{self.zone_id} "
            f"actual={self.actual_seconds:.2f}s "
            f"std={self.standard_seconds:.2f}s "
            f"diff={sign}{self.diff_seconds:.2f}s ({sign}{self.diff_pct:.1f}%)"
        )


@dataclass
class ComparisonResult:
    """
    ผลลัพธ์ทั้งหมดของการเปรียบเทียบ 1 cycle กับ golden

    Fields
    ──────
    deviation_per_zone   : {zone_id: ZoneDeviation}  ค่าเบี่ยงเบนต่อ zone
    total_cycle_time     : เวลารวม cycle จริง (วินาที)
    total_standard_time  : เวลารวม golden (วินาที)
    total_cycle_time_diff: total_cycle_time − total_standard_time
    total_diff_pct       : total_cycle_time_diff / total_standard_time × 100
    similarity_score     : 0–100  (100 = เส้นทางเหมือน golden ทุกประการ)
    dtw_distance         : raw DTW distance (pixel-normalised units)
    alignment_path       : [(live_idx, golden_idx), ...]  สำหรับ ghost sync
    has_golden           : False ถ้าไม่มี golden reference (ค่า diff ทั้งหมด = 0)
    """

    deviation_per_zone:    dict[int, ZoneDeviation] = field(default_factory=dict)
    total_cycle_time:      float = 0.0
    total_standard_time:   float = 0.0
    total_cycle_time_diff: float = 0.0
    total_diff_pct:        float = 0.0
    similarity_score:      float = 0.0
    dtw_distance:          float = 0.0
    alignment_path:        AlignmentPath = field(default_factory=list)
    has_golden:            bool = False

    # ── Convenience properties ─────────────────────────────────────

    @property
    def is_over_threshold(self) -> bool:
        """True ถ้า zone ใด zone หนึ่งเกิน alert threshold"""
        return any(z.is_over_threshold for z in self.deviation_per_zone.values())

    @property
    def over_threshold_zones(self) -> list[int]:
        return [
            zid for zid, zd in self.deviation_per_zone.items()
            if zd.is_over_threshold
        ]

    def summary_line(self) -> str:
        sign = "+" if self.total_cycle_time_diff >= 0 else ""
        return (
            f"Total: {self.total_cycle_time:.2f}s "
            f"(std {self.total_standard_time:.2f}s, "
            f"diff {sign}{self.total_cycle_time_diff:.2f}s / "
            f"{sign}{self.total_diff_pct:.1f}%)  "
            f"DTW similarity: {self.similarity_score:.1f}"
        )

    def ghost_golden_idx(self, live_frame_idx: int) -> int:
        """
        แปลง live frame index → golden frame index ผ่าน alignment_path
        ใช้สำหรับ ghost overlay: หา position ของ ghost ณ frame นั้น

        ถ้าไม่มี alignment_path ให้ interpolate แบบ linear แทน
        """
        if not self.alignment_path:
            # Linear fallback
            n_live = max(live_frame_idx + 1, 1)
            ratio  = live_frame_idx / n_live
            return int(ratio * 199)   # golden resampled ที่ 200 จุด

        # หา entry แรกใน path ที่ live_idx >= live_frame_idx
        for li, gi in self.alignment_path:
            if li >= live_frame_idx:
                return gi
        # ถ้าเลยทั้งหมดแล้ว คืน index สุดท้ายของ golden
        return self.alignment_path[-1][1]


# ============================================================================
# ── GoldenReference ─────────────────────────────────────────────────────────
# ============================================================================

class GoldenReference:
    """
    ข้อมูล golden cycle ที่ผ่านการประมวลผลแล้ว

    Attributes
    ──────────
    standard_times       : {zone_id: median_seconds}  เวลา median ต่อ zone
    total_standard_time  : median ของ total cycle time
    zone_order           : ลำดับ zone ที่ถูกต้อง
    representative_record: CycleRecord ที่ใกล้ median ที่สุด
    trajectory           : ndarray (200, 2)  resampled จาก representative
    raw_trajectory       : list[dict]  trajectory ดิบของ representative cycle
    n_cycles_recorded    : จำนวน cycle ที่ใช้คำนวณ golden
    """

    def __init__(
        self,
        raw_trajectory:       list[dict],
        standard_times:       dict[int, float],
        total_standard_time:  float,
        zone_order:           list[int],
        n_cycles_recorded:    int = 0,
        representative_record=None,
        n_resample:           int = 200,
    ) -> None:
        self.raw_trajectory      = raw_trajectory
        self.standard_times      = standard_times
        self.total_standard_time = total_standard_time
        self.zone_order          = zone_order
        self.n_cycles_recorded   = n_cycles_recorded
        self.representative_record = representative_record
        self._n_resample         = n_resample

        arr = _to_array(raw_trajectory)
        self.trajectory: np.ndarray = _resample(arr, n_resample)

    # ── Ghost overlay helpers ──────────────────────────────────────

    def ghost_position_at_progress(self, progress_pct: float) -> tuple[float, float]:
        """
        คืน (x, y) ของ ghost ณ % progress ของ cycle (0–100)
        ใช้สำหรับ overlay แบบ progress-based (ไม่ผ่าน alignment)
        """
        if len(self.trajectory) == 0:
            return 0.0, 0.0
        idx = int(np.clip(
            progress_pct / 100.0 * (len(self.trajectory) - 1),
            0, len(self.trajectory) - 1,
        ))
        return float(self.trajectory[idx, 0]), float(self.trajectory[idx, 1])

    def ghost_position_at_frame(self, golden_frame_idx: int) -> tuple[float, float]:
        """
        คืน (x, y) จาก golden_frame_idx โดยตรง
        ใช้คู่กับ ComparisonResult.ghost_golden_idx()
        """
        idx = int(np.clip(golden_frame_idx, 0, len(self.trajectory) - 1))
        return float(self.trajectory[idx, 0]), float(self.trajectory[idx, 1])

    def __repr__(self) -> str:
        return (
            f"GoldenReference("
            f"n={self.n_cycles_recorded}, "
            f"total_std={self.total_standard_time:.2f}s, "
            f"zones={self.zone_order})"
        )


# ============================================================================
# ── GoldenCycleProcessor ─────────────────────────────────────────────────────
# ============================================================================

class GoldenCycleProcessor:
    """
    สร้าง GoldenReference จาก N cycle records ที่บันทึกไว้

    กระบวนการ
    ──────────
    1. คำนวณ median duration ต่อ zone จากทุก cycle
    2. คำนวณ median total cycle time
    3. เลือก "representative cycle" = cycle ที่มี total_time
       ใกล้ median ที่สุด  (ไม่ใช่ mean ของ trajectory ทุกอัน)
       เหตุผล: mean-trajectory อาจได้เส้นทาง "ลอยอยู่กลาง" ที่ไม่ตรงกับ
               การเคลื่อนที่จริงของผู้ปฏิบัติงานคนใด เลือก representative
               จริงๆ จาก corpus แทน
    4. ใช้ trajectory ของ representative cycle เป็น reference
    """

    N_RESAMPLE       = 200
    MIN_CYCLES       = 1    # อนุญาตให้ทำงานกับ cycle เดียวได้ (warn ถ้าน้อยกว่า 3)
    RECOMMENDED_MIN  = 3
    RECOMMENDED_MAX  = 5

    @staticmethod
    def process(
        cycle_records: list,         # list[CycleRecord]
        zone_order:    list[int] | None = None,
        alert_threshold_pct: int = 30,
    ) -> GoldenReference:
        """
        Parameters
        ──────────
        cycle_records        : list ของ CycleRecord จาก CycleTracker
        zone_order           : ลำดับ zone (default [1, 2, 3])
        alert_threshold_pct  : ใช้ตั้ง is_over_threshold ใน ZoneDeviation
                               (เก็บไว้ใน metadata เท่านั้น ณ จุดนี้)

        Returns
        ───────
        GoldenReference พร้อมใช้งาน
        """
        from core.cycle_tracker import CycleRecord  # avoid circular import

        zone_order = zone_order or [1, 2, 3]

        if not cycle_records:
            raise ValueError("ต้องมี cycle record อย่างน้อย 1 รายการ")

        if len(cycle_records) < GoldenCycleProcessor.RECOMMENDED_MIN:
            import warnings
            warnings.warn(
                f"Golden cycle ควรบันทึกอย่างน้อย "
                f"{GoldenCycleProcessor.RECOMMENDED_MIN} cycle "
                f"(มีแค่ {len(cycle_records)}) — median อาจไม่น่าเชื่อถือ",
                UserWarning,
                stacklevel=2,
            )

        # ── 1. per-zone median ────────────────────────────────────
        zone_time_lists: dict[int, list[float]] = {z: [] for z in zone_order}

        for rec in cycle_records:
            for zid, zt in rec.zone_timings.items():
                dur = zt.duration()
                if dur > 0 and zid in zone_time_lists:
                    zone_time_lists[zid].append(dur)

        standard_times: dict[int, float] = {}
        for zid, times in zone_time_lists.items():
            if times:
                standard_times[zid] = float(np.median(times))

        # ── 2. median total time ──────────────────────────────────
        valid_totals = [r.total_time for r in cycle_records if r.total_time > 0]
        median_total = float(np.median(valid_totals)) if valid_totals else 0.0

        # ── 3. เลือก representative cycle ────────────────────────
        # cycle ที่ total_time ใกล้ median_total ที่สุด
        rep_record = min(
            cycle_records,
            key=lambda r: abs(r.total_time - median_total),
        )

        raw_traj = rep_record.trajectory if rep_record.trajectory else []

        # ถ้า representative cycle ไม่มี trajectory ใช้ cycle แรกที่มี
        if not raw_traj:
            for r in cycle_records:
                if r.trajectory:
                    raw_traj = r.trajectory
                    break

        return GoldenReference(
            raw_trajectory      = raw_traj,
            standard_times      = standard_times,
            total_standard_time = median_total,
            zone_order          = zone_order,
            n_cycles_recorded   = len(cycle_records),
            representative_record = rep_record,
        )


# ============================================================================
# ── DTWComparator ────────────────────────────────────────────────────────────
# ============================================================================

class DTWComparator:
    """
    เปรียบเทียบ cycle จริงกับ GoldenReference โดยใช้ DTW

    Usage
    ─────
        comparator = DTWComparator(alert_threshold_pct=30)
        comparator.set_golden(golden)

        # เมื่อ cycle เสร็จ:
        result = comparator.compare(live_cycle_record)
        print(result.summary_line())

        # ทุก frame สำหรับ ghost overlay:
        live_idx  = len(live_frames_so_far) - 1
        gold_idx  = result.ghost_golden_idx(live_idx)
        gx, gy    = comparator.golden.ghost_position_at_frame(gold_idx)
    """

    _N_RESAMPLE = 200

    def __init__(self, alert_threshold_pct: int = 30) -> None:
        self._golden: Optional[GoldenReference] = None
        self._alert_threshold = alert_threshold_pct
        # เก็บ result ล่าสุดไว้ให้ ghost overlay ใช้ได้ทุก frame
        self._last_result: Optional[ComparisonResult] = None

    # ── Configuration ────────────────────────────────────────────

    def set_golden(self, golden: GoldenReference) -> None:
        self._golden      = golden
        self._last_result = None

    def has_golden(self) -> bool:
        return self._golden is not None

    @property
    def golden(self) -> Optional[GoldenReference]:
        return self._golden

    @property
    def last_result(self) -> Optional[ComparisonResult]:
        return self._last_result

    # ── Main comparison ──────────────────────────────────────────

    def compare(self, live_record) -> "ComparisonResult | float":
        """
        เปรียบเทียบ cycle จริงกับ golden

        Parameters
        ──────────
        live_record : CycleRecord จาก CycleTracker
                      หรือ list[dict] trajectory (legacy — คืน float similarity_score)

        Returns
        ───────
        ComparisonResult  — เมื่อรับ CycleRecord (API ใหม่)
        float             — เมื่อรับ list[dict] (legacy API สำหรับ backward compat)
        """
        # ── Legacy path: รับ list[dict] trajectory โดยตรง ───────────
        # (vision_thread.py เดิมเรียก compare(record.trajectory))
        if isinstance(live_record, list):
            return self._compare_trajectory_legacy(live_record)
        result = ComparisonResult(
            total_cycle_time  = live_record.total_time,
            total_standard_time = self._golden.total_standard_time if self._golden else 0.0,
            has_golden        = self._golden is not None,
        )

        if not self._golden:
            self._last_result = result
            return result

        golden = self._golden

        # ── Zone-level deviations ───────────────────────────────
        for zid in golden.zone_order:
            zt        = live_record.zone_timings.get(zid)
            actual    = zt.duration() if zt else 0.0
            standard  = golden.standard_times.get(zid, 0.0)
            diff_s    = actual - standard
            diff_pct  = (diff_s / standard * 100.0) if standard > 0 else 0.0
            over      = abs(diff_pct) > self._alert_threshold

            result.deviation_per_zone[zid] = ZoneDeviation(
                zone_id           = zid,
                actual_seconds    = actual,
                standard_seconds  = standard,
                diff_seconds      = diff_s,
                diff_pct          = diff_pct,
                is_over_threshold = over,
            )

        # ── Total time diff ─────────────────────────────────────
        std_total = golden.total_standard_time
        result.total_cycle_time_diff = live_record.total_time - std_total
        result.total_diff_pct = (
            result.total_cycle_time_diff / std_total * 100.0
            if std_total > 0 else 0.0
        )

        # ── DTW trajectory comparison ───────────────────────────
        live_traj = live_record.trajectory
        if live_traj and len(live_traj) >= 2:
            live_arr    = _to_array(live_traj)
            live_res    = _resample(live_arr, self._N_RESAMPLE)
            golden_res  = golden.trajectory   # already resampled

            # Normalise to [0,1]×[0,1] for shape comparison (not pixel-space)
            live_norm   = _normalise(live_res)
            golden_norm = _normalise(golden_res)

            dtw_dist, align_path = _dtw_compute(live_norm, golden_norm)

            result.dtw_distance    = dtw_dist
            result.alignment_path  = align_path
            result.similarity_score = self._score(dtw_dist)
        else:
            # ไม่มี trajectory → ใช้ linear alignment path แทน
            result.alignment_path   = [(i, i) for i in range(self._N_RESAMPLE)]
            result.similarity_score = 0.0
            result.dtw_distance     = 0.0

        self._last_result = result
        return result

    # ── Ghost overlay shortcuts ──────────────────────────────────

    def ghost_position_at_progress(self, progress_pct: float) -> tuple[float, float]:
        """
        คืน ghost (x, y) จาก % progress (0–100)
        ใช้ได้ตลอดเวลา — ไม่ต้องรอให้ cycle เสร็จ
        """
        if self._golden:
            return self._golden.ghost_position_at_progress(progress_pct)
        return 0.0, 0.0

    def ghost_at_live_frame(
        self,
        live_frame_idx: int,
        result: Optional[ComparisonResult] = None,
    ) -> tuple[float, float]:
        """
        คืน ghost (x, y) โดย map live_frame_idx → golden_frame_idx
        ผ่าน alignment_path (แม่นยำกว่า progress-based เพราะ DTW-aligned)

        ใช้ result ที่ส่งเข้ามา หรือ last_result ถ้าไม่ระบุ
        """
        if not self._golden:
            return 0.0, 0.0

        r = result or self._last_result
        if r is None or not r.alignment_path:
            # fallback: linear
            n = self._N_RESAMPLE
            ratio = min(live_frame_idx / max(n - 1, 1), 1.0)
            golden_idx = int(ratio * (n - 1))
        else:
            golden_idx = r.ghost_golden_idx(live_frame_idx)

        return self._golden.ghost_position_at_frame(golden_idx)

    # ── Internal helpers ─────────────────────────────────────────

    def _compare_trajectory_legacy(self, live_traj: list[dict]) -> float:
        """
        Legacy API: รับ trajectory list[dict] โดยตรง คืน similarity float 0-100
        เพื่อ backward compat กับ vision_thread.py เดิม
        """
        if not self._golden or not live_traj or len(live_traj) < 2:
            return 0.0
        live_arr    = _to_array(live_traj)
        live_res    = _resample(live_arr, self._N_RESAMPLE)
        golden_res  = self._golden.trajectory
        live_norm   = _normalise(live_res)
        golden_norm = _normalise(golden_res)
        dtw_dist, _ = _dtw_compute(live_norm, golden_norm)
        return self._score(dtw_dist)

    def _score(self, dtw_distance: float) -> float:
        """
        แปลง raw DTW distance → similarity score 0–100

        ใช้ exponential decay: score = 100 × exp(−λ × distance)
        โดย λ เลือกให้ distance = 0.5 (normalised units) → score ≈ 60
        นั่นคือ λ = −ln(0.6) / 0.5 ≈ 1.02
        """
        if dtw_distance <= 0:
            return 100.0
        lam   = -math.log(0.6) / 0.5
        score = 100.0 * math.exp(-lam * dtw_distance)
        return round(max(0.0, min(100.0, score)), 1)
