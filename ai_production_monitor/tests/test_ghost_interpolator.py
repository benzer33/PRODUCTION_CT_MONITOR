"""
tests/test_ghost_interpolator.py
Unit tests for core/ghost_interpolator.py

Run with:  pytest ai_production_monitor/tests/test_ghost_interpolator.py -v
"""

from __future__ import annotations

import sys
import os

# Allow running from repo root or from the tests directory
_PKG_DIR = os.path.join(os.path.dirname(__file__), "..")
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

import pytest
from core.ghost_interpolator import (
    GhostState,
    interpolate_ghost_position,
    scale_to_widget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_traj(points: list[tuple[float, float, float]]) -> list[dict]:
    """Build trajectory list from [(t_norm, x, y), ...]."""
    return [{"t_norm": t, "x": x, "y": y, "zone_id": 1} for t, x, y in points]


# ---------------------------------------------------------------------------
# interpolate_ghost_position
# ---------------------------------------------------------------------------

class TestInterpolateGhostPosition:

    def test_empty_trajectory_returns_none(self):
        assert interpolate_ghost_position([], elapsed=1.0, golden_total_time=5.0) is None

    def test_zero_golden_time_returns_none(self):
        traj = _make_traj([(0.0, 100.0, 200.0)])
        assert interpolate_ghost_position(traj, elapsed=1.0, golden_total_time=0.0) is None

    def test_exact_first_point(self):
        traj = _make_traj([(0.0, 50.0, 100.0), (0.5, 150.0, 200.0), (1.0, 250.0, 300.0)])
        result = interpolate_ghost_position(traj, elapsed=0.0, golden_total_time=10.0)
        assert result is not None
        assert result.x == pytest.approx(50.0)
        assert result.y == pytest.approx(100.0)
        assert result.clamped is False

    def test_exact_last_point(self):
        traj = _make_traj([(0.0, 50.0, 100.0), (0.5, 150.0, 200.0), (1.0, 250.0, 300.0)])
        result = interpolate_ghost_position(traj, elapsed=10.0, golden_total_time=10.0)
        assert result is not None
        assert result.x == pytest.approx(250.0)
        assert result.y == pytest.approx(300.0)

    def test_midpoint_interpolation(self):
        """At elapsed=5s with golden=10s → t_norm=0.5 → halfway between first and last."""
        traj = _make_traj([(0.0, 0.0, 0.0), (1.0, 100.0, 200.0)])
        result = interpolate_ghost_position(traj, elapsed=5.0, golden_total_time=10.0)
        assert result is not None
        assert result.x == pytest.approx(50.0, abs=0.01)
        assert result.y == pytest.approx(100.0, abs=0.01)

    def test_quarter_interpolation(self):
        """t_norm=0.25 → 25% of the way between two points."""
        traj = _make_traj([(0.0, 0.0, 0.0), (1.0, 400.0, 800.0)])
        result = interpolate_ghost_position(traj, elapsed=2.5, golden_total_time=10.0)
        assert result is not None
        assert result.x == pytest.approx(100.0, abs=0.01)
        assert result.y == pytest.approx(200.0, abs=0.01)

    def test_three_segment_trajectory(self):
        """Five-point trajectory; interpolate between segments 2→3."""
        traj = _make_traj([
            (0.0,  0.0,   0.0),
            (0.25, 100.0, 50.0),
            (0.50, 200.0, 100.0),
            (0.75, 300.0, 150.0),
            (1.00, 400.0, 200.0),
        ])
        # elapsed=6.25s → t_norm=0.625 → halfway between 0.5 and 0.75
        result = interpolate_ghost_position(traj, elapsed=6.25, golden_total_time=10.0)
        assert result is not None
        assert result.x == pytest.approx(250.0, abs=0.5)
        assert result.y == pytest.approx(125.0, abs=0.5)

    def test_elapsed_beyond_golden_clamps(self):
        """elapsed > golden_total_time → ghost stops at last point, clamped=True."""
        traj = _make_traj([(0.0, 0.0, 0.0), (1.0, 300.0, 400.0)])
        result = interpolate_ghost_position(traj, elapsed=15.0, golden_total_time=10.0)
        assert result is not None
        assert result.clamped is True
        assert result.x == pytest.approx(300.0)
        assert result.y == pytest.approx(400.0)

    def test_no_t_norm_key_skips_points(self):
        """Points missing t_norm should be skipped gracefully."""
        traj = [
            {"x": 0.0, "y": 0.0, "zone_id": 1},           # no t_norm → skip
            {"x": 100.0, "y": 200.0, "zone_id": 1, "t_norm": 0.5},
            {"x": 200.0, "y": 400.0, "zone_id": 1, "t_norm": 1.0},
        ]
        result = interpolate_ghost_position(traj, elapsed=5.0, golden_total_time=10.0)
        assert result is not None
        assert result.x == pytest.approx(100.0)  # at t_norm=0.5

    def test_single_point_trajectory(self):
        """Only one valid point → always return that point."""
        traj = _make_traj([(0.5, 77.0, 88.0)])
        result = interpolate_ghost_position(traj, elapsed=3.0, golden_total_time=10.0)
        assert result is not None
        assert result.x == pytest.approx(77.0)
        assert result.y == pytest.approx(88.0)


# ---------------------------------------------------------------------------
# scale_to_widget
# ---------------------------------------------------------------------------

class TestScaleToWidget:

    def _ghost(self, x, y):
        return GhostState(x=x, y=y, t_norm=0.5, clamped=False)

    def test_no_letterbox_square(self):
        """Frame and widget same size → 1:1 mapping."""
        wx, wy = scale_to_widget(self._ghost(100.0, 200.0), 640, 480, 640, 480)
        assert wx == pytest.approx(100.0, abs=0.5)
        assert wy == pytest.approx(200.0, abs=0.5)

    def test_letterbox_centre_offset(self):
        """640×480 frame in 640×360 widget → pillarbox on top/bottom."""
        # scale = min(640/640, 360/480) = 0.75
        # scaled_h = 480 * 0.75 = 360; scaled_w = 640 * 0.75 = 480
        # off_x = (640 - 480) / 2 = 80; off_y = 0
        wx, wy = scale_to_widget(self._ghost(0.0, 0.0), 640, 480, 640, 360)
        assert wx == pytest.approx(80.0, abs=0.5)
        assert wy == pytest.approx(0.0, abs=0.5)

    def test_scale_up(self):
        """Small frame displayed in large widget."""
        # frame 320×240 → widget 640×480: scale=2.0, no letterbox
        wx, wy = scale_to_widget(self._ghost(100.0, 50.0), 320, 240, 640, 480)
        assert wx == pytest.approx(200.0, abs=0.5)
        assert wy == pytest.approx(100.0, abs=0.5)

    def test_zero_frame_size_returns_unchanged(self):
        wx, wy = scale_to_widget(self._ghost(50.0, 60.0), 0, 0, 640, 480)
        assert wx == pytest.approx(50.0)
        assert wy == pytest.approx(60.0)
