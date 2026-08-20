"""
core/ghost_interpolator.py
Pure functions for computing ghost-hand position from a golden trajectory.

All functions are stateless and unit-testable without a GUI.

Trajectory point format (as recorded by CycleTracker.tick()):
    {"x": float, "y": float, "zone_id": int|None, "t_norm": float}
    where t_norm ∈ [0, 1] — fraction of the golden cycle's total_time.

Coordinates are in *camera-frame pixel space* (same units as the
TriggerPoint.x / .y values).  The caller is responsible for scaling
them to widget pixels before painting.
"""

from __future__ import annotations

from typing import NamedTuple


class GhostState(NamedTuple):
    """Result of a single ghost-position query."""
    x: float          # frame-pixel x
    y: float          # frame-pixel y
    t_norm: float     # fraction through golden cycle [0, 1]
    clamped: bool     # True when elapsed > golden_total_time (ghost at end)


def interpolate_ghost_position(
    trajectory: list[dict],
    elapsed: float,
    golden_total_time: float,
) -> GhostState | None:
    """
    Compute the ghost hand position at *elapsed* seconds into the current cycle
    using linear interpolation over the golden trajectory.

    Parameters
    ----------
    trajectory        : list of {"x", "y", "t_norm", ...} dicts from GoldenReference.raw_trajectory
    elapsed           : seconds since current cycle started (from CycleTracker.cycle_elapsed)
    golden_total_time : total duration (seconds) of the golden cycle
                        (GoldenReference.total_standard_time)

    Returns
    -------
    GhostState or None if trajectory is empty / invalid.
    """
    if not trajectory or golden_total_time <= 0:
        return None

    # Normalise elapsed into [0, 1]
    clamped = elapsed > golden_total_time
    target_t = min(elapsed / golden_total_time, 1.0)

    # Build a sorted list of (t_norm, x, y) — trajectory is already in order
    # but guard against missing t_norm keys (legacy data recorded without it)
    pts = []
    for pt in trajectory:
        t = pt.get("t_norm")
        if t is None:
            continue
        pts.append((float(t), float(pt["x"]), float(pt["y"])))

    if not pts:
        return None

    # Edge cases: before first point or after last point
    if target_t <= pts[0][0]:
        t0, x0, y0 = pts[0]
        return GhostState(x=x0, y=y0, t_norm=t0, clamped=clamped)
    if target_t >= pts[-1][0]:
        t_last, x_last, y_last = pts[-1]
        return GhostState(x=x_last, y=y_last, t_norm=t_last, clamped=clamped)

    # Binary-search for the bracketing pair [i, i+1]
    lo, hi = 0, len(pts) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if pts[mid][0] <= target_t:
            lo = mid
        else:
            hi = mid

    t0, x0, y0 = pts[lo]
    t1, x1, y1 = pts[hi]

    # Linear interpolation
    span = t1 - t0
    if span < 1e-9:
        # Coincident timestamps — return the first
        return GhostState(x=x0, y=y0, t_norm=t0, clamped=clamped)

    alpha = (target_t - t0) / span
    x_interp = x0 + alpha * (x1 - x0)
    y_interp = y0 + alpha * (y1 - y0)

    return GhostState(x=x_interp, y=y_interp, t_norm=target_t, clamped=clamped)


def scale_to_widget(
    ghost: GhostState,
    frame_w: int,
    frame_h: int,
    widget_w: int,
    widget_h: int,
) -> tuple[float, float]:
    """
    Convert frame-pixel (x, y) → widget-pixel (x, y) using the same
    letterbox transform that VideoWidget.paintEvent applies.

    Returns the widget-space (x, y) tuple.
    """
    if frame_w <= 0 or frame_h <= 0:
        return ghost.x, ghost.y

    # Compute scale factor and letterbox offset (same logic as VideoWidget)
    scale = min(widget_w / frame_w, widget_h / frame_h)
    scaled_w = frame_w * scale
    scaled_h = frame_h * scale
    off_x = (widget_w - scaled_w) / 2.0
    off_y = (widget_h - scaled_h) / 2.0

    wx = off_x + ghost.x * scale
    wy = off_y + ghost.y * scale
    return wx, wy
