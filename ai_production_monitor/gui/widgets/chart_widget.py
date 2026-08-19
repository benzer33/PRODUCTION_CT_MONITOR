"""
gui/widgets/chart_widget.py
Embedded Matplotlib charts for the AI Summary screen.

Provides
--------
- CycleTimeChart   : bar chart of cycle times vs standard
- ZoneBreakdownChart : stacked bar showing zone time contribution per cycle
- DeviationTrendChart: line chart of % deviation over time
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QSizePolicy, QWidget, QVBoxLayout


# Dark theme colours matching the industrial GUI
_BG       = "#0d1b2a"
_FG       = "#cfd8dc"
_GRID     = "#1e3040"
_PASS     = "#00c853"
_FAIL     = "#d50000"
_WARN     = "#ffab00"
_STANDARD = "#00bcd4"
_ACCENT   = "#1565c0"


def _apply_dark_axes(ax, fig) -> None:
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_FG, labelsize=8)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.5, linestyle="--")


# ---------------------------------------------------------------------------
# Base canvas
# ---------------------------------------------------------------------------

class _BaseChart(FigureCanvas):
    def __init__(self, figsize=(6, 3), dpi=90) -> None:
        self.fig = Figure(figsize=figsize, dpi=dpi, tight_layout=True)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #0d1b2a;")


# ---------------------------------------------------------------------------
# Cycle Time Chart
# ---------------------------------------------------------------------------

class CycleTimeChart(_BaseChart):
    """
    Bar chart: one bar per cycle coloured by pass/fail.
    Horizontal dashed line for the golden standard.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(figsize=(8, 3))
        self._ax = self.fig.add_subplot(111)

    def plot(
        self,
        cycle_times: list[float],
        standard_time: float,
        statuses: list[str] | None = None,
    ) -> None:
        self._ax.clear()
        _apply_dark_axes(self._ax, self.fig)

        n  = len(cycle_times)
        xs = list(range(1, n + 1))

        if statuses and len(statuses) == n:
            colors = [
                _PASS if s == "pass"
                else _FAIL if s in ("fail", "sequence_error")
                else _WARN
                for s in statuses
            ]
        else:
            colors = [
                _PASS if t <= standard_time * 1.05 else
                _WARN  if t <= standard_time * 1.30 else
                _FAIL
                for t in cycle_times
            ]

        bars = self._ax.bar(xs, cycle_times, color=colors, alpha=0.85, zorder=3)

        if standard_time > 0:
            self._ax.axhline(
                standard_time,
                color=_STANDARD, linewidth=1.5,
                linestyle="--", label=f"Standard {standard_time:.1f}s",
                zorder=4,
            )
            self._ax.legend(
                facecolor=_BG, edgecolor=_GRID,
                labelcolor=_FG, fontsize=8,
            )

        self._ax.set_xlabel("Cycle #", fontsize=9)
        self._ax.set_ylabel("Time (s)", fontsize=9)
        self._ax.set_title("Cycle Times vs Standard", fontsize=10)
        self._ax.set_xticks(xs)
        self.draw()


# ---------------------------------------------------------------------------
# Zone Breakdown Chart (stacked bar)
# ---------------------------------------------------------------------------

class ZoneBreakdownChart(_BaseChart):
    """Stacked bar chart showing per-zone time contribution per cycle."""

    ZONE_COLORS = ["#1565c0", "#00838f", "#558b2f", "#6a1b9a"]

    def __init__(self, parent=None) -> None:
        super().__init__(figsize=(8, 3))
        self._ax = self.fig.add_subplot(111)

    def plot(
        self,
        cycle_zone_times: list[dict[str, float]],
        zone_names: dict[str, str] | None = None,
        standard_times: dict[str, float] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        cycle_zone_times : list of {zone_id_str: seconds} per cycle
        zone_names       : {zone_id_str: name}
        standard_times   : {zone_id_str: seconds}
        """
        self._ax.clear()
        _apply_dark_axes(self._ax, self.fig)

        if not cycle_zone_times:
            self.draw()
            return

        all_zones = sorted(
            {z for d in cycle_zone_times for z in d.keys()},
            key=lambda z: int(z) if z.isdigit() else z
        )
        n      = len(cycle_zone_times)
        xs     = list(range(1, n + 1))
        bottom = np.zeros(n)

        for i, zid in enumerate(all_zones):
            values = [d.get(zid, 0.0) for d in cycle_zone_times]
            color  = self.ZONE_COLORS[i % len(self.ZONE_COLORS)]
            label  = zone_names.get(zid, f"Zone {zid}") if zone_names else f"Zone {zid}"
            self._ax.bar(
                xs, values, bottom=bottom,
                color=color, alpha=0.85, label=label, zorder=3,
            )
            bottom += np.array(values)

        self._ax.set_xlabel("Cycle #", fontsize=9)
        self._ax.set_ylabel("Time (s)", fontsize=9)
        self._ax.set_title("Zone Time Breakdown per Cycle", fontsize=10)
        self._ax.set_xticks(xs)
        self._ax.legend(
            facecolor=_BG, edgecolor=_GRID,
            labelcolor=_FG, fontsize=8,
            loc="upper right",
        )
        self.draw()


# ---------------------------------------------------------------------------
# Deviation Trend Chart
# ---------------------------------------------------------------------------

class DeviationTrendChart(_BaseChart):
    """Line chart of % deviation from standard over cycle sequence."""

    def __init__(self, parent=None) -> None:
        super().__init__(figsize=(8, 2.5))
        self._ax = self.fig.add_subplot(111)

    def plot(
        self,
        deviations: list[float],
        warning_pct: float = 15.0,
        critical_pct: float = 30.0,
    ) -> None:
        self._ax.clear()
        _apply_dark_axes(self._ax, self.fig)

        if not deviations:
            self.draw()
            return

        xs = list(range(1, len(deviations) + 1))
        self._ax.plot(xs, deviations, color=_ACCENT, linewidth=1.5,
                      marker="o", markersize=4, zorder=3)

        self._ax.axhline(0,            color=_PASS,     linewidth=1, linestyle="-",  alpha=0.5)
        self._ax.axhline(warning_pct,  color=_WARN,     linewidth=1, linestyle="--", alpha=0.7,
                         label=f"Warning +{warning_pct:.0f}%")
        self._ax.axhline(critical_pct, color=_FAIL,     linewidth=1, linestyle="--", alpha=0.7,
                         label=f"Critical +{critical_pct:.0f}%")

        self._ax.fill_between(xs, deviations, 0,
                              where=[d > 0 for d in deviations],
                              alpha=0.15, color=_FAIL, interpolate=True)
        self._ax.fill_between(xs, deviations, 0,
                              where=[d <= 0 for d in deviations],
                              alpha=0.15, color=_PASS, interpolate=True)

        self._ax.set_xlabel("Cycle #", fontsize=9)
        self._ax.set_ylabel("Deviation (%)", fontsize=9)
        self._ax.set_title("Cycle Time Deviation Trend", fontsize=10)
        self._ax.set_xticks(xs)
        self._ax.legend(
            facecolor=_BG, edgecolor=_GRID,
            labelcolor=_FG, fontsize=8,
        )
        self.draw()
