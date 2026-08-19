"""
core/cycle_tracker.py
Production-cycle state machine.

State machine
-------------
IDLE
  │  hand enters Zone 1
  ▼
ZONE_1_ACTIVE   (timing Zone 1)
  │  hand exits Zone 1
  ▼
TRANSIT_1_2
  │  hand enters Zone 2
  ▼
ZONE_2_ACTIVE   (timing Zone 2)
  │  hand exits Zone 2
  ▼
TRANSIT_2_3
  │  hand enters Zone 3
  ▼
ZONE_3_ACTIVE   (timing Zone 3)
  │  hand exits Zone 3  →  cycle complete!
  ▼
IDLE (next cycle starts immediately)

Sequence violations are detected if the hand enters a zone out of order.
The tracker emits Python callbacks (used by the QThread wrapper to emit Signals).

Features
--------
- Per-zone elapsed time tracking in real-time
- Threshold alerting: fires alert_callback when zone time exceeds threshold %
- Sequence-violation detection
- Golden-cycle trajectory recording (list of {x, y, zone_id, t_norm})
- Live progress % calculation (for ghost overlay sync)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Enums / constants
# ---------------------------------------------------------------------------

class CycleState(Enum):
    IDLE          = auto()
    ZONE_1_ACTIVE = auto()
    TRANSIT_1_2   = auto()
    ZONE_2_ACTIVE = auto()
    TRANSIT_2_3   = auto()
    ZONE_3_ACTIVE = auto()
    COMPLETED     = auto()


# Expected zone visit order
ZONE_SEQUENCE = [1, 2, 3]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ZoneTiming:
    zone_id: int
    enter_time: float = 0.0
    exit_time: float  = 0.0
    elapsed: float    = 0.0   # updated in real-time before exit too

    def duration(self) -> float:
        if self.exit_time:
            return self.exit_time - self.enter_time
        return 0.0


@dataclass
class CycleRecord:
    """Data captured for a single completed cycle."""
    cycle_number: int
    start_time: float
    end_time: float
    zone_timings: dict[int, ZoneTiming] = field(default_factory=dict)
    trajectory: list[dict]              = field(default_factory=list)
    # {x, y, zone_id, t_norm} — t_norm in [0,1] of cycle duration
    sequence_errors: list[str]          = field(default_factory=list)
    status: str = "pass"   # pass | fail | sequence_error

    @property
    def total_time(self) -> float:
        return self.end_time - self.start_time

    def zone_times_dict(self) -> dict[str, float]:
        return {str(zid): zt.duration() for zid, zt in self.zone_timings.items()}


# ---------------------------------------------------------------------------
# CycleTracker
# ---------------------------------------------------------------------------

class CycleTracker:
    """
    Core state machine for one workstation.

    All callbacks are optional and called synchronously from the
    vision thread — keep them lightweight (emit Qt signals, don't do GUI work).

    Parameters
    ----------
    zone_ids          : ordered list of zone IDs that form one cycle (default [1,2,3])
    alert_threshold   : % over standard time that triggers alert (e.g. 30 = 130%)
    standard_times    : {zone_id: seconds} from golden cycle (can be set later)
    on_cycle_complete : called with CycleRecord when a cycle finishes
    on_zone_enter     : called with (zone_id, elapsed_in_zone) on zone entry
    on_zone_exit      : called with (zone_id, duration_in_zone) on zone exit
    on_alert          : called with (zone_id, message) when threshold exceeded
    on_sequence_error : called with (expected_zone, actual_zone) on violation
    on_state_change   : called with new CycleState on any state transition
    """

    def __init__(
        self,
        zone_ids: list[int] = None,
        alert_threshold: int = 30,
        standard_times: dict[int, float] | None = None,
        on_cycle_complete: Callable[[CycleRecord], None] | None = None,
        on_zone_enter: Callable[[int, float], None] | None = None,
        on_zone_exit: Callable[[int, float], None] | None = None,
        on_alert: Callable[[int, str], None] | None = None,
        on_sequence_error: Callable[[int, int], None] | None = None,
        on_state_change: Callable[[CycleState], None] | None = None,
    ) -> None:
        self._zone_sequence: list[int] = zone_ids or ZONE_SEQUENCE
        self._alert_threshold          = alert_threshold        # %
        self._standard_times: dict[int, float] = standard_times or {}

        # Callbacks
        self._on_cycle_complete  = on_cycle_complete
        self._on_zone_enter      = on_zone_enter
        self._on_zone_exit       = on_zone_exit
        self._on_alert           = on_alert
        self._on_sequence_error  = on_sequence_error
        self._on_state_change    = on_state_change

        # Runtime state
        self._state              = CycleState.IDLE
        self._cycle_number       = 0
        self._cycle_start: float = 0.0
        self._current_zone_idx   = 0       # index into _zone_sequence
        self._zone_timings: dict[int, ZoneTiming] = {}
        self._trajectory: list[dict]  = []
        self._sequence_errors: list[str] = []
        self._current_zone_id: int | None = None

        # Alert cooldown — don't fire repeatedly for same zone
        self._alerted_zones: set[int] = set()

        # Recording mode (golden cycle)
        self.recording_mode: bool = False
        self.recorded_cycles: list[CycleRecord] = []

    # ------------------------------------------------------------------
    # Configuration setters
    # ------------------------------------------------------------------

    def set_standard_times(self, times: dict[int, float]) -> None:
        self._standard_times = times

    def set_alert_threshold(self, pct: int) -> None:
        self._alert_threshold = pct

    def reset(self) -> None:
        """Full reset — call when stopping a session."""
        self._state            = CycleState.IDLE
        self._cycle_number     = 0
        self._cycle_start      = 0.0
        self._current_zone_idx = 0
        self._zone_timings     = {}
        self._trajectory       = []
        self._sequence_errors  = []
        self._current_zone_id  = None
        self._alerted_zones    = set()
        self.recorded_cycles   = []

    # ------------------------------------------------------------------
    # Public event handlers (called by vision thread per frame)
    # ------------------------------------------------------------------

    def on_zone_event(self, zone_id: int, event_type: str) -> None:
        """
        Process a zone entry or exit event.

        Parameters
        ----------
        zone_id    : which zone
        event_type : "enter" | "exit"
        """
        if event_type == "enter":
            self._handle_enter(zone_id)
        elif event_type == "exit":
            self._handle_exit(zone_id)

    def tick(self, hand_x: float, hand_y: float) -> None:
        """
        Call every frame with current hand position.
        Used to:
          1. Record trajectory points for golden cycle
          2. Check real-time threshold alerts for the active zone
        """
        if self._state == CycleState.IDLE:
            return

        now = time.monotonic()

        # Update current zone elapsed time (real-time, before exit)
        if self._current_zone_id is not None:
            zt = self._zone_timings.get(self._current_zone_id)
            if zt:
                zt.elapsed = now - zt.enter_time
                self._check_threshold(self._current_zone_id, zt.elapsed)

        # Record trajectory
        if self._cycle_start > 0:
            t_abs = now - self._cycle_start
            self._trajectory.append({
                "x":      hand_x,
                "y":      hand_y,
                "zone_id": self._current_zone_id,
                "t_abs":  t_abs,
            })

    # ------------------------------------------------------------------
    # Internal state-machine handlers
    # ------------------------------------------------------------------

    def _handle_enter(self, zone_id: int) -> None:
        expected_idx = self._current_zone_idx
        expected_zone = (
            self._zone_sequence[expected_idx]
            if expected_idx < len(self._zone_sequence)
            else None
        )

        # ---- IDLE → start of new cycle on Zone 1 entry ----
        if self._state == CycleState.IDLE:
            if zone_id == self._zone_sequence[0]:
                self._start_cycle()
                self._enter_zone(zone_id)
            else:
                # Entered a zone other than Zone 1 — sequence error
                self._record_sequence_error(
                    expected=self._zone_sequence[0],
                    actual=zone_id,
                )
            return

        # ---- Already in a cycle ----
        if zone_id != expected_zone:
            self._record_sequence_error(expected=expected_zone, actual=zone_id)
            # Still track the zone so timing doesn't break entirely
        self._enter_zone(zone_id)

    def _handle_exit(self, zone_id: int) -> None:
        if zone_id not in self._zone_timings:
            return  # spurious exit before enter

        zt = self._zone_timings[zone_id]
        zt.exit_time = time.monotonic()
        zt.elapsed   = zt.duration()
        self._current_zone_id = None

        if self._on_zone_exit:
            self._on_zone_exit(zone_id, zt.elapsed)

        # Advance pointer
        try:
            idx = self._zone_sequence.index(zone_id)
        except ValueError:
            return

        self._current_zone_idx = idx + 1

        # Check if cycle is complete (exited the LAST zone)
        if self._current_zone_idx >= len(self._zone_sequence):
            self._complete_cycle()
        else:
            # Update state to TRANSIT
            next_state = self._transit_state_for_zone_idx(self._current_zone_idx)
            self._set_state(next_state)

    def _start_cycle(self) -> None:
        self._cycle_number    += 1
        self._cycle_start      = time.monotonic()
        self._current_zone_idx = 0
        self._zone_timings     = {}
        self._trajectory       = []
        self._sequence_errors  = []
        self._alerted_zones    = set()
        self._set_state(CycleState.ZONE_1_ACTIVE)

    def _enter_zone(self, zone_id: int) -> None:
        now = time.monotonic()
        zt = ZoneTiming(zone_id=zone_id, enter_time=now)
        self._zone_timings[zone_id] = zt
        self._current_zone_id = zone_id

        state_for_zone = {
            self._zone_sequence[0]: CycleState.ZONE_1_ACTIVE,
            self._zone_sequence[1]: CycleState.ZONE_2_ACTIVE,
            self._zone_sequence[2]: CycleState.ZONE_3_ACTIVE,
        }
        new_state = state_for_zone.get(zone_id, self._state)
        self._set_state(new_state)

        if self._on_zone_enter:
            self._on_zone_enter(zone_id, 0.0)

    def _complete_cycle(self) -> None:
        end_time = time.monotonic()
        self._set_state(CycleState.COMPLETED)

        # Build CycleRecord
        has_seq_error = bool(self._sequence_errors)
        status = "sequence_error" if has_seq_error else "pass"

        # Check total time deviation
        total = end_time - self._cycle_start
        std_total = sum(self._standard_times.values()) if self._standard_times else 0
        if std_total > 0:
            dev_pct = ((total - std_total) / std_total) * 100
            if dev_pct > self._alert_threshold:
                status = "fail"

        record = CycleRecord(
            cycle_number   = self._cycle_number,
            start_time     = self._cycle_start,
            end_time       = end_time,
            zone_timings   = dict(self._zone_timings),
            trajectory     = list(self._trajectory),
            sequence_errors = list(self._sequence_errors),
            status         = status,
        )

        # Normalise trajectory t values to [0,1]
        if record.trajectory and record.total_time > 0:
            for pt in record.trajectory:
                pt["t_norm"] = pt["t_abs"] / record.total_time

        if self.recording_mode:
            self.recorded_cycles.append(record)

        if self._on_cycle_complete:
            self._on_cycle_complete(record)

        # Reset for next cycle
        self._state            = CycleState.IDLE
        self._current_zone_idx = 0
        self._zone_timings     = {}
        self._trajectory       = []
        self._sequence_errors  = []
        self._alerted_zones    = set()
        self._current_zone_id  = None

    def _record_sequence_error(self, expected: int | None, actual: int) -> None:
        msg = f"Expected Zone {expected}, got Zone {actual}"
        self._sequence_errors.append(msg)
        if self._on_sequence_error and expected is not None:
            self._on_sequence_error(expected, actual)

    def _check_threshold(self, zone_id: int, elapsed: float) -> None:
        if zone_id in self._alerted_zones:
            return
        std = self._standard_times.get(zone_id)
        if not std or std <= 0:
            return
        over_pct = ((elapsed - std) / std) * 100
        if over_pct > self._alert_threshold:
            self._alerted_zones.add(zone_id)
            msg = (
                f"Zone {zone_id} exceeded threshold: "
                f"{elapsed:.1f}s vs standard {std:.1f}s "
                f"(+{over_pct:.0f}%)"
            )
            if self._on_alert:
                self._on_alert(zone_id, msg)

    def _set_state(self, state: CycleState) -> None:
        if self._state != state:
            self._state = state
            if self._on_state_change:
                self._on_state_change(state)

    @staticmethod
    def _transit_state_for_zone_idx(next_zone_idx: int) -> CycleState:
        mapping = {
            1: CycleState.TRANSIT_1_2,
            2: CycleState.TRANSIT_2_3,
        }
        return mapping.get(next_zone_idx, CycleState.IDLE)

    # ------------------------------------------------------------------
    # Read-only properties (for GUI polling / rendering)
    # ------------------------------------------------------------------

    @property
    def state(self) -> CycleState:
        return self._state

    @property
    def cycle_number(self) -> int:
        return self._cycle_number

    @property
    def cycle_elapsed(self) -> float:
        """Seconds since current cycle started (0 if IDLE)."""
        if self._cycle_start == 0 or self._state == CycleState.IDLE:
            return 0.0
        return time.monotonic() - self._cycle_start

    @property
    def current_zone_elapsed(self) -> float:
        """Seconds hand has been in current zone (real-time)."""
        if self._current_zone_id is None:
            return 0.0
        zt = self._zone_timings.get(self._current_zone_id)
        if zt:
            return time.monotonic() - zt.enter_time
        return 0.0

    @property
    def zone_timings(self) -> dict[int, ZoneTiming]:
        return dict(self._zone_timings)

    @property
    def current_zone_id(self) -> int | None:
        return self._current_zone_id

    def cycle_progress_pct(self) -> float:
        """
        Estimate % completion of the current cycle based on elapsed time
        vs golden standard total.  Used to sync ghost overlay.
        """
        std_total = sum(self._standard_times.values()) if self._standard_times else 0
        if std_total <= 0 or self._state == CycleState.IDLE:
            return 0.0
        return min(100.0, (self.cycle_elapsed / std_total) * 100.0)

    def get_zone_status(self) -> dict[int, dict]:
        """
        Return live status for all zones — used by GUI panel.
        {zone_id: {elapsed, standard, over_pct, is_active}}
        """
        result = {}
        for zid in self._zone_sequence:
            zt    = self._zone_timings.get(zid)
            std   = self._standard_times.get(zid)
            if zt:
                elapsed  = zt.elapsed if zt.exit_time else (
                    time.monotonic() - zt.enter_time
                )
            else:
                elapsed  = 0.0
            over_pct = ((elapsed - std) / std * 100) if (std and elapsed > 0) else 0.0
            result[zid] = {
                "elapsed":   elapsed,
                "standard":  std or 0.0,
                "over_pct":  over_pct,
                "is_active": zid == self._current_zone_id,
                "completed": zt is not None and zt.exit_time > 0,
            }
        return result
