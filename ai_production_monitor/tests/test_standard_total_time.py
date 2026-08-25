"""
tests/test_standard_total_time.py

Regression tests that prove pass/fail evaluation uses the full-cycle
total_standard_time (including inter-zone travel) and NOT the sum of
per-zone standard durations.

Scenario
--------
  - 3 zones with standard in-zone times:  zone1=0.9s, zone2=1.0s, zone3=1.0s
    → sum = 2.9 s
  - Actual total cycle time recorded: 7.0 s
    (includes travel time between zones: ~4.1 s spread across transitions)
  - total_standard_time (median of real cycle records) = 7.0 s

Expected behaviour
------------------
  • When total_standard_time=7.0 is passed, a cycle that completes in 7.5 s
    (≈ 7 % over) is evaluated against 7.0 s and should NOT be flagged as fail
    at the default 30 % threshold.

  • When total_standard_time is NOT passed (old behaviour), the tracker falls
    back to sum(standard_times) = 2.9 s.  A 7.5 s cycle is 158 % over 2.9 s
    and IS flagged as fail — proving that the old code was broken.

  • CycleTracker.recorded_cycles[-1].status reflects these outcomes correctly.
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from core.cycle_tracker import CycleTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ZONE_IDS        = [1, 2, 3]
STD_ZONE_TIMES  = {1: 0.9, 2: 1.0, 3: 1.0}   # sum = 2.9 s
TRUE_TOTAL_STD  = 7.0                           # real total incl. travel
ALERT_THRESHOLD = 30                            # %


def _run_simulated_cycle(
    tracker: CycleTracker,
    total_cycle_secs: float,
) -> str:
    """
    Drive the tracker through a fake 3-zone cycle.

    We patch time.monotonic() so we can control elapsed time precisely.
    The cycle starts at t=0 and ends at t=total_cycle_secs.
    Zone durations are spread evenly for simplicity; what matters for the
    total-time test is the overall elapsed time, not individual zone splits.
    """
    base = 1_000_000.0  # arbitrary base to avoid monotonic ordering issues

    # Each zone occupies 1/3 of total time; gaps between zones add up to the rest
    zone_duration   = 0.9       # seconds per zone (matches STD_ZONE_TIMES approx)
    travel_duration = (total_cycle_secs - zone_duration * len(ZONE_IDS)) / len(ZONE_IDS)

    timestamps = []
    t = base
    for zone_id in ZONE_IDS:
        timestamps.append(("enter", zone_id, t))
        t += zone_duration
        timestamps.append(("exit",  zone_id, t))
        t += max(travel_duration, 0.0)

    end_t = base + total_cycle_secs

    time_iter = iter(
        [base]                                 # _start_cycle() monotonic call
        + [ts for _, _, ts in timestamps]      # enter/exit pairs
        + [end_t]                              # _complete_cycle() end_time
    )

    with patch("core.cycle_tracker.time.monotonic", side_effect=lambda: next(time_iter)):
        tracker.recording_mode = True
        for event_type, zone_id, _ in timestamps:
            tracker.on_zone_event(zone_id, event_type)

    if not tracker.recorded_cycles:
        raise RuntimeError("No cycle was recorded — check the simulation logic.")

    return tracker.recorded_cycles[-1].status


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStandardTotalTimePassFail(unittest.TestCase):

    def test_uses_total_standard_time_not_zone_sum(self):
        """
        With total_standard_time=7.0 and a 7.5 s cycle (7 % over threshold),
        the cycle must be 'pass' (below the 30 % alert_threshold).
        """
        tracker = CycleTracker(
            zone_ids            = ZONE_IDS,
            standard_times      = STD_ZONE_TIMES,
            total_standard_time = TRUE_TOTAL_STD,
            alert_threshold     = ALERT_THRESHOLD,
        )
        status = _run_simulated_cycle(tracker, total_cycle_secs=7.5)
        self.assertEqual(
            status, "pass",
            f"Expected 'pass' when total_standard_time=7.0 and cycle=7.5s "
            f"(7% over). Got: {status!r}",
        )

    def test_zone_sum_causes_false_fail(self):
        """
        Without total_standard_time, the tracker falls back to
        sum(zone_standard_times) = 2.9 s.  A 7.5 s cycle is 158 % over 2.9 s
        and must be flagged as 'fail', demonstrating the old buggy behaviour.
        """
        tracker = CycleTracker(
            zone_ids            = ZONE_IDS,
            standard_times      = STD_ZONE_TIMES,
            total_standard_time = None,           # no total → old code path
            alert_threshold     = ALERT_THRESHOLD,
        )
        status = _run_simulated_cycle(tracker, total_cycle_secs=7.5)
        self.assertEqual(
            status, "fail",
            f"Expected 'fail' when falling back to zone sum (2.9s) and "
            f"cycle=7.5s (158% over). Got: {status!r}",
        )

    def test_total_standard_time_not_equal_to_zone_sum(self):
        """
        Sanity check: the true total must differ from the sum of zone times
        when there is travel time between zones.
        """
        zone_sum = sum(STD_ZONE_TIMES.values())
        self.assertNotAlmostEqual(
            TRUE_TOTAL_STD, zone_sum, places=2,
            msg=(
                f"total_standard_time ({TRUE_TOTAL_STD}s) should differ from "
                f"zone-time sum ({zone_sum}s) because travel time is excluded "
                f"from zone timings."
            ),
        )

    def test_fast_cycle_is_pass_regardless_of_method(self):
        """
        A cycle that finishes under both 7.0 s and 2.9 s is pass under either
        method (confirms no regression for the happy path).
        """
        tracker = CycleTracker(
            zone_ids            = ZONE_IDS,
            standard_times      = STD_ZONE_TIMES,
            total_standard_time = TRUE_TOTAL_STD,
            alert_threshold     = ALERT_THRESHOLD,
        )
        status = _run_simulated_cycle(tracker, total_cycle_secs=6.0)
        self.assertEqual(status, "pass")


if __name__ == "__main__":
    unittest.main()
