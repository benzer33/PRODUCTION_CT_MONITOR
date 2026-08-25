"""
tests/test_camera_mirror.py

Regression tests for the camera mirror/flip feature.

Key assertion
-------------
When CameraManager is created with mirror_horizontal=True, every frame
returned by read() must be horizontally flipped relative to the raw frame
coming from the capture device.  This guarantees that hand-detection
coordinates from MediaPipe and the displayed image are always aligned:
the flip is applied at one single point (CameraManager.read) before any
downstream consumer ever sees the frame.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from vision.camera_handler import CameraManager, WebcamSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_asymmetric_frame(w: int = 64, h: int = 48) -> np.ndarray:
    """
    Create a frame whose left and right halves are visually distinct,
    so a horizontal flip is unambiguous.
    """
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, : w // 2, 0] = 200   # left half is red
    frame[:, w // 2 :, 2] = 200   # right half is blue
    return frame


def _make_manager(mirror: bool) -> CameraManager:
    source = WebcamSource(device_index=0)
    return CameraManager(source=source, width=64, height=48, mirror_horizontal=mirror)


def _inject_open_cap(mgr: CameraManager, raw_frame: np.ndarray) -> None:
    """Patch the internal cap so read() returns raw_frame without opening a real camera."""
    cap_mock = MagicMock()
    cap_mock.isOpened.return_value = True
    cap_mock.read.return_value = (True, raw_frame.copy())
    mgr._cap      = cap_mock
    mgr._is_open  = True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCameraMirror(unittest.TestCase):

    def test_mirror_off_returns_original_frame(self):
        """With mirror_horizontal=False the frame is returned unchanged."""
        raw = _make_asymmetric_frame()
        mgr = _make_manager(mirror=False)
        _inject_open_cap(mgr, raw)

        ok, frame = mgr.read()
        self.assertTrue(ok)
        np.testing.assert_array_equal(
            frame, raw,
            err_msg="Frame should be identical to raw when mirror is off.",
        )

    def test_mirror_on_flips_frame_horizontally(self):
        """With mirror_horizontal=True the frame must be a horizontal mirror of raw."""
        raw = _make_asymmetric_frame()
        expected = np.fliplr(raw)

        mgr = _make_manager(mirror=True)
        _inject_open_cap(mgr, raw)

        ok, frame = mgr.read()
        self.assertTrue(ok)
        np.testing.assert_array_equal(
            frame, expected,
            err_msg="Frame should be horizontally flipped when mirror is on.",
        )

    def test_mirror_flip_is_not_identity(self):
        """Sanity: flipped != original for an asymmetric frame."""
        raw = _make_asymmetric_frame()
        self.assertFalse(
            np.array_equal(raw, np.fliplr(raw)),
            "Test frame must be asymmetric for the flip assertions to be meaningful.",
        )

    def test_mirror_pixel_positions_match_detection_coordinates(self):
        """
        Simulate the critical production concern: a bright pixel in the
        top-left of the raw frame must appear in the top-right of the
        mirrored frame.

        When hand detection runs on the same mirrored frame, the detected
        x-coordinate will naturally be near the right edge — matching what
        the operator sees on screen.
        """
        raw = np.zeros((48, 64, 3), dtype=np.uint8)
        raw[0, 0] = (255, 255, 255)   # bright pixel at top-left

        mgr = _make_manager(mirror=True)
        _inject_open_cap(mgr, raw)

        ok, frame = mgr.read()
        self.assertTrue(ok)

        # After flip the bright pixel should be at the top-RIGHT (col 63)
        self.assertTrue(
            np.any(frame[0, -1] > 200),
            "After horizontal flip, the top-left bright pixel must be at top-right.",
        )
        # And top-left should now be black
        self.assertTrue(
            np.all(frame[0, 0] == 0),
            "After horizontal flip, the top-left corner should be dark.",
        )

    def test_toggle_mirror_on_live_manager(self):
        """
        Toggling mirror_horizontal at runtime (as the UI checkbox does)
        must take effect immediately on the next read().
        """
        raw = _make_asymmetric_frame()
        mgr = _make_manager(mirror=False)

        # Each read() call needs a fresh frame from cap.read()
        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = True
        cap_mock.read.side_effect = [(True, raw.copy()), (True, raw.copy())]
        mgr._cap     = cap_mock
        mgr._is_open = True

        # First read — no flip
        ok1, f1 = mgr.read()
        np.testing.assert_array_equal(f1, raw, err_msg="Should be unflipped")

        # Toggle mirror on (simulates checkbox stateChanged)
        mgr.mirror_horizontal = True

        ok2, f2 = mgr.read()
        np.testing.assert_array_equal(
            f2, np.fliplr(raw),
            err_msg="Should be flipped after toggling mirror_horizontal=True",
        )


if __name__ == "__main__":
    unittest.main()
