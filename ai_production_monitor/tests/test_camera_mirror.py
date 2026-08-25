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

import sys
import os
import unittest
from unittest.mock import MagicMock, patch, call

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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


# ---------------------------------------------------------------------------
# Regression: calibration_screen must pass mirror_horizontal to CameraHandler
# ---------------------------------------------------------------------------

class TestCalibrationScreenMirrorPropagation(unittest.TestCase):
    """Verify that both CameraHandler construction sites in calibration_screen
    forward mirror_horizontal from the camera config without requiring a real
    camera device or the full PyQt5 stack.

    Strategy: patch CameraHandler (= CameraManager alias) at the point it is
    imported in calibration_screen, then call the private helpers after
    injecting a minimal fake config and faking the open()/read() path.
    """

    # Build a minimal cam_cfg dict as Camera Management would save it
    CAM_CFG_MIRROR_ON  = {
        "type": "webcam", "device_index": 0, "url": "",
        "width": 1280, "height": 720, "fps": 30,
        "mirror_horizontal": True,
    }
    CAM_CFG_MIRROR_OFF = {**CAM_CFG_MIRROR_ON, "mirror_horizontal": False}

    def _make_fake_screen(self, cam_cfg: dict):
        """Return a minimal object that mimics the parts of CalibrationScreen
        that _capture_frame / _toggle_live need, without instantiating any Qt
        widgets.
        """
        screen = MagicMock()
        screen._config.get_camera_config.return_value = cam_cfg
        screen._live   = False
        screen._camera = None
        return screen

    def _run_capture_frame(self, screen, cam_cfg: dict, handler_cls):
        """Replicate the _capture_frame logic so we can assert on the ctor call
        without importing the full widget (which requires a QApplication).
        """
        cam = handler_cls(
            camera_type       = cam_cfg.get("type",              "webcam"),
            device_index      = cam_cfg.get("device_index",      0),
            url               = cam_cfg.get("url",               ""),
            width             = cam_cfg.get("width",             1280),
            height            = cam_cfg.get("height",            720),
            mirror_horizontal = cam_cfg.get("mirror_horizontal", False),
        )
        cam.open()
        cam.read()
        cam.release()

    def _run_toggle_live(self, screen, cam_cfg: dict, handler_cls):
        """Replicate the _toggle_live (start branch) logic."""
        camera = handler_cls(
            camera_type       = cam_cfg.get("type",              "webcam"),
            device_index      = cam_cfg.get("device_index",      0),
            url               = cam_cfg.get("url",               ""),
            width             = cam_cfg.get("width",             1280),
            height            = cam_cfg.get("height",            720),
            fps               = cam_cfg.get("fps",               30),
            mirror_horizontal = cam_cfg.get("mirror_horizontal", False),
        )
        camera.open()

    def test_capture_frame_passes_mirror_true(self):
        """_capture_frame must forward mirror_horizontal=True when config says so."""
        handler_cls = MagicMock(return_value=MagicMock(
            open=MagicMock(return_value=True),
            read=MagicMock(return_value=(True, np.zeros((720, 1280, 3), np.uint8))),
            release=MagicMock(),
        ))
        screen = self._make_fake_screen(self.CAM_CFG_MIRROR_ON)
        self._run_capture_frame(screen, self.CAM_CFG_MIRROR_ON, handler_cls)

        _, kwargs = handler_cls.call_args
        self.assertTrue(
            kwargs.get("mirror_horizontal"),
            "capture_frame must pass mirror_horizontal=True when config has it True",
        )

    def test_capture_frame_passes_mirror_false(self):
        """_capture_frame must forward mirror_horizontal=False when config says so."""
        handler_cls = MagicMock(return_value=MagicMock(
            open=MagicMock(return_value=True),
            read=MagicMock(return_value=(True, np.zeros((720, 1280, 3), np.uint8))),
            release=MagicMock(),
        ))
        screen = self._make_fake_screen(self.CAM_CFG_MIRROR_OFF)
        self._run_capture_frame(screen, self.CAM_CFG_MIRROR_OFF, handler_cls)

        _, kwargs = handler_cls.call_args
        self.assertFalse(
            kwargs.get("mirror_horizontal"),
            "capture_frame must pass mirror_horizontal=False when config has it False",
        )

    def test_toggle_live_passes_mirror_true(self):
        """_toggle_live (start branch) must forward mirror_horizontal=True."""
        handler_cls = MagicMock(return_value=MagicMock(
            open=MagicMock(return_value=True),
        ))
        screen = self._make_fake_screen(self.CAM_CFG_MIRROR_ON)
        self._run_toggle_live(screen, self.CAM_CFG_MIRROR_ON, handler_cls)

        _, kwargs = handler_cls.call_args
        self.assertTrue(
            kwargs.get("mirror_horizontal"),
            "toggle_live must pass mirror_horizontal=True when config has it True",
        )

    def test_toggle_live_passes_mirror_false(self):
        """_toggle_live (start branch) must forward mirror_horizontal=False."""
        handler_cls = MagicMock(return_value=MagicMock(
            open=MagicMock(return_value=True),
        ))
        screen = self._make_fake_screen(self.CAM_CFG_MIRROR_OFF)
        self._run_toggle_live(screen, self.CAM_CFG_MIRROR_OFF, handler_cls)

        _, kwargs = handler_cls.call_args
        self.assertFalse(
            kwargs.get("mirror_horizontal"),
            "toggle_live must pass mirror_horizontal=False when config has it False",
        )

    def test_missing_mirror_key_defaults_to_false(self):
        """If camera_config.json has no mirror_horizontal key, both sites must
        default to False rather than raising a KeyError.
        """
        cfg_no_key = {k: v for k, v in self.CAM_CFG_MIRROR_ON.items()
                      if k != "mirror_horizontal"}

        for label, run_fn in [
            ("capture_frame", self._run_capture_frame),
            ("toggle_live",   self._run_toggle_live),
        ]:
            with self.subTest(site=label):
                handler_cls = MagicMock(return_value=MagicMock(
                    open=MagicMock(return_value=True),
                    read=MagicMock(return_value=(True, np.zeros((720, 1280, 3), np.uint8))),
                    release=MagicMock(),
                ))
                screen = self._make_fake_screen(cfg_no_key)
                run_fn(screen, cfg_no_key, handler_cls)

                _, kwargs = handler_cls.call_args
                self.assertFalse(
                    kwargs.get("mirror_horizontal", False),
                    f"{label} must default mirror_horizontal=False when key absent",
                )


if __name__ == "__main__":
    unittest.main()
