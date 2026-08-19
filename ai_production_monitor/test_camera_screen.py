"""
test_camera_screen.py
Run the CameraScreen and Camera Strategy classes in isolation.

Usage
-----
    cd ai_production_monitor
    python test_camera_screen.py

What it tests
-------------
1. Strategy pattern unit tests (no camera hardware needed)
2. Opens the full CameraScreen GUI
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# ── Unit tests for camera_handler (no GUI, no hardware) ──────────────────

def run_unit_tests() -> None:
    from vision.camera_handler import (
        WebcamSource, IPCameraSource, RTSPSource,
        CameraConnectionResult, CameraErrorCode,
        source_from_dict, camera_manager_from_config,
    )

    print("=" * 60)
    print("Camera Handler — Unit Tests")
    print("=" * 60)

    # ---- WebcamSource ----
    ws = WebcamSource(device_index=0)
    assert ws.build_capture_source() == 0
    assert ws.validate() is None
    assert ws.to_dict() == {"type": "webcam", "device_index": 0}
    ws2 = WebcamSource.from_dict({"device_index": 2})
    assert ws2.device_index == 2
    print("✔  WebcamSource")

    # ---- IPCameraSource ----
    ip = IPCameraSource(
        url="http://192.168.1.100/video",
        username="admin",
        password="pass123",
    )
    src = ip.build_capture_source()
    assert "admin" in src and "pass123" in src, f"Auth not injected: {src}"
    assert ip.validate() is None
    bad_ip = IPCameraSource(url="ftp://bad-url")
    assert bad_ip.validate() is not None   # should fail
    assert bad_ip.validate().error_code == CameraErrorCode.INVALID_URL
    print("✔  IPCameraSource  (auth injection + URL validation)")

    # ---- RTSPSource ----
    rt = RTSPSource(
        url="rtsp://192.168.1.200:554/stream1",
        username="user",
        password="secret",
        transport="tcp",
    )
    assert rt.validate() is None
    src_rt = rt.build_capture_source()
    assert "user" in src_rt and "secret" in src_rt
    bad_rt = RTSPSource(url="http://wrong-scheme/stream")
    assert bad_rt.validate() is not None
    print("✔  RTSPSource  (auth injection + scheme validation)")

    # ---- source_from_dict round-trip ----
    for original in [
        {"type": "webcam", "device_index": 1},
        {"type": "ip_camera", "url": "http://cam/video", "username": "a", "password": "b"},
        {"type": "rtsp", "url": "rtsp://cam:554/ch1", "username": "", "password": "", "transport": "udp"},
    ]:
        s = source_from_dict(original)
        d = s.to_dict()
        assert d["type"] == original["type"], f"Round-trip failed: {d}"
    print("✔  source_from_dict  round-trip")

    # ---- CameraConnectionResult ----
    ok_r = CameraConnectionResult.ok(w=1280, h=720, fps=30.0)
    assert ok_r.success
    assert "1280" in ok_r.message

    fail_r = CameraConnectionResult.fail(CameraErrorCode.AUTH_FAILED, "401")
    assert not fail_r.success
    assert "Password" in fail_r.message or "password" in fail_r.message.lower()
    print("✔  CameraConnectionResult  (ok / fail messages)")

    # ---- camera_manager_from_config ----
    mgr = camera_manager_from_config({
        "type": "webcam", "device_index": 0,
        "width": 640, "height": 480, "fps": 15,
        "reconnect_on_failure": False,
    })
    assert mgr.width == 640
    print("✔  camera_manager_from_config")

    print()
    print("All unit tests passed.")
    print()


# ── GUI test ──────────────────────────────────────────────────────────────

def run_gui() -> None:
    from PyQt5.QtWidgets import QApplication, QMainWindow
    from PyQt5.QtGui import QPalette, QColor, QFont

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(13,  27,  42))
    pal.setColor(QPalette.WindowText,      QColor(207, 216, 220))
    pal.setColor(QPalette.Base,            QColor(21,  32,  40))
    pal.setColor(QPalette.AlternateBase,   QColor(13,  27,  42))
    pal.setColor(QPalette.ToolTipBase,     QColor(0,   0,   0))
    pal.setColor(QPalette.ToolTipText,     QColor(207, 216, 220))
    pal.setColor(QPalette.Text,            QColor(207, 216, 220))
    pal.setColor(QPalette.Button,          QColor(38,  50,  56))
    pal.setColor(QPalette.ButtonText,      QColor(207, 216, 220))
    pal.setColor(QPalette.Highlight,       QColor(21, 101, 192))
    pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(pal)
    app.setFont(QFont("Consolas", 9))

    from data.config_handler import ConfigHandler
    from gui.camera_screen import CameraScreen

    config = ConfigHandler()
    win = QMainWindow()
    win.setWindowTitle("Camera Management — Test")
    win.setMinimumSize(1100, 700)

    screen = CameraScreen(config)
    screen.camera_saved.connect(lambda: print("[SAVED] Camera config saved"))
    screen.back_requested.connect(win.close)

    win.setCentralWidget(screen)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_unit_tests()
    run_gui()
