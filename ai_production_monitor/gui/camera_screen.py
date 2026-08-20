"""
gui/camera_screen.py
Camera Management Screen.

Layout (1 280 × 780 min)
───────────────────────────────────────────────────────────────
 📷  CAMERA MANAGEMENT                              [← Back]
───────────────────────────────────────────────────────────────
┌─── LEFT PANEL (400 px) ──────┐  ┌─── RIGHT PANEL (flex) ───┐
│  Camera Type  [▾ Webcam    ] │  │                           │
│                               │  │     LIVE PREVIEW          │
│  ┌── Webcam Settings ───────┐ │  │   (VideoWidget)           │
│  │  Device  [▾ Camera 0    ]│ │  │                           │
│  │          [⟳ Scan]        │ │  │                           │
│  └──────────────────────────┘ │  │                           │
│  ┌── Network Settings ──────┐ │  │                           │
│  │  URL  [_______________]  │ │  └───────────────────────────┘
│  │  User [_______________]  │ │
│  │  Pass [●●●●●●●] [👁]     │ │  [▶ Start Preview]
│  │  Transport [▾ TCP      ] │ │
│  └──────────────────────────┘ │  ┌─── Test Result ──────────┐
│                               │  │  ✔ Connected 1280×720    │
│  Resolution [▾ 1280×720    ] │  │    @ 30 fps               │
│  FPS        [30 ±]            │  │                           │
│                               │  │  [preview frame here]     │
│  [⚡ Test Connection]         │  └───────────────────────────┘
│  [💾 Save & Continue →]       │
└───────────────────────────────┘

Workers (QThread)
─────────────────
DeviceScanWorker   — scans webcam indices without blocking GUI
ConnectionTestWorker — opens, grabs one frame, releases; emits result
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import (
    QSize, Qt, QThread, QTimer, pyqtSignal,
)
from PyQt5.QtGui import (
    QColor, QFont, QImage, QPainter, QPixmap,
)
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout,
    QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QSizePolicy, QSpinBox,
    QStackedWidget, QVBoxLayout, QWidget,
)

from data.config_handler import ConfigHandler
from gui.theme import (
    FONT_FAMILY, FONT_SIZE_BODY, FONT_SIZE_CAPTION, FONT_SIZE_ICON,
    FONT_SIZE_LABEL, FONT_SIZE_SECTION, FONT_SIZE_TITLE,
    make_font,
    btn_stylesheet, groupbox_stylesheet, input_stylesheet,
    CLR_ACCENT, CLR_MUTED, CLR_SECTION_HDR, CLR_TEXT,
)
from gui.widgets.video_widget import VideoWidget
from vision.camera_handler import (
    CameraConnectionResult,
    CameraErrorCode,
    CameraManager,
    DeviceScanner,
    IPCameraSource,
    RTSPSource,
    WebcamSource,
    camera_manager_from_config,
    source_from_dict,
)


# ============================================================================
# Background Workers
# ============================================================================

class DeviceScanWorker(QThread):
    """
    Scans webcam device indices 0–N in a background thread.

    Signals
    -------
    scan_complete(list)  — list of {"index", "label", "width", "height"}
    """
    scan_complete = pyqtSignal(list)

    def __init__(self, max_index: int = 8, parent=None) -> None:
        super().__init__(parent)
        self._max = max_index

    def run(self) -> None:
        devices = DeviceScanner.scan(self._max)
        self.scan_complete.emit(devices)


class ConnectionTestWorker(QThread):
    """
    Tests a camera config in a background thread, emits structured result.

    Signals
    -------
    test_complete(CameraConnectionResult)
    """
    test_complete = pyqtSignal(object)   # CameraConnectionResult

    def __init__(self, manager: CameraManager, parent=None) -> None:
        super().__init__(parent)
        self._manager = manager

    def run(self) -> None:
        result = self._manager.test()
        self.test_complete.emit(result)


# ============================================================================
# Sub-panel: Webcam settings
# ============================================================================

class _WebcamPanel(QFrame):
    """Settings specific to USB/integrated webcam sources."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet("QFrame { background: transparent; }")
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(8)
        layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # Device combo + scan button in one row
        device_row = QHBoxLayout()
        self.combo_device = QComboBox()
        self.combo_device.addItem("Camera 0   [detecting…]", 0)
        self.combo_device.setStyleSheet(_INPUT_STYLE)
        self.combo_device.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        device_row.addWidget(self.combo_device)

        self.btn_scan = QPushButton("⟳")
        self.btn_scan.setToolTip("Scan for connected webcams")
        self.btn_scan.setFixedSize(28, 28)
        self.btn_scan.setStyleSheet(_BTN_STYLE)
        device_row.addWidget(self.btn_scan)
        layout.addRow(_lbl("Device:"), device_row)

        # Scan status
        self.lbl_scan_status = QLabel("Click ⟳ to detect cameras")
        self.lbl_scan_status.setFont(make_font(FONT_SIZE_CAPTION))
        self.lbl_scan_status.setStyleSheet("color: #546e7a;")
        layout.addRow("", self.lbl_scan_status)

    def set_device_index(self, index: int) -> None:
        for i in range(self.combo_device.count()):
            if self.combo_device.itemData(i) == index:
                self.combo_device.setCurrentIndex(i)
                return
        # Add if not found
        self.combo_device.insertItem(0, f"Camera {index}", index)
        self.combo_device.setCurrentIndex(0)

    def selected_index(self) -> int:
        return int(self.combo_device.currentData() or 0)

    def populate_devices(self, devices: list[dict]) -> None:
        current = self.selected_index()
        self.combo_device.clear()
        if not devices:
            self.combo_device.addItem("No cameras found", 0)
            self.lbl_scan_status.setText("⚠  No webcams detected")
            self.lbl_scan_status.setStyleSheet("color: #ffab00;")
            return
        for d in devices:
            self.combo_device.addItem(d["label"], d["index"])
        self.lbl_scan_status.setText(
            f"✔  {len(devices)} camera{'s' if len(devices) != 1 else ''} found"
        )
        self.lbl_scan_status.setStyleSheet("color: #00c853;")
        # Restore selection
        for i in range(self.combo_device.count()):
            if self.combo_device.itemData(i) == current:
                self.combo_device.setCurrentIndex(i)
                break


# ============================================================================
# Sub-panel: Network camera settings (IP + RTSP)
# ============================================================================

class _NetworkPanel(QFrame):
    """Settings for IP Camera and RTSP sources (URL + auth)."""

    def __init__(self, is_rtsp: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._is_rtsp = is_rtsp
        self.setStyleSheet("QFrame { background: transparent; }")
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(8)
        layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # URL
        self.edit_url = QLineEdit()
        placeholder = (
            "rtsp://192.168.1.100:554/stream1"
            if is_rtsp else
            "http://192.168.1.100/video"
        )
        self.edit_url.setPlaceholderText(placeholder)
        self.edit_url.setStyleSheet(_INPUT_STYLE)
        layout.addRow(_lbl("URL:"), self.edit_url)

        # Username
        self.edit_user = QLineEdit()
        self.edit_user.setPlaceholderText("(optional)")
        self.edit_user.setStyleSheet(_INPUT_STYLE)
        layout.addRow(_lbl("Username:"), self.edit_user)

        # Password (with show/hide toggle)
        pass_row = QHBoxLayout()
        self.edit_pass = QLineEdit()
        self.edit_pass.setPlaceholderText("(optional)")
        self.edit_pass.setEchoMode(QLineEdit.Password)
        self.edit_pass.setStyleSheet(_INPUT_STYLE)
        pass_row.addWidget(self.edit_pass)

        self.btn_show_pass = QPushButton("👁")
        self.btn_show_pass.setCheckable(True)
        self.btn_show_pass.setFixedSize(28, 28)
        self.btn_show_pass.setToolTip("Show / hide password")
        self.btn_show_pass.setStyleSheet(_BTN_STYLE)
        self.btn_show_pass.toggled.connect(self._toggle_pass_visibility)
        pass_row.addWidget(self.btn_show_pass)
        layout.addRow(_lbl("Password:"), pass_row)

        # Transport (RTSP only)
        if is_rtsp:
            self.combo_transport = QComboBox()
            self.combo_transport.addItem("TCP  (more reliable on LAN)", "tcp")
            self.combo_transport.addItem("UDP  (lower latency, may drop)", "udp")
            self.combo_transport.setStyleSheet(_INPUT_STYLE)
            layout.addRow(_lbl("Transport:"), self.combo_transport)
        else:
            self.combo_transport = None

        # URL hint
        hint_text = (
            "e.g.  rtsp://admin:pass@192.168.1.100:554/ch01"
            if is_rtsp else
            "e.g.  http://192.168.1.100:8080/video?channel=1"
        )
        hint = QLabel(hint_text)
        hint.setFont(make_font(FONT_SIZE_CAPTION))
        hint.setStyleSheet("color: #455a64; font-style: italic;")
        hint.setWordWrap(True)
        layout.addRow("", hint)

    def _toggle_pass_visibility(self, checked: bool) -> None:
        mode = QLineEdit.Normal if checked else QLineEdit.Password
        self.edit_pass.setEchoMode(mode)

    # ---- Getters ----

    def url(self)       -> str: return self.edit_url.text().strip()
    def username(self)  -> str: return self.edit_user.text().strip()
    def password(self)  -> str: return self.edit_pass.text()
    def transport(self) -> str:
        if self.combo_transport:
            return self.combo_transport.currentData() or "tcp"
        return "tcp"

    # ---- Setters ----

    def set_url(self, v: str)      -> None: self.edit_url.setText(v)
    def set_username(self, v: str) -> None: self.edit_user.setText(v)
    def set_password(self, v: str) -> None: self.edit_pass.setText(v)
    def set_transport(self, v: str) -> None:
        if self.combo_transport:
            for i in range(self.combo_transport.count()):
                if self.combo_transport.itemData(i) == v:
                    self.combo_transport.setCurrentIndex(i)
                    break


# ============================================================================
# Test-result widget (shows message + snapshot frame)
# ============================================================================

class _TestResultWidget(QFrame):
    """Compact widget displaying the last CameraConnectionResult."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            "QFrame { background:#0a1520; border:1px solid #263238; border-radius:5px; }"
        )
        self.setMinimumHeight(80)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        self._lbl_icon = QLabel("")
        self._lbl_icon.setFont(make_font(FONT_SIZE_ICON))
        self._lbl_icon.setAlignment(Qt.AlignCenter)

        self._lbl_msg = QLabel("Press  ⚡ Test Connection  to verify")
        self._lbl_msg.setFont(make_font(FONT_SIZE_BODY))
        self._lbl_msg.setStyleSheet("color: #546e7a;")
        self._lbl_msg.setWordWrap(True)

        self._snapshot = QLabel()
        self._snapshot.setAlignment(Qt.AlignCenter)
        self._snapshot.hide()

        layout.addWidget(self._lbl_icon, alignment=Qt.AlignCenter)
        layout.addWidget(self._lbl_msg)
        layout.addWidget(self._snapshot)

    def show_result(self, result: CameraConnectionResult) -> None:
        if result.success:
            self._lbl_icon.setText("✔")
            self._lbl_icon.setStyleSheet("color: #00c853;")
            self._lbl_msg.setStyleSheet(
                f"color: #00c853; font-family:{FONT_FAMILY}; font-size:{FONT_SIZE_BODY}px;"
            )
            self._lbl_msg.setText(result.message)
            if result.frame is not None:
                self._show_snapshot(result.frame)
        else:
            self._lbl_icon.setText("✖")
            self._lbl_icon.setStyleSheet("color: #d50000;")
            self._lbl_msg.setStyleSheet(
                f"color: #ef9a9a; font-family:{FONT_FAMILY}; font-size:{FONT_SIZE_BODY}px;"
            )
            self._lbl_msg.setText(result.message)
            self._snapshot.hide()

    def show_testing(self) -> None:
        self._lbl_icon.setText("…")
        self._lbl_icon.setStyleSheet("color: #ffab00;")
        self._lbl_msg.setStyleSheet(f"color: #ffab00; font-family:{FONT_FAMILY}; font-size:{FONT_SIZE_BODY}px;")
        self._lbl_msg.setText("กำลังทดสอบการเชื่อมต่อ…  (Testing connection…)")
        self._snapshot.hide()

    def _show_snapshot(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qi   = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pm   = QPixmap.fromImage(qi).scaled(
            320, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._snapshot.setPixmap(pm)
        self._snapshot.show()


# ============================================================================
# Main Camera Screen
# ============================================================================

# Default path for camera_config.json (one level up from gui/)
_CONFIG_DIR = Path(__file__).parent.parent / "config"

# Camera type display names and their keys
_CAM_TYPES = [
    ("Webcam  (USB / Built-in)",    "webcam"),
    ("IP Camera  (HTTP / MJPEG)",   "ip_camera"),
    ("RTSP Stream  (NVR / IP Cam)", "rtsp"),
]


class CameraScreen(QWidget):
    """
    Camera management screen.

    Signals
    -------
    camera_saved()       — user saved config and wants to continue
    back_requested()     — user clicked Back without saving
    """

    camera_saved    = pyqtSignal()
    back_requested  = pyqtSignal()

    def __init__(self, config: ConfigHandler, parent=None) -> None:
        super().__init__(parent)
        self._config = config

        # Live preview state
        self._preview_manager: Optional[CameraManager] = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(33)
        self._preview_timer.timeout.connect(self._grab_preview_frame)

        # Workers
        self._scan_worker: Optional[DeviceScanWorker] = None
        self._test_worker: Optional[ConnectionTestWorker] = None

        # Last successful test result (used to decide if Save is safe)
        self._last_test_result: Optional[CameraConnectionResult] = None

        self._build_ui()
        self._load_saved_config()

    # ================================================================== #
    # UI construction
    # ================================================================== #

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)
        root.addLayout(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(14)
        body.addWidget(self._build_left_panel(),  stretch=0)
        body.addWidget(self._build_right_panel(), stretch=1)
        root.addLayout(body)

    # ---- Header ----

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        title = QLabel("📷  CAMERA MANAGEMENT")
        title.setFont(make_font(FONT_SIZE_TITLE, bold=True))
        title.setStyleSheet("color: #00bcd4;")
        row.addWidget(title)
        row.addStretch()

        self._lbl_source_desc = QLabel("")
        self._lbl_source_desc.setFont(make_font(FONT_SIZE_BODY))
        self._lbl_source_desc.setStyleSheet("color: #455a64;")
        row.addWidget(self._lbl_source_desc)

        btn_back = _make_btn("← Back", accent="#263238")
        btn_back.clicked.connect(self._on_back)
        row.addWidget(btn_back)
        return row

    # ---- Left panel ----

    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setFixedWidth(400)
        panel.setStyleSheet(
            "QFrame { background:#0d1b2a; border:1px solid #1e3040; border-radius:6px; }"
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)

        # ── Camera type selector ──
        type_grp = _group("Camera Type")
        type_lay = QVBoxLayout(type_grp)
        self._combo_type = QComboBox()
        for label, key in _CAM_TYPES:
            self._combo_type.addItem(label, key)
        self._combo_type.setStyleSheet(_INPUT_STYLE)
        self._combo_type.setFont(make_font(FONT_SIZE_BODY))
        self._combo_type.currentIndexChanged.connect(self._on_type_changed)
        type_lay.addWidget(self._combo_type)
        layout.addWidget(type_grp)

        # ── Source-specific settings (QStackedWidget) ──
        src_grp = _group("Source Settings")
        src_lay = QVBoxLayout(src_grp)

        self._stack = QStackedWidget()
        self._panel_webcam   = _WebcamPanel()
        self._panel_ip       = _NetworkPanel(is_rtsp=False)
        self._panel_rtsp     = _NetworkPanel(is_rtsp=True)
        self._stack.addWidget(self._panel_webcam)    # index 0
        self._stack.addWidget(self._panel_ip)        # index 1
        self._stack.addWidget(self._panel_rtsp)      # index 2
        src_lay.addWidget(self._stack)

        # Wire scan button
        self._panel_webcam.btn_scan.clicked.connect(self._start_device_scan)
        layout.addWidget(src_grp)

        # ── Capture settings ──
        cap_grp = _group("Capture Settings")
        cap_form = QFormLayout(cap_grp)
        cap_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        cap_form.setSpacing(8)

        self._combo_res = QComboBox()
        for r in ["1280 × 720  (HD)", "1920 × 1080  (Full HD)",
                  "640 × 480  (VGA)", "800 × 600  (SVGA)",
                  "2560 × 1440  (2K)", "3840 × 2160  (4K)"]:
            self._combo_res.addItem(r)
        self._combo_res.setStyleSheet(_INPUT_STYLE)
        cap_form.addRow(_lbl("Resolution:"), self._combo_res)

        fps_row = QHBoxLayout()
        self._spin_fps = QSpinBox()
        self._spin_fps.setRange(1, 120)
        self._spin_fps.setValue(30)
        self._spin_fps.setSuffix("  fps")
        self._spin_fps.setStyleSheet(_INPUT_STYLE)
        fps_row.addWidget(self._spin_fps)
        fps_row.addStretch()
        cap_form.addRow(_lbl("Frame Rate:"), fps_row)

        # Reconnect on failure
        self._chk_reconnect = QCheckBox("Auto-reconnect on failure")
        self._chk_reconnect.setChecked(True)
        self._chk_reconnect.setStyleSheet(f"color: #b0bec5; font-family:{FONT_FAMILY}; font-size:{FONT_SIZE_BODY}px;")
        cap_form.addRow("", self._chk_reconnect)

        layout.addWidget(cap_grp)
        layout.addStretch()

        # ── Action buttons ──
        self._btn_test = _make_btn("⚡  Test Connection", accent="#00695c")
        self._btn_test.setFixedHeight(36)
        self._btn_test.clicked.connect(self._test_connection)
        layout.addWidget(self._btn_test)

        self._btn_save = _make_btn("💾  Save & Continue  →", accent="#0d47a1")
        self._btn_save.setFixedHeight(42)
        self._btn_save.clicked.connect(self._save_and_continue)
        layout.addWidget(self._btn_save)

        return panel

    # ---- Right panel ----

    def _build_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet(
            "QFrame { background:#0d1b2a; border:1px solid #1e3040; border-radius:6px; }"
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Preview label + live video
        preview_hdr = QHBoxLayout()
        preview_hdr.addWidget(_lbl("Live Preview", size=10))
        preview_hdr.addStretch()
        self._lbl_resolution = _lbl("— × —")
        preview_hdr.addWidget(self._lbl_resolution)
        layout.addLayout(preview_hdr)

        self._video = VideoWidget()
        self._video.setMinimumSize(480, 270)
        layout.addWidget(self._video, stretch=1)

        # Preview toggle
        preview_row = QHBoxLayout()
        self._btn_preview = _make_btn("▶  Start Preview", accent="#1b5e20")
        self._btn_preview.clicked.connect(self._toggle_preview)
        preview_row.addWidget(self._btn_preview)
        preview_row.addStretch()
        layout.addLayout(preview_row)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("QFrame { background:#1e3040; border:none; max-height:1px; }")
        layout.addWidget(sep)

        # Test result widget
        layout.addWidget(_lbl("Connection Test Result:", size=9))
        self._test_result_widget = _TestResultWidget()
        layout.addWidget(self._test_result_widget)

        return panel

    # ================================================================== #
    # Logic — type switching
    # ================================================================== #

    def _on_type_changed(self, idx: int) -> None:
        self._stack.setCurrentIndex(idx)
        key = self._combo_type.itemData(idx)
        self._lbl_source_desc.setText({
            "webcam":    "USB / Built-in camera",
            "ip_camera": "HTTP / MJPEG network stream",
            "rtsp":      "RTSP stream (NVR / IP camera)",
        }.get(key, ""))
        # Auto-trigger scan when switching to webcam with no devices listed
        if key == "webcam" and self._panel_webcam.combo_device.count() <= 1:
            self._start_device_scan()

    # ================================================================== #
    # Logic — device scan
    # ================================================================== #

    def _start_device_scan(self) -> None:
        if self._scan_worker and self._scan_worker.isRunning():
            return
        self._panel_webcam.lbl_scan_status.setText("⟳  Scanning…")
        self._panel_webcam.lbl_scan_status.setStyleSheet("color: #ffab00;")
        self._panel_webcam.btn_scan.setEnabled(False)

        self._scan_worker = DeviceScanWorker(max_index=8)
        self._scan_worker.scan_complete.connect(self._on_scan_complete)
        self._scan_worker.start()

    def _on_scan_complete(self, devices: list) -> None:
        self._panel_webcam.populate_devices(devices)
        self._panel_webcam.btn_scan.setEnabled(True)

    # ================================================================== #
    # Logic — connection test
    # ================================================================== #

    def _test_connection(self) -> None:
        if self._test_worker and self._test_worker.isRunning():
            return
        # Stop preview if running so it doesn't hold the camera
        if self._preview_timer.isActive():
            self._toggle_preview()

        mgr = self._build_camera_manager()
        if mgr is None:
            return

        self._btn_test.setEnabled(False)
        self._test_result_widget.show_testing()

        self._test_worker = ConnectionTestWorker(mgr)
        self._test_worker.test_complete.connect(self._on_test_complete)
        self._test_worker.start()

    def _on_test_complete(self, result: CameraConnectionResult) -> None:
        self._last_test_result = result
        self._test_result_widget.show_result(result)
        self._btn_test.setEnabled(True)

        if result.success and result.frame is not None:
            self._video.set_frame(result.frame)
            self._lbl_resolution.setText(
                f"{result.actual_width} × {result.actual_height} "
                f"@ {result.actual_fps:.0f} fps"
            )

    # ================================================================== #
    # Logic — live preview
    # ================================================================== #

    def _toggle_preview(self) -> None:
        if self._preview_timer.isActive():
            self._stop_preview()
            self._btn_preview.setText("▶  Start Preview")
        else:
            mgr = self._build_camera_manager()
            if mgr is None:
                return
            result = mgr.open()
            if result.success:
                self._preview_manager = mgr
                self._preview_timer.start()
                self._btn_preview.setText("⏹  Stop Preview")
                self._lbl_resolution.setText(
                    f"{result.actual_width} × {result.actual_height} "
                    f"@ {result.actual_fps:.0f} fps"
                )
            else:
                self._test_result_widget.show_result(result)

    def _stop_preview(self) -> None:
        self._preview_timer.stop()
        if self._preview_manager:
            self._preview_manager.release()
            self._preview_manager = None

    def _grab_preview_frame(self) -> None:
        if self._preview_manager and self._preview_manager.is_open():
            ok, frame = self._preview_manager.read()
            if ok and frame is not None:
                self._video.set_frame(frame)
            elif not ok:
                self._stop_preview()
                self._btn_preview.setText("▶  Start Preview")

    # ================================================================== #
    # Logic — save / load
    # ================================================================== #

    def _build_camera_manager(self) -> Optional[CameraManager]:
        """Build a CameraManager from the current UI state."""
        type_key = self._combo_type.currentData()
        try:
            if type_key == "webcam":
                source = WebcamSource(
                    device_index=self._panel_webcam.selected_index()
                )
            elif type_key == "ip_camera":
                source = IPCameraSource(
                    url      = self._panel_ip.url(),
                    username = self._panel_ip.username(),
                    password = self._panel_ip.password(),
                )
            else:
                source = RTSPSource(
                    url       = self._panel_rtsp.url(),
                    username  = self._panel_rtsp.username(),
                    password  = self._panel_rtsp.password(),
                    transport = self._panel_rtsp.transport(),
                )

            # Pre-flight validation
            err = source.validate()
            if err:
                self._test_result_widget.show_result(err)
                return None

            w, h = self._parse_resolution()
            return CameraManager(
                source    = source,
                width     = w,
                height    = h,
                fps       = self._spin_fps.value(),
                reconnect = self._chk_reconnect.isChecked(),
            )
        except Exception as exc:
            self._test_result_widget.show_result(
                CameraConnectionResult.fail(CameraErrorCode.UNKNOWN, str(exc))
            )
            return None

    def _parse_resolution(self) -> tuple[int, int]:
        text = self._combo_res.currentText()
        # "1280 × 720  (HD)" → 1280, 720
        import re
        m = re.search(r"(\d+)\s*[×x]\s*(\d+)", text)
        if m:
            return int(m.group(1)), int(m.group(2))
        return 1280, 720

    def _build_config_dict(self) -> dict:
        type_key = self._combo_type.currentData()
        w, h = self._parse_resolution()

        base = {
            "type":                 type_key,
            "width":                w,
            "height":               h,
            "fps":                  self._spin_fps.value(),
            "reconnect_on_failure": self._chk_reconnect.isChecked(),
        }

        if type_key == "webcam":
            base["device_index"] = self._panel_webcam.selected_index()
            base["url"]          = ""
            base["username"]     = ""
            base["password"]     = ""
        elif type_key == "ip_camera":
            base["device_index"] = 0
            base["url"]          = self._panel_ip.url()
            base["username"]     = self._panel_ip.username()
            base["password"]     = self._panel_ip.password()
        else:
            base["device_index"] = 0
            base["url"]          = self._panel_rtsp.url()
            base["username"]     = self._panel_rtsp.username()
            base["password"]     = self._panel_rtsp.password()
            base["transport"]    = self._panel_rtsp.transport()

        return base

    def _save_and_continue(self) -> None:
        cfg = self._build_config_dict()

        # Persist to camera_config.json
        path = _CONFIG_DIR / "camera_config.json"
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

        # Also update the in-memory ConfigHandler so VisionThread picks it up
        self._config.set_camera_config(
            camera_type  = cfg["type"],
            device_index = cfg.get("device_index", 0),
            url          = cfg.get("url", ""),
            width        = cfg["width"],
            height       = cfg["height"],
            fps          = cfg["fps"],
        )
        self._config.save()

        self._stop_preview()
        self.camera_saved.emit()

    def _load_saved_config(self) -> None:
        """Populate UI from camera_config.json (if it exists), else from ConfigHandler."""
        cfg_path = _CONFIG_DIR / "camera_config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                cfg = self._config.get_camera_config()
        else:
            cfg = self._config.get_camera_config()

        self._apply_config_to_ui(cfg)

    def _apply_config_to_ui(self, cfg: dict) -> None:
        type_key = cfg.get("type", "webcam")
        # Set combo
        for i in range(self._combo_type.count()):
            if self._combo_type.itemData(i) == type_key:
                self._combo_type.setCurrentIndex(i)
                break

        # Per-type fields
        if type_key == "webcam":
            idx = cfg.get("device_index", 0)
            self._panel_webcam.set_device_index(idx)
        elif type_key == "ip_camera":
            self._panel_ip.set_url(cfg.get("url", ""))
            self._panel_ip.set_username(cfg.get("username", ""))
            self._panel_ip.set_password(cfg.get("password", ""))
        else:
            self._panel_rtsp.set_url(cfg.get("url", ""))
            self._panel_rtsp.set_username(cfg.get("username", ""))
            self._panel_rtsp.set_password(cfg.get("password", ""))
            self._panel_rtsp.set_transport(cfg.get("transport", "tcp"))

        # Resolution
        w, h = cfg.get("width", 1280), cfg.get("height", 720)
        target = f"{w} × {h}"
        for i in range(self._combo_res.count()):
            if self._combo_res.itemText(i).startswith(target):
                self._combo_res.setCurrentIndex(i)
                break

        self._spin_fps.setValue(cfg.get("fps", 30))
        self._chk_reconnect.setChecked(cfg.get("reconnect_on_failure", True))

    # ================================================================== #
    # Navigation
    # ================================================================== #

    def _on_back(self) -> None:
        self._stop_preview()
        self.back_requested.emit()

    def closeEvent(self, event) -> None:
        self._stop_preview()
        super().closeEvent(event)


# ============================================================================
# Style constants  (module-level for reuse across sub-widgets)
# ============================================================================

_INPUT_STYLE = input_stylesheet()

_BTN_STYLE = f"""
    QPushButton {{
        background-color: #263238;
        color: #b0bec5;
        border: 1px solid #455a64;
        border-radius: 3px;
        padding: 4px 8px;
        font-family: {FONT_FAMILY};
        font-size: {FONT_SIZE_BODY}px;
    }}
    QPushButton:hover   {{ background-color: #37474f; border-color: {CLR_ACCENT}; }}
    QPushButton:pressed {{ background-color: #0d1b2a; }}
    QPushButton:disabled {{ color: #37474f; }}
    QPushButton:checked {{ background-color: #1565c0; color: #fff; }}
"""


# ============================================================================
# Helpers
# ============================================================================

def _make_btn(label: str, accent: str = "#263238") -> QPushButton:
    btn = QPushButton(label)
    btn.setFont(make_font(FONT_SIZE_BODY))
    btn.setStyleSheet(btn_stylesheet(accent))
    return btn


def _lbl(text: str, size: int | None = None) -> QLabel:
    l = QLabel(text)
    l.setFont(make_font(size if size is not None else FONT_SIZE_LABEL))
    l.setStyleSheet(f"color: {CLR_MUTED}; background: transparent; border: none;")
    return l


def _group(title: str) -> QGroupBox:
    g = QGroupBox(title)
    g.setFont(make_font(FONT_SIZE_SECTION, bold=True))
    g.setStyleSheet(groupbox_stylesheet())
    return g
