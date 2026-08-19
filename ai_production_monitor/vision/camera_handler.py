"""
vision/camera_handler.py
Camera abstraction layer — Strategy pattern.

Architecture
------------
BaseCameraSource  (ABC)            ← Strategy interface
├── WebcamSource                   ← cv2.VideoCapture(int)
├── IPCameraSource                 ← cv2.VideoCapture("http://...")
└── RTSPSource                     ← cv2.VideoCapture("rtsp://...")

CameraManager  (Context)           ← owns one BaseCameraSource
    .open()  → CameraConnectionResult
    .read()  → (ok, frame | None)
    .release()
    .test()  → CameraConnectionResult   # non-destructive, closes after

DeviceScanner  (static)            ← probes webcam indices 0-N

CameraConnectionResult  (dataclass) ← structured result with friendly msg

Backward-compat alias
---------------------
CameraHandler = CameraManager      ← vision_thread.py imports this name

Usage example
-------------
    src = WebcamSource(device_index=0)
    mgr = CameraManager(src, width=1280, height=720, fps=30)
    result = mgr.test()
    if result.success:
        mgr.open()
        ok, frame = mgr.read()
        mgr.release()
"""

from __future__ import annotations

import platform
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import numpy as np


# ============================================================================
# Error taxonomy
# ============================================================================

class CameraErrorCode(str, Enum):
    DEVICE_NOT_FOUND    = "DEVICE_NOT_FOUND"
    DEVICE_BUSY         = "DEVICE_BUSY"
    CONNECTION_TIMEOUT  = "CONNECTION_TIMEOUT"
    AUTH_FAILED         = "AUTH_FAILED"
    INVALID_URL         = "INVALID_URL"
    NO_FRAME            = "NO_FRAME"
    CODEC_ERROR         = "CODEC_ERROR"
    UNKNOWN             = "UNKNOWN"
    OK                  = "OK"


# Human-readable messages (Thai + English) shown directly in the GUI.
_FRIENDLY: dict[CameraErrorCode, str] = {
    CameraErrorCode.DEVICE_NOT_FOUND: (
        "ไม่พบกล้อง (Device not found)\n"
        "→ ตรวจสอบ:\n"
        "   • ต่อ USB เรียบร้อยแล้วหรือยัง?\n"
        "   • ลองเปลี่ยน Device Index (0, 1, 2 …)\n"
        "   • Device Manager: ตรวจว่า driver ถูก enable"
    ),
    CameraErrorCode.DEVICE_BUSY: (
        "กล้องถูกใช้งานโดยโปรแกรมอื่นอยู่ (Camera busy)\n"
        "→ ปิดโปรแกรมเหล่านี้ก่อน:\n"
        "   Zoom · Teams · OBS · Skype · Discord · WhatsApp"
    ),
    CameraErrorCode.CONNECTION_TIMEOUT: (
        "เชื่อมต่อไม่ได้ภายในเวลาที่กำหนด (Connection timed out)\n"
        "→ ตรวจสอบ:\n"
        "   • IP address / URL ถูกต้องไหม?\n"
        "   • กล้องและเครื่องคอมอยู่ใน network เดียวกันไหม?\n"
        "   • Firewall / router ปิดกั้น port อยู่ไหม?"
    ),
    CameraErrorCode.AUTH_FAILED: (
        "Username / Password ไม่ถูกต้อง (Authentication failed)\n"
        "→ ตรวจสอบ credentials ของกล้อง\n"
        "   (ค่า default มักเป็น  admin / admin  หรือ  admin / 12345)"
    ),
    CameraErrorCode.INVALID_URL: (
        "URL ไม่ถูกรูปแบบ (Invalid URL)\n"
        "→ ตัวอย่างที่ถูกต้อง:\n"
        "   http://192.168.1.100/video\n"
        "   http://192.168.1.100:8080/video?channel=1\n"
        "   rtsp://192.168.1.100:554/stream1\n"
        "   rtsp://admin:pass@192.168.1.100:554/ch01/main"
    ),
    CameraErrorCode.NO_FRAME: (
        "เชื่อมต่อสำเร็จแต่ไม่ได้รับภาพ (Connected but no frames)\n"
        "→ ตรวจสอบ:\n"
        "   • Stream path ถูกต้องไหม?  (/stream1, /ch01, /video …)\n"
        "   • Codec ที่กล้องใช้รองรับไหม? (H.264 / MJPEG แนะนำ)"
    ),
    CameraErrorCode.CODEC_ERROR: (
        "Codec ไม่รองรับ (Codec not supported)\n"
        "→ เปลี่ยน encoding ของกล้องเป็น H.264 หรือ MJPEG"
    ),
    CameraErrorCode.UNKNOWN: (
        "เกิดข้อผิดพลาดที่ไม่ทราบสาเหตุ (Unknown error)"
    ),
    CameraErrorCode.OK: "เชื่อมต่อสำเร็จ (Connected)",
}


def friendly_message(
    code: CameraErrorCode,
    detail: str = "",
) -> str:
    base = _FRIENDLY.get(code, _FRIENDLY[CameraErrorCode.UNKNOWN])
    return f"{base}\n\nDetail: {detail}" if detail else base


# ============================================================================
# Result dataclass
# ============================================================================

@dataclass
class CameraConnectionResult:
    """
    Structured result returned by CameraManager.test() and .open().
    Never raises; all errors are captured here.
    """
    success:        bool
    error_code:     CameraErrorCode = CameraErrorCode.OK
    message:        str  = ""          # friendly multi-line message
    technical_msg:  str  = ""          # raw OpenCV / exception text (log only)
    frame:          Optional[np.ndarray] = field(default=None, repr=False)
    actual_width:   int   = 0
    actual_height:  int   = 0
    actual_fps:     float = 0.0

    @classmethod
    def ok(
        cls,
        frame: Optional[np.ndarray] = None,
        w: int = 0, h: int = 0, fps: float = 0.0,
    ) -> "CameraConnectionResult":
        return cls(
            success       = True,
            error_code    = CameraErrorCode.OK,
            message       = f"✔  เชื่อมต่อสำเร็จ  {w}×{h} @ {fps:.0f} fps",
            frame         = frame,
            actual_width  = w,
            actual_height = h,
            actual_fps    = fps,
        )

    @classmethod
    def fail(
        cls,
        code: CameraErrorCode,
        detail: str = "",
    ) -> "CameraConnectionResult":
        return cls(
            success       = False,
            error_code    = code,
            message       = friendly_message(code, detail),
            technical_msg = detail,
        )

    def __bool__(self) -> bool:
        """Allow  `if not manager.open(): ...`  in legacy call-sites."""
        return self.success


# ============================================================================
# Strategy — BaseCameraSource
# ============================================================================

class BaseCameraSource(ABC):
    """
    Abstract strategy.  Knows only how to produce:
    - a cv2 capture source (int or URL string)
    - a validation result before attempting to open
    - serialisation to/from dict (for config JSON)
    """

    # Sub-classes must set this
    SOURCE_TYPE: str = ""

    @abstractmethod
    def build_capture_source(self) -> int | str:
        """Return the value passed to cv2.VideoCapture(...)."""

    @abstractmethod
    def validate(self) -> Optional[CameraConnectionResult]:
        """
        Fast pre-flight check (no network call).
        Returns a failure CameraConnectionResult if invalid, else None.
        """

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialise to config dict."""

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> "BaseCameraSource":
        """Deserialise from config dict."""

    def description(self) -> str:
        """Short human-readable description for UI."""
        return f"{self.SOURCE_TYPE}"


# ============================================================================
# WebcamSource
# ============================================================================

class WebcamSource(BaseCameraSource):
    """USB / integrated webcam accessed by integer device index."""

    SOURCE_TYPE = "webcam"

    # Use DirectShow on Windows for faster open & richer error info
    _BACKEND = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index

    def build_capture_source(self) -> int:
        return self.device_index

    def validate(self) -> Optional[CameraConnectionResult]:
        if not isinstance(self.device_index, int) or self.device_index < 0:
            return CameraConnectionResult.fail(
                CameraErrorCode.INVALID_URL,
                f"Device index must be a non-negative integer, got: {self.device_index}",
            )
        return None   # passes

    def to_dict(self) -> dict:
        return {"type": self.SOURCE_TYPE, "device_index": self.device_index}

    @classmethod
    def from_dict(cls, d: dict) -> "WebcamSource":
        return cls(device_index=d.get("device_index", 0))

    def description(self) -> str:
        return f"Webcam  (index {self.device_index})"

    def open_capture(self) -> cv2.VideoCapture:
        """Open with platform-optimal backend."""
        return cv2.VideoCapture(self.device_index, self._BACKEND)


# ============================================================================
# IPCameraSource
# ============================================================================

class IPCameraSource(BaseCameraSource):
    """HTTP / MJPEG IP camera."""

    SOURCE_TYPE = "ip_camera"

    # Regex: requires a host, optional port, optional path
    _URL_RE = re.compile(
        r"^https?://"               # http:// or https://
        r"([^/:@\s]+)"              # host
        r"(:\d+)?"                  # optional :port
        r"(/[^\s]*)?"               # optional path
        r"$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        url: str,
        username: str = "",
        password: str = "",
    ) -> None:
        self.url      = url.strip()
        self.username = username
        self.password = password

    def build_capture_source(self) -> str:
        return self._inject_auth(self.url, self.username, self.password)

    def validate(self) -> Optional[CameraConnectionResult]:
        if not self.url:
            return CameraConnectionResult.fail(
                CameraErrorCode.INVALID_URL, "URL is empty"
            )
        if not self._URL_RE.match(self.url):
            return CameraConnectionResult.fail(
                CameraErrorCode.INVALID_URL, f"Not a valid HTTP URL: {self.url}"
            )
        return None

    def to_dict(self) -> dict:
        return {
            "type":     self.SOURCE_TYPE,
            "url":      self.url,
            "username": self.username,
            "password": self.password,   # store; caller should encrypt in prod
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IPCameraSource":
        return cls(
            url      = d.get("url", ""),
            username = d.get("username", ""),
            password = d.get("password", ""),
        )

    def description(self) -> str:
        host = re.sub(r"^https?://", "", self.url).split("/")[0]
        return f"IP Camera  ({host})"

    @staticmethod
    def _inject_auth(url: str, username: str, password: str) -> str:
        """Embed credentials into URL: http://user:pass@host/path"""
        if not username:
            return url
        # Strip existing auth if present
        url = re.sub(r"(https?://)([^@/]+@)?", r"\1", url)
        # Insert after scheme
        return re.sub(
            r"(https?://)",
            rf"\1{re.escape(username)}:{re.escape(password)}@",
            url,
        )


# ============================================================================
# RTSPSource
# ============================================================================

class RTSPSource(BaseCameraSource):
    """RTSP stream (e.g., IP cameras, NVRs, encoders)."""

    SOURCE_TYPE = "rtsp"

    _URL_RE = re.compile(
        r"^rtsps?://"
        r"([^/:@\s]+)"
        r"(:\d+)?"
        r"(/[^\s]*)?"
        r"$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        url: str,
        username: str = "",
        password: str = "",
        transport: str = "tcp",   # "tcp" | "udp"
    ) -> None:
        self.url       = url.strip()
        self.username  = username
        self.password  = password
        self.transport = transport   # TCP is more reliable over LAN

    def build_capture_source(self) -> str:
        full_url = self._inject_auth(self.url, self.username, self.password)
        # Append OpenCV RTSP transport hint
        if "?" not in full_url and self.transport:
            # Use ffmpeg / gstreamer pipeline hint via env, or just pass URL
            pass
        return full_url

    def validate(self) -> Optional[CameraConnectionResult]:
        if not self.url:
            return CameraConnectionResult.fail(
                CameraErrorCode.INVALID_URL, "RTSP URL is empty"
            )
        if not self._URL_RE.match(self.url):
            return CameraConnectionResult.fail(
                CameraErrorCode.INVALID_URL,
                f"Not a valid RTSP URL: {self.url}\n"
                "Expected format: rtsp://[user:pass@]host[:port]/path",
            )
        return None

    def to_dict(self) -> dict:
        return {
            "type":      self.SOURCE_TYPE,
            "url":       self.url,
            "username":  self.username,
            "password":  self.password,
            "transport": self.transport,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RTSPSource":
        return cls(
            url       = d.get("url", ""),
            username  = d.get("username", ""),
            password  = d.get("password", ""),
            transport = d.get("transport", "tcp"),
        )

    def description(self) -> str:
        host = re.sub(r"^rtsps?://[^@/]*@?", "", self.url).split("/")[0]
        return f"RTSP  ({host})"

    @staticmethod
    def _inject_auth(url: str, username: str, password: str) -> str:
        if not username:
            return url
        url = re.sub(r"(rtsps?://)([^@/]+@)?", r"\1", url)
        return re.sub(
            r"(rtsps?://)",
            rf"\1{re.escape(username)}:{re.escape(password)}@",
            url,
        )


# ============================================================================
# Source factory
# ============================================================================

def source_from_dict(d: dict) -> BaseCameraSource:
    """Reconstruct the correct strategy subclass from a config dict."""
    t = d.get("type", "webcam")
    if t == WebcamSource.SOURCE_TYPE:
        return WebcamSource.from_dict(d)
    if t == IPCameraSource.SOURCE_TYPE:
        return IPCameraSource.from_dict(d)
    if t == RTSPSource.SOURCE_TYPE:
        return RTSPSource.from_dict(d)
    raise ValueError(f"Unknown camera type: {t!r}")


# ============================================================================
# DeviceScanner
# ============================================================================

class DeviceScanner:
    """
    Probe integer device indices to discover available webcams.

    Notes
    -----
    - Scanning can take 1–3 s depending on driver response.
    - Call from a background thread / QThread to avoid freezing the GUI.
    - Uses DirectShow on Windows for faster probing.
    """

    DEFAULT_MAX_INDEX = 8

    @staticmethod
    def scan(max_index: int = DEFAULT_MAX_INDEX) -> list[dict]:
        """
        Returns list of:
            {"index": int, "label": str, "width": int, "height": int}
        for every index where a camera responds.
        """
        backend = (
            cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        )
        found = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i, backend)
            if not cap.isOpened():
                cap.release()
                continue
            ok, frame = cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if ok and frame is not None:
                found.append({
                    "index":  i,
                    "label":  f"Camera {i}   [{w}×{h}]",
                    "width":  w,
                    "height": h,
                })
        return found


# ============================================================================
# CameraManager  (Context)
# ============================================================================

# How long to wait for network camera to open (seconds)
_NETWORK_OPEN_TIMEOUT = 8.0
# How long to wait for network camera to deliver first frame
_FRAME_TIMEOUT = 5.0
# Delay between reconnect attempts
_RECONNECT_DELAY = 2.0
_MAX_RECONNECT   = 5


class CameraManager:
    """
    Unified camera context.  Accepts any BaseCameraSource strategy and
    presents a single read/release interface to the rest of the application.

    Core logic (VisionThread, preview timers, etc.) only calls:
        .open()  → CameraConnectionResult
        .read()  → (ok: bool, frame: ndarray | None)
        .release()
        .is_open() → bool
        .test()  → CameraConnectionResult   (non-destructive)

    Resolution / FPS hints are passed here; the camera may ignore them.
    """

    def __init__(
        self,
        source: "BaseCameraSource | None" = None,
        width:  int   = 1280,
        height: int   = 720,
        fps:    int   = 30,
        reconnect: bool = True,
        # ---- Legacy kwargs so old CameraHandler(camera_type=...) still works ----
        camera_type:  str = "",
        device_index: int = 0,
        url:          str = "",
        **_kwargs,
    ) -> None:
        # Build source from legacy keyword args when source not provided directly
        if source is None or not isinstance(source, BaseCameraSource):
            source = _legacy_build_source(
                camera_type  or "webcam",
                device_index,
                url,
            )

        self._source    = source
        self.width      = width
        self.height     = height
        self.fps        = fps
        self._reconnect = reconnect

        self._cap:          Optional[cv2.VideoCapture] = None
        self._is_open:      bool  = False
        self._fail_count:   int   = 0
        self._actual_width  = width
        self._actual_height = height
        self._actual_fps    = float(fps)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> CameraConnectionResult:
        """
        Open the camera.  Returns CameraConnectionResult.
        Backward-compat: also returns bool-like (result.success) via __bool__.
        """
        # Pre-flight validation
        err = self._source.validate()
        if err:
            return err

        cap_source = self._source.build_capture_source()

        # Network cameras need a timeout wrapper
        is_network = isinstance(self._source, (IPCameraSource, RTSPSource))

        if is_network:
            result_holder: list[Optional[cv2.VideoCapture]] = [None]
            exc_holder: list[str] = [""]

            def _open_thread():
                try:
                    cap = cv2.VideoCapture(cap_source)
                    result_holder[0] = cap
                except Exception as exc:
                    exc_holder[0] = str(exc)

            t = threading.Thread(target=_open_thread, daemon=True)
            t.start()
            t.join(timeout=_NETWORK_OPEN_TIMEOUT)

            if t.is_alive():
                # Thread still blocking → timeout
                return CameraConnectionResult.fail(
                    CameraErrorCode.CONNECTION_TIMEOUT,
                    f"Timed out after {_NETWORK_OPEN_TIMEOUT:.0f}s opening {cap_source}",
                )
            if exc_holder[0]:
                return CameraConnectionResult.fail(
                    CameraErrorCode.UNKNOWN, exc_holder[0]
                )

            cap = result_holder[0]
        else:
            # Webcam — fast open via DirectShow / platform backend
            if isinstance(self._source, WebcamSource):
                cap = self._source.open_capture()
            else:
                cap = cv2.VideoCapture(cap_source)

        if cap is None or not cap.isOpened():
            code = self._classify_open_failure(cap_source)
            if cap:
                cap.release()
            return CameraConnectionResult.fail(code, f"Source: {cap_source}")

        # Request resolution / fps
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS,          self.fps)

        # Read actual values back
        aw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        afp = cap.get(cv2.CAP_PROP_FPS) or float(self.fps)

        # Verify we can actually get a frame
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return CameraConnectionResult.fail(
                CameraErrorCode.NO_FRAME,
                f"Opened {cap_source} but first read() failed",
            )

        self._cap           = cap
        self._is_open       = True
        self._fail_count    = 0
        self._actual_width  = aw
        self._actual_height = ah
        self._actual_fps    = afp

        return CameraConnectionResult.ok(frame=frame, w=aw, h=ah, fps=afp)

    def release(self) -> None:
        if self._cap:
            self._cap.release()
        self._cap     = None
        self._is_open = False

    def is_open(self) -> bool:
        return self._is_open and self._cap is not None and self._cap.isOpened()

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read one frame.  Auto-reconnects on transient failure."""
        if not self.is_open():
            if self._reconnect:
                return self._try_reconnect()
            return False, None

        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._fail_count += 1
            if self._reconnect and self._fail_count <= _MAX_RECONNECT:
                time.sleep(_RECONNECT_DELAY)
                return self._try_reconnect()
            return False, None

        self._fail_count = 0
        return True, frame

    # ------------------------------------------------------------------
    # Non-destructive connection test
    # ------------------------------------------------------------------

    def test(self) -> CameraConnectionResult:
        """
        Open → read one frame → release.
        Does NOT affect the current open state (if already open, uses it).
        """
        if self.is_open():
            ok, frame = self._cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                return CameraConnectionResult.ok(
                    frame=frame, w=w, h=h, fps=self._actual_fps
                )
            return CameraConnectionResult.fail(CameraErrorCode.NO_FRAME)

        # Temporarily open just for the test
        result = self.open()
        if result.success:
            self.release()
        return result

    # Alias for legacy callers: cam.test_connection() → (bool, str)
    def test_connection(self) -> tuple[bool, str]:
        r = self.test()
        return r.success, r.message

    # ------------------------------------------------------------------
    # Properties (backward compat)
    # ------------------------------------------------------------------

    @property
    def actual_width(self)  -> int:   return self._actual_width
    @property
    def actual_height(self) -> int:   return self._actual_height
    @property
    def actual_fps(self)    -> float: return self._actual_fps
    @property
    def frame_size(self)    -> tuple[int, int]:
        return (self._actual_width, self._actual_height)

    # Expose source for GUI introspection
    @property
    def source(self) -> BaseCameraSource:
        return self._source

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_open_failure(source) -> CameraErrorCode:
        """
        Heuristic: distinguish "not found" from "busy".
        On Windows, if DirectShow can list the device but open fails →
        likely busy.  We can't always tell without OS calls.
        """
        if isinstance(source, int):
            # Try a second open with a different backend
            cap2 = cv2.VideoCapture(source, cv2.CAP_ANY)
            opened = cap2.isOpened()
            cap2.release()
            # If even CAP_ANY can't open → device not found
            return CameraErrorCode.DEVICE_BUSY if opened else CameraErrorCode.DEVICE_NOT_FOUND
        # Network source: assume connection issue
        return CameraErrorCode.CONNECTION_TIMEOUT

    def _try_reconnect(self) -> tuple[bool, Optional[np.ndarray]]:
        self.release()
        time.sleep(_RECONNECT_DELAY)
        result = self.open()
        if result.success and self._cap:
            ok, frame = self._cap.read()
            return (True, frame) if ok else (False, None)
        return False, None

    def __repr__(self) -> str:
        return f"<CameraManager {self._source.description()}>"


# ============================================================================
# Legacy constructor support
# ============================================================================

def _legacy_build_source(
    camera_type: str,
    device_index: int,
    url: str,
) -> BaseCameraSource:
    """Build a strategy from the old keyword-argument style."""
    if camera_type == "webcam":
        return WebcamSource(device_index=device_index)
    if camera_type == "ip_camera":
        return IPCameraSource(url=url)
    if camera_type == "rtsp":
        return RTSPSource(url=url)
    return WebcamSource(device_index=device_index)


# Backward-compatibility alias — vision_thread.py uses CameraHandler
CameraHandler = CameraManager


# ============================================================================
# Convenience: build CameraManager directly from a config dict
# ============================================================================

def camera_manager_from_config(cfg: dict) -> CameraManager:
    """
    Build a fully-configured CameraManager from a camera_config.json dict.

    Expected keys: type, device_index, url, username, password,
                   width, height, fps, transport (RTSP only)
    """
    source = source_from_dict(cfg)
    return CameraManager(
        source    = source,
        width     = cfg.get("width",  1280),
        height    = cfg.get("height", 720),
        fps       = cfg.get("fps",    30),
        reconnect = cfg.get("reconnect_on_failure", True),
    )
