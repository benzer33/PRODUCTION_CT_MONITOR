"""
gui/detector_settings_screen.py
Hand Detector Settings Screen

Layout
------
┌──────────────────────────────┬──────────────────────────────┐
│   SETTINGS (left)            │   STATUS (right)             │
│                              │                              │
│  Backend: [combo]            │  [ทดสอบการโหลด]              │
│                              │                              │
│  -- MediaPipe --             │  torch      : 2.x.x+cpu ✅  │
│  Detection conf: [spin]      │  CUDA       : ไม่พร้อม ❌    │
│  Tracking conf:  [spin]      │  ultralytics: 8.x.x ✅       │
│                              │  Device     : cpu            │
│  -- YOLO Pose --             │  Model file : ✅ พบแล้ว      │
│  Model:     [combo]          │                              │
│  Confidence:[spin]           │  [log area]                  │
│  Device:    [combo]          │                              │
│  เลื่อนมือ: [spin]            │                              │
│                              │                              │
│  [บันทึก]                    │                              │
└──────────────────────────────┴──────────────────────────────┘
"""

from __future__ import annotations

import os
import sys

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSizePolicy, QTextEdit, QVBoxLayout, QWidget,
)

from data.config_handler import ConfigHandler
from gui.theme import (
    FONT_FAMILY, FONT_SIZE_BODY, FONT_SIZE_CAPTION, FONT_SIZE_LABEL,
    FONT_SIZE_TITLE, make_font,
    btn_stylesheet, groupbox_stylesheet, input_stylesheet,
    CLR_MUTED,
)

_INPUT_STYLE = input_stylesheet()


# ---------------------------------------------------------------------------
# Background worker — runs detector probe without blocking GUI
# ---------------------------------------------------------------------------

class _ProbeWorker(QThread):
    """Run torch/ultralytics import checks in a background thread."""

    result_ready = pyqtSignal(dict)   # emits probe result dict

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Map nvidia-smi CUDA driver version → PyTorch wheel suffix
    # https://pytorch.org/get-started/locally/
    # ------------------------------------------------------------------
    _CUDA_WHL_MAP = [
        # (min_major, min_minor, whl_tag, label)
        (12, 8, "cu128", "CUDA 12.8"),
        (12, 4, "cu124", "CUDA 12.4"),
        (12, 1, "cu121", "CUDA 12.1"),
        (11, 8, "cu118", "CUDA 11.8"),
    ]

    @staticmethod
    def _probe_smi() -> tuple[str, str, str]:
        """
        Run nvidia-smi and parse GPU name + driver CUDA version.
        Returns (gpu_name, cuda_version_str, raw_first_line).
        e.g. ("NVIDIA GeForce MX450", "12.4", "Driver Version: 551.78  CUDA Version: 12.4")
        """
        import subprocess, re
        try:
            out = subprocess.check_output(
                ["nvidia-smi"], stderr=subprocess.DEVNULL, timeout=8,
                creationflags=0x08000000,   # CREATE_NO_WINDOW on Windows
            ).decode(errors="replace")
        except Exception:
            return "", "", ""

        cuda_ver = ""
        # nvidia-smi may show "CUDA Version:" or "CUDA UMD Version:" depending on driver
        m = re.search(r"CUDA(?:\s+UMD)?\s+Version:\s*([\d.]+)", out)
        if m:
            cuda_ver = m.group(1)

        gpu_name = ""
        # nvidia-smi table: | idx | name | ...
        m2 = re.search(r"\|\s*\d+\s+(NVIDIA[^\|]+?)\s+(?:Off|\|)", out)
        if m2:
            gpu_name = m2.group(1).strip()

        first_line = out.splitlines()[2] if len(out.splitlines()) > 2 else ""
        return gpu_name, cuda_ver, first_line

    @classmethod
    def _best_whl(cls, cuda_ver_str: str) -> tuple[str, str]:
        """Return (whl_tag, label) for the best matching PyTorch wheel, or ('', '')."""
        if not cuda_ver_str:
            return "", ""
        try:
            parts = cuda_ver_str.split(".")
            major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        except ValueError:
            return "", ""
        for min_maj, min_min, tag, label in cls._CUDA_WHL_MAP:
            if (major, minor) >= (min_maj, min_min):
                return tag, label
        return "", ""   # CUDA too old

    def run(self) -> None:
        result: dict = {
            "torch_version":        "ไม่พบ",
            "torch_ok":             False,
            "torch_is_cpu_only":    False,
            "cuda_available":       False,
            "cuda_device_name":     "",
            "cuda_device_count":    0,
            "ultralytics_version":  "ไม่พบ",
            "ultralytics_ok":       False,
            "model_file_found":     False,
            "model_file_path":      "",
            "device_actual":        "—",
            "inference_ok":         False,
            "inference_ms":         0.0,
            "error":                "",
            "install_hint":         "",
            # nvidia-smi extras
            "smi_gpu_name":         "",
            "smi_cuda_version":     "",
            "pip_whl_tag":          "",
            "pip_whl_label":        "",
        }

        # 0. Probe nvidia-smi (independent of torch — works even with CPU-only wheel)
        smi_gpu, smi_cuda, smi_line = self._probe_smi()
        result["smi_gpu_name"]     = smi_gpu
        result["smi_cuda_version"] = smi_cuda
        whl_tag, whl_label = self._best_whl(smi_cuda)
        result["pip_whl_tag"]   = whl_tag
        result["pip_whl_label"] = whl_label

        # 0b. Add torch/lib to DLL search path before import
        #     Needed when running inside QThread on Windows (DLL search path differs)
        try:
            import importlib.util as _ilu
            _spec = _ilu.find_spec("torch")
            if _spec and _spec.submodule_search_locations:
                _torch_root = list(_spec.submodule_search_locations)[0]
                _torch_lib = os.path.join(_torch_root, "lib")
                if os.path.isdir(_torch_lib):
                    os.add_dll_directory(_torch_lib)
        except Exception:
            pass

        # 1. Check torch
        try:
            import torch
            ver = torch.__version__
            result["torch_version"] = ver
            result["torch_ok"]      = True
            result["torch_is_cpu_only"] = "+cpu" in ver
            result["cuda_available"]    = torch.cuda.is_available()
            result["cuda_device_count"] = torch.cuda.device_count()
            if result["cuda_available"]:
                result["cuda_device_name"] = torch.cuda.get_device_name(0)
            elif result["torch_is_cpu_only"]:
                # Build install hint using the auto-detected whl_tag
                if whl_tag:
                    pip_cmd = (
                        f"pip install torch torchvision torchaudio "
                        f"--index-url https://download.pytorch.org/whl/{whl_tag}"
                    )
                    gpu_hint = f"สำหรับ {smi_gpu} ({whl_label})" if smi_gpu else whl_label
                else:
                    pip_cmd = (
                        "pip install torch torchvision torchaudio "
                        "--index-url https://download.pytorch.org/whl/cu121"
                    )
                    gpu_hint = "(ไม่พบ GPU / CUDA driver — ใช้ cu121 เป็นค่าเริ่มต้น)"
                result["install_hint"] = (
                    f"pip uninstall torch torchvision torchaudio -y\n{pip_cmd}"
                )
        except Exception as e:
            result["error"] += f"torch: {e}\n"

        # 2. Check ultralytics
        try:
            import ultralytics
            result["ultralytics_version"] = ultralytics.__version__
            result["ultralytics_ok"]      = True
        except Exception as e:
            result["error"] += f"ultralytics: {e}\n"

        # 3. Find model file
        model_name = self._cfg.get("yolo_model", "yolov8n-pose.pt")
        # Check _MEIPASS/models/ first (frozen bundle)
        meipass = getattr(sys, "_MEIPASS", "")
        candidates = []
        if meipass:
            candidates.append(os.path.join(meipass, "models", os.path.basename(model_name)))
        # ultralytics cache
        cache_dir = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "Ultralytics")
        candidates.append(os.path.join(cache_dir, os.path.basename(model_name)))
        # local
        candidates.append(model_name)
        for c in candidates:
            if os.path.isfile(c):
                result["model_file_found"] = True
                result["model_file_path"]  = c
                break

        # 4. Try to load detector and run a tiny inference
        if result["ultralytics_ok"] and result["torch_ok"]:
            try:
                import time
                import numpy as np
                from vision.hand_tracker import create_detector
                det = create_detector(self._cfg)
                dummy = np.zeros((64, 64, 3), dtype="uint8")
                t0 = time.perf_counter()
                det.process_all(dummy)
                result["inference_ms"] = (time.perf_counter() - t0) * 1000
                result["inference_ok"] = True
                # figure out actual device
                backend = self._cfg.get("backend", "mediapipe")
                if backend == "yolo_pose":
                    result["device_actual"] = self._cfg.get("yolo_device") or "cpu"
                else:
                    result["device_actual"] = "CPU (MediaPipe)"
                det.close()
            except Exception as e:
                result["error"] += f"inference: {e}\n"

        self.result_ready.emit(result)


# ---------------------------------------------------------------------------
# Main screen
# ---------------------------------------------------------------------------

class DetectorSettingsScreen(QWidget):
    """Standalone hand-detector settings + status screen."""

    back_requested = pyqtSignal()

    def __init__(self, config: ConfigHandler, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._worker: _ProbeWorker | None = None
        self._build_ui()
        self._load_config()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # Header
        hdr = QHBoxLayout()
        title = QLabel("🤖  HAND DETECTOR SETTINGS")
        title.setFont(make_font(FONT_SIZE_TITLE, bold=True))
        title.setStyleSheet("color: #00bcd4;")
        hdr.addWidget(title)
        hdr.addStretch()
        btn_back = QPushButton("← Back")
        btn_back.setStyleSheet(btn_stylesheet("#263238"))
        btn_back.setFixedHeight(32)
        btn_back.clicked.connect(self.back_requested)
        hdr.addWidget(btn_back)
        root.addLayout(hdr)

        # Body
        body = QHBoxLayout()
        body.setSpacing(14)
        body.addWidget(self._build_settings_panel(), stretch=1)
        body.addWidget(self._build_status_panel(),   stretch=1)
        root.addLayout(body)

    # ---- Left: settings ----

    def _build_settings_panel(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet(
            "QFrame { background:#0d1b2a; border:1px solid #1e3040; border-radius:6px; }"
        )
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(14)

        # Backend
        grp_back = self._grp("Backend")
        f_back = QFormLayout(grp_back)
        self._combo_backend = QComboBox()
        self._combo_backend.addItem("MediaPipe Hands", "mediapipe")
        self._combo_backend.addItem("YOLO Pose (ยาว/มุมสูง)", "yolo_pose")
        self._combo_backend.setStyleSheet(_INPUT_STYLE)
        self._combo_backend.setFont(make_font(FONT_SIZE_BODY))
        self._combo_backend.currentIndexChanged.connect(self._on_backend_changed)
        f_back.addRow(self._lbl("Backend:"), self._combo_backend)
        lay.addWidget(grp_back)

        # MediaPipe params
        self._grp_mp = self._grp("MediaPipe — ตั้งค่า")
        f_mp = QFormLayout(self._grp_mp)
        self._spin_mp_detect = self._dspin(0.1, 1.0, 0.05, 0.6)
        self._spin_mp_track  = self._dspin(0.1, 1.0, 0.05, 0.5)
        f_mp.addRow(self._lbl("Detection conf:"), self._spin_mp_detect)
        f_mp.addRow(self._lbl("Tracking conf:"),  self._spin_mp_track)
        lay.addWidget(self._grp_mp)

        # YOLO params
        self._grp_yolo = self._grp("YOLO Pose — ตั้งค่า")
        f_yolo = QFormLayout(self._grp_yolo)

        self._combo_model = QComboBox()
        for lbl, val in [
            ("Nano  (เร็ว)",   "yolov8n-pose.pt"),
            ("Small",          "yolov8s-pose.pt"),
            ("Medium",         "yolov8m-pose.pt"),
            ("v11 Nano",       "yolo11n-pose.pt"),
            ("v11 Small",      "yolo11s-pose.pt"),
        ]:
            self._combo_model.addItem(lbl, val)
        self._combo_model.setStyleSheet(_INPUT_STYLE)
        f_yolo.addRow(self._lbl("Model:"), self._combo_model)

        self._spin_yolo_conf = self._dspin(0.1, 1.0, 0.05, 0.4)
        f_yolo.addRow(self._lbl("Confidence:"), self._spin_yolo_conf)

        self._combo_device = QComboBox()
        self._combo_device.addItem("CPU (แนะนำ)", "cpu")
        self._combo_device.addItem("Auto",         "")
        self._combo_device.addItem("CUDA (GPU)",   "cuda")
        self._combo_device.setStyleSheet(_INPUT_STYLE)
        f_yolo.addRow(self._lbl("Device:"), self._combo_device)

        self._spin_extend = self._dspin(0.0, 1.0, 0.05, 0.35)
        self._spin_extend.setToolTip(
            "0 = ใช้ข้อมือตรงๆ\n"
            "0.35 = เลื่อน 35% ของ forearm ออกไปหน้ามือ (ค่าแนะนำ)"
        )
        f_yolo.addRow(self._lbl("เลื่อนมือ:"), self._spin_extend)
        lay.addWidget(self._grp_yolo)

        lay.addStretch()

        btn_save = QPushButton("💾  บันทึกการตั้งค่า")
        btn_save.setStyleSheet(btn_stylesheet("#0d47a1"))
        btn_save.setFixedHeight(40)
        btn_save.clicked.connect(self._save)
        lay.addWidget(btn_save)

        self._on_backend_changed()
        return panel

    # ---- Right: status ----

    def _build_status_panel(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet(
            "QFrame { background:#0d1b2a; border:1px solid #1e3040; border-radius:6px; }"
        )
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        status_title = QLabel("📊  STATUS")
        status_title.setFont(make_font(FONT_SIZE_LABEL, bold=True))
        status_title.setStyleSheet("color: #00bcd4;")
        lay.addWidget(status_title)

        # Status rows
        grp_status = self._grp("ผลการตรวจสอบ")
        f_st = QFormLayout(grp_status)
        f_st.setSpacing(8)

        self._lbl_torch    = self._status_lbl("—")
        self._lbl_cuda     = self._status_lbl("—")
        self._lbl_smi_gpu  = self._status_lbl("—")
        self._lbl_smi_cuda = self._status_lbl("—")
        self._lbl_ulytic   = self._status_lbl("—")
        self._lbl_device   = self._status_lbl("—")
        self._lbl_model    = self._status_lbl("—")
        self._lbl_infer    = self._status_lbl("—")

        f_st.addRow(self._lbl("torch:"),          self._lbl_torch)
        f_st.addRow(self._lbl("CUDA (torch):"),   self._lbl_cuda)
        f_st.addRow(self._lbl("GPU (smi):"),      self._lbl_smi_gpu)
        f_st.addRow(self._lbl("CUDA driver:"),    self._lbl_smi_cuda)
        f_st.addRow(self._lbl("ultralytics:"),    self._lbl_ulytic)
        f_st.addRow(self._lbl("Device ที่ใช้:"),  self._lbl_device)
        f_st.addRow(self._lbl("Model file:"),     self._lbl_model)
        f_st.addRow(self._lbl("Inference:"),      self._lbl_infer)
        lay.addWidget(grp_status)

        # Test button
        self._btn_test = QPushButton("🔍  ทดสอบการโหลด")
        self._btn_test.setStyleSheet(btn_stylesheet("#00695c"))
        self._btn_test.setFixedHeight(38)
        self._btn_test.clicked.connect(self._run_probe)
        lay.addWidget(self._btn_test)

        # CPU-only warning banner (hidden by default, updated after probe)
        self._banner_cpu = QFrame()
        self._banner_cpu.setStyleSheet(
            "QFrame { background:#3e2723; border:1px solid #ff8f00; border-radius:4px; }"
        )
        banner_lay = QVBoxLayout(self._banner_cpu)
        banner_lay.setContentsMargins(10, 8, 10, 8)
        banner_lay.setSpacing(6)

        self._banner_warn_lbl = QLabel(
            "⚠️  <b>torch ที่ติดตั้งอยู่เป็น CPU-only</b> — ต้องติดตั้งใหม่เพื่อใช้ CUDA"
        )
        self._banner_warn_lbl.setFont(make_font(FONT_SIZE_LABEL, bold=True))
        self._banner_warn_lbl.setStyleSheet("color: #ffb300; background: transparent; border: none;")
        self._banner_warn_lbl.setWordWrap(True)
        banner_lay.addWidget(self._banner_warn_lbl)

        self._banner_cmd_lbl = QLabel("")
        self._banner_cmd_lbl.setTextFormat(Qt.RichText)
        self._banner_cmd_lbl.setStyleSheet("background: transparent; border: none;")
        self._banner_cmd_lbl.setWordWrap(True)
        banner_lay.addWidget(self._banner_cmd_lbl)

        btn_copy_pip = QPushButton("📋  Copy คำสั่ง pip install")
        btn_copy_pip.setStyleSheet(btn_stylesheet("#e65100"))
        btn_copy_pip.setFixedHeight(28)
        btn_copy_pip.clicked.connect(self._copy_pip_cmd)
        banner_lay.addWidget(btn_copy_pip)

        self._banner_cpu.setVisible(False)
        self._pip_cmd_to_copy = ""   # filled by probe result
        lay.addWidget(self._banner_cpu)

        # Log area
        log_lbl = QLabel("Log:")
        log_lbl.setFont(make_font(FONT_SIZE_CAPTION))
        log_lbl.setStyleSheet(f"color: {CLR_MUTED};")
        lay.addWidget(log_lbl)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(make_font(FONT_SIZE_CAPTION))
        self._log.setStyleSheet(
            "QTextEdit { background:#060f17; color:#90a4ae; border:1px solid #1e3040; "
            "border-radius:4px; padding:4px; }"
        )
        self._log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self._log)

        return panel

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _grp(title: str) -> QGroupBox:
        g = QGroupBox(title)
        g.setStyleSheet(groupbox_stylesheet())
        return g

    @staticmethod
    def _lbl(text: str) -> QLabel:
        l = QLabel(text)
        l.setFont(make_font(FONT_SIZE_LABEL))
        l.setStyleSheet(f"color: {CLR_MUTED};")
        return l

    @staticmethod
    def _status_lbl(text: str) -> QLabel:
        l = QLabel(text)
        l.setFont(make_font(FONT_SIZE_BODY, bold=True))
        l.setStyleSheet("color: #607d8b;")
        return l

    @staticmethod
    def _dspin(lo: float, hi: float, step: float, val: float) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(2)
        s.setValue(val)
        s.setStyleSheet(_INPUT_STYLE)
        return s

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _copy_pip_cmd(self) -> None:
        from PyQt5.QtWidgets import QApplication
        cmd = self._pip_cmd_to_copy or (
            "pip uninstall torch torchvision torchaudio -y\n"
            "pip install torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu121"
        )
        QApplication.clipboard().setText(cmd)
        self._log.append("📋 คัดลอกคำสั่ง pip แล้ว — วางใน Command Prompt / PowerShell ได้เลย")

    def _on_backend_changed(self) -> None:
        is_yolo = self._combo_backend.currentData() == "yolo_pose"
        self._grp_mp.setVisible(not is_yolo)
        self._grp_yolo.setVisible(is_yolo)

    def _load_config(self) -> None:
        cfg = self._config.get_detector_config()
        backend = cfg.get("backend", "mediapipe")
        for i in range(self._combo_backend.count()):
            if self._combo_backend.itemData(i) == backend:
                self._combo_backend.setCurrentIndex(i)
                break
        self._spin_mp_detect.setValue(cfg.get("mp_detection_conf",  0.6))
        self._spin_mp_track.setValue( cfg.get("mp_tracking_conf",   0.5))
        yolo_model = cfg.get("yolo_model", "yolov8n-pose.pt")
        for i in range(self._combo_model.count()):
            if self._combo_model.itemData(i) == yolo_model:
                self._combo_model.setCurrentIndex(i)
                break
        self._spin_yolo_conf.setValue(cfg.get("yolo_confidence",  0.4))
        yolo_dev = cfg.get("yolo_device", "cpu")
        for i in range(self._combo_device.count()):
            if self._combo_device.itemData(i) == yolo_dev:
                self._combo_device.setCurrentIndex(i)
                break
        self._spin_extend.setValue(cfg.get("yolo_hand_extend", 0.35))
        self._on_backend_changed()

    def _save(self) -> None:
        cfg = {
            "backend":           self._combo_backend.currentData(),
            "mp_max_hands":      2,
            "mp_detection_conf": self._spin_mp_detect.value(),
            "mp_tracking_conf":  self._spin_mp_track.value(),
            "yolo_model":        self._combo_model.currentData(),
            "yolo_confidence":   self._spin_yolo_conf.value(),
            "yolo_device":       self._combo_device.currentData(),
            "yolo_kp_conf_min":  0.3,
            "yolo_hand_extend":  self._spin_extend.value(),
        }
        self._config.set_detector_config(cfg)
        self._log.append("✅ บันทึกการตั้งค่าแล้ว")

    def _run_probe(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        self._btn_test.setEnabled(False)
        self._btn_test.setText("⏳  กำลังทดสอบ…")
        self._log.clear()
        self._log.append("กำลังโหลด torch / ultralytics…")

        # Reset labels
        for lbl in (self._lbl_torch, self._lbl_cuda, self._lbl_smi_gpu,
                    self._lbl_smi_cuda, self._lbl_ulytic,
                    self._lbl_device, self._lbl_model, self._lbl_infer):
            lbl.setText("…")
            lbl.setStyleSheet("color: #607d8b;")

        # Build cfg from current UI (not saved config)
        cfg = {
            "backend":          self._combo_backend.currentData(),
            "yolo_model":       self._combo_model.currentData(),
            "yolo_confidence":  self._spin_yolo_conf.value(),
            "yolo_device":      self._combo_device.currentData(),
            "yolo_kp_conf_min": 0.3,
            "yolo_hand_extend": self._spin_extend.value(),
            "mp_detection_conf": self._spin_mp_detect.value(),
            "mp_tracking_conf":  self._spin_mp_track.value(),
        }
        self._worker = _ProbeWorker(cfg, self)
        self._worker.result_ready.connect(self._on_probe_done)
        self._worker.start()

    def _on_probe_done(self, r: dict) -> None:
        self._btn_test.setEnabled(True)
        self._btn_test.setText("🔍  ทดสอบการโหลด")

        def _ok(lbl: QLabel, text: str) -> None:
            lbl.setText(text)
            lbl.setStyleSheet("color: #66bb6a; font-weight: bold;")

        def _fail(lbl: QLabel, text: str) -> None:
            lbl.setText(text)
            lbl.setStyleSheet("color: #ef9a9a; font-weight: bold;")

        def _warn(lbl: QLabel, text: str) -> None:
            lbl.setText(text)
            lbl.setStyleSheet("color: #ffb300; font-weight: bold;")

        def _info(lbl: QLabel, text: str) -> None:
            lbl.setText(text)
            lbl.setStyleSheet("color: #4dd0e1; font-weight: bold;")

        # nvidia-smi GPU name
        smi_gpu  = r.get("smi_gpu_name", "")
        smi_cuda = r.get("smi_cuda_version", "")
        whl_tag  = r.get("pip_whl_tag", "")
        whl_lbl  = r.get("pip_whl_label", "")

        if smi_gpu:
            _ok(self._lbl_smi_gpu, smi_gpu)
        else:
            _warn(self._lbl_smi_gpu, "ไม่พบ (nvidia-smi ไม่ตอบ)")

        if smi_cuda:
            if whl_tag:
                _ok(self._lbl_smi_cuda, f"{smi_cuda}  →  wheel: {whl_tag}")
            else:
                _warn(self._lbl_smi_cuda, f"{smi_cuda}  (เก่าเกิน / ไม่รองรับ)")
        else:
            _warn(self._lbl_smi_cuda, "ไม่พบ")

        # torch
        if r["torch_ok"]:
            if r["torch_is_cpu_only"]:
                _warn(self._lbl_torch, f"⚠️ {r['torch_version']}  (CPU-only)")
            else:
                _ok(self._lbl_torch, f"✅ {r['torch_version']}")
        else:
            _fail(self._lbl_torch, "❌ โหลดไม่ได้")

        # CUDA (torch)
        if r["cuda_available"]:
            _ok(self._lbl_cuda, f"✅ พร้อม — {r['cuda_device_name']}")
        elif r["torch_is_cpu_only"]:
            _warn(self._lbl_cuda, "⚠️ ไม่พร้อม — torch เป็น CPU-only")
        else:
            _fail(self._lbl_cuda, f"❌ ไม่พร้อม (device_count={r.get('cuda_device_count',0)})")

        # ultralytics
        if r["ultralytics_ok"]:
            _ok(self._lbl_ulytic, f"✅ {r['ultralytics_version']}")
        else:
            _fail(self._lbl_ulytic, "❌ โหลดไม่ได้")

        # device
        _info(self._lbl_device, r["device_actual"] or "—")

        # Update banner with auto-detected pip command
        is_cpu_only = bool(r.get("torch_is_cpu_only"))
        self._banner_cpu.setVisible(is_cpu_only)
        if is_cpu_only:
            hint = r.get("install_hint", "")
            # Store for copy button
            self._pip_cmd_to_copy = hint
            # Update banner subtitle to show which GPU/CUDA detected
            if smi_gpu and whl_tag:
                subtitle = (
                    f"ตรวจพบ: <b>{smi_gpu}</b>  |  CUDA driver: <b>{smi_cuda}</b>  "
                    f"→  แนะนำ wheel: <b style='color:#a5d6a7'>{whl_tag}</b>"
                )
            elif smi_gpu:
                subtitle = f"ตรวจพบ GPU: <b>{smi_gpu}</b>  (CUDA driver ไม่พบ)"
            else:
                subtitle = "ไม่พบ GPU / nvidia-smi — ใช้ cu121 เป็นค่าเริ่มต้น"
            self._banner_warn_lbl.setText(
                f"⚠️  <b>torch ที่ติดตั้งอยู่เป็น CPU-only</b><br>"
                f"<span style='font-size:9pt; color:#cfd8dc;'>{subtitle}</span>"
            )
            # Show pip command in banner
            lines = hint.replace("\n", "<br>")
            self._banner_cmd_lbl.setText(
                f"<code style='color:#a5d6a7; font-size:9pt;'>{lines}</code>"
            )

        # model file
        if r["model_file_found"]:
            fname = os.path.basename(r["model_file_path"])
            _ok(self._lbl_model, f"✅ {fname}")
            self._log.append(f"Model: {r['model_file_path']}")
        else:
            _fail(self._lbl_model, "❌ ไม่พบไฟล์ .pt")
            self._log.append("⚠️ ไม่พบไฟล์ model — YOLO จะพยายาม download อัตโนมัติ")

        # inference
        if r["inference_ok"]:
            _ok(self._lbl_infer, f"✅ {r['inference_ms']:.1f} ms")
        else:
            _fail(self._lbl_infer, "❌ inference ล้มเหลว")

        # errors
        if r["error"]:
            self._log.append(f"\n⚠️ Errors:\n{r['error']}")

        # install hint (CPU-only torch detected)
        if r.get("install_hint"):
            self._log.append(f"\n{'─'*50}\n{r['install_hint']}\n{'─'*50}")
        elif not r["error"]:
            self._log.append("\n✅ ทุกอย่างพร้อมใช้งาน")
