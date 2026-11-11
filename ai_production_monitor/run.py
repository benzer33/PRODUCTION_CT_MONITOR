"""
run.py — Entry point สำหรับรันโปรแกรมจาก ai_production_monitor/
ใช้คำสั่ง:
    cd ai_production_monitor
    python run.py
หรือรันจาก VS / root folder ก็ได้ — sys.path จัดการให้อัตโนมัติ
"""

import sys
import os
import logging

# ── เพิ่ม torch/lib เข้า DLL search path ก่อนทุกอย่าง (Windows CUDA fix) ──────
# แก้ปัญหา WinError 1114: c10.dll / cublas ใน Python 3.8+ บน Windows
if sys.platform == "win32":
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

# ── Setup file logging (เก็บ log ไว้ดู error บนเครื่องปลายทาง) ────────────
def _setup_logging():
    if getattr(sys, "frozen", False):
        log_dir = os.path.dirname(sys.executable)
    else:
        log_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(log_dir, "app.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("=== App started (frozen=%s) ===", getattr(sys, "frozen", False))
    logging.info("sys._MEIPASS = %s", getattr(sys, "_MEIPASS", "N/A"))
    return log_path

_log_path = _setup_logging()

# ── Preload torch บน MAIN THREAD (สำคัญมาก) ──────────────────────────────────
# WinError 1114 เกิดเมื่อ import torch ครั้งแรกใน QThread (เธรดรอง)
# ต้อง import บน main thread ก่อน เพื่อให้ DLL initialization สำเร็จ
# จากนั้น probe worker ใน QThread จะใช้ torch ที่โหลดไว้แล้วได้เลย
try:
    import torch as _torch_preload  # noqa: F401
    logging.info("torch preloaded on main thread: %s (cuda=%s)",
                 _torch_preload.__version__,
                 _torch_preload.cuda.is_available())
except Exception as _e:
    logging.warning("torch preload failed (จะใช้ MediaPipe เท่านั้น): %s", _e)

# ── ชี้ sys.path ให้ import module ได้ไม่ว่าจะรันจากที่ไหน ──────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../ai_production_monitor
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# โหลด .env ถ้ามี python-dotenv
try:
    from dotenv import load_dotenv
    if getattr(sys, "frozen", False):
        _env_dir = os.path.dirname(sys.executable)
    else:
        _env_dir = os.path.dirname(os.path.abspath(__file__))
    _env_path = os.path.join(_env_dir, ".env")
    load_dotenv(_env_path)
    logging.info(".env loaded from %s", _env_path)
except ImportError:
    logging.warning("python-dotenv not installed, skipping .env load")

logging.info("app.log is at: %s", _log_path)

from PyQt5.QtWidgets import QApplication
from data.config_handler import ConfigHandler
from data.database import DatabaseManager
from gui.main_window import MainWindow
from gui import theme
from utils.font_downloader import load_fonts_from_dir


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AI Production Cycle Monitor")

    config = ConfigHandler()
    db = DatabaseManager()   # init_db ถูกเรียกใน __init__ อัตโนมัติแล้ว

    # โหลดฟ้อนต์ที่ดาวน์โหลดไว้ใน fonts/ เข้า Qt ก่อนสร้าง window
    loaded = load_fonts_from_dir()
    if loaded:
        print(f"[INFO] Loaded custom fonts: {loaded}")

    # Apply persisted font preferences before any window is created
    theme.apply_font_settings(config.get_font_settings())

    window = MainWindow(config, db)
    window.show()

    print("[INFO] Application started")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
