"""
run.py — Entry point สำหรับรันโปรแกรมจาก ai_production_monitor/
ใช้คำสั่ง:
    cd ai_production_monitor
    python run.py
หรือรันจาก VS / root folder ก็ได้ — sys.path จัดการให้อัตโนมัติ
"""

import sys
import os

# ── ชี้ sys.path ให้ import module ได้ไม่ว่าจะรันจากที่ไหน ──────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../ai_production_monitor
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# โหลด .env ถ้ามี python-dotenv
try:
    from dotenv import load_dotenv
    # ตอนรันเป็น EXE: sys.executable อยู่ที่ folder เดียวกับ .exe
    # ตอนรัน python run.py: ใช้ __file__ เหมือนเดิม
    if getattr(sys, "frozen", False):
        # PyInstaller EXE — หา .env ข้างๆ .exe
        _env_dir = os.path.dirname(sys.executable)
    else:
        _env_dir = os.path.dirname(os.path.abspath(__file__))
    _env_path = os.path.join(_env_dir, ".env")
    load_dotenv(_env_path)
    print(f"[INFO] .env loaded from {_env_path}")
except ImportError:
    print("[WARN] python-dotenv not installed, skipping .env load")

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
