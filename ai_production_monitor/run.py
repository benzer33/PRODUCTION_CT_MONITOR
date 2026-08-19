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
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    print("[INFO] .env loaded")
except ImportError:
    print("[WARN] python-dotenv not installed, skipping .env load")

from PyQt5.QtWidgets import QApplication
from data.config_handler import ConfigHandler
from data.database import DatabaseManager
from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AI Production Cycle Monitor")

    config = ConfigHandler()
    db = DatabaseManager()   # init_db ถูกเรียกใน __init__ อัตโนมัติแล้ว

    window = MainWindow(config, db)
    window.show()

    print("[INFO] Application started")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
