"""
test_calibration.py
Run the CalibrationScreen in isolation for development/testing.

Usage
-----
    cd ai_production_monitor
    python test_calibration.py
"""

import sys
import os

# Make project root importable
sys.path.insert(0, os.path.dirname(__file__))

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QFont
from data.config_handler import ConfigHandler
from gui.calibration_screen import CalibrationScreen


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    from PyQt5.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(13, 27, 42))
    palette.setColor(QPalette.WindowText,      QColor(207, 216, 220))
    palette.setColor(QPalette.Base,            QColor(21, 32, 40))
    palette.setColor(QPalette.AlternateBase,   QColor(13, 27, 42))
    palette.setColor(QPalette.ToolTipBase,     QColor(0, 0, 0))
    palette.setColor(QPalette.ToolTipText,     QColor(207, 216, 220))
    palette.setColor(QPalette.Text,            QColor(207, 216, 220))
    palette.setColor(QPalette.Button,          QColor(38, 50, 56))
    palette.setColor(QPalette.ButtonText,      QColor(207, 216, 220))
    palette.setColor(QPalette.Highlight,       QColor(21, 101, 192))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    app.setFont(QFont("Consolas", 9))

    config = ConfigHandler()

    win = QMainWindow()
    win.setWindowTitle("Zone Calibration — Test")
    win.setMinimumSize(1280, 780)

    screen = CalibrationScreen(config)
    screen.calibration_saved.connect(
        lambda d: print(f"\n[SAVED] {len(d['zones'])} zones\n"
                        + "\n".join(
                            f"  Z{z['id']} '{z['name']}': "
                            f"({z['x1']},{z['y1']})→({z['x2']},{z['y2']})"
                            for z in d["zones"]
                        ))
    )
    screen.back_requested.connect(win.close)

    win.setCentralWidget(screen)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
