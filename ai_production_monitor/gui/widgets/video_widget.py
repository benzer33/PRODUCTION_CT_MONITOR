"""
gui/widgets/video_widget.py
Custom QWidget for displaying OpenCV frames with correct aspect-ratio scaling.

Features
--------
- Accepts BGR numpy arrays via set_frame() (called from signal handler)
- Maintains aspect ratio by letterboxing
- Optionally draws a "no signal" placeholder
- Thread-safe: convert np.ndarray → QImage in the slot (GUI thread only)
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPainter, QPixmap, QColor, QFont
from PyQt5.QtWidgets import QWidget, QSizePolicy


class VideoWidget(QWidget):
    """
    Lightweight video frame renderer.

    Call set_frame(bgr_ndarray) from a Qt slot connected to
    VisionThread.frame_ready to display live video.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pixmap:  QPixmap | None = None
        self._no_signal = True

        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #0a0a0a;")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_frame(self, frame: np.ndarray) -> None:
        """Convert BGR ndarray to QPixmap and trigger repaint."""
        if frame is None or frame.size == 0:
            return

        h, w = frame.shape[:2]
        rgb = frame[..., ::-1].copy()   # BGR → RGB
        qimg = QImage(
            rgb.data, w, h,
            w * 3,
            QImage.Format_RGB888,
        )
        self._pixmap  = QPixmap.fromImage(qimg)
        self._no_signal = False
        self.update()

    def clear(self) -> None:
        self._pixmap    = None
        self._no_signal = True
        self.update()

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        if self._pixmap is None or self._no_signal:
            self._draw_no_signal(painter)
            return

        # Letterbox scaling — keep aspect ratio
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        x = (self.width()  - scaled.width())  // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)

    def _draw_no_signal(self, painter: QPainter) -> None:
        painter.fillRect(self.rect(), QColor(10, 10, 10))
        painter.setPen(QColor(60, 60, 60))
        painter.setFont(QFont("Consolas", 14))
        painter.drawText(
            self.rect(),
            Qt.AlignCenter,
            "[ NO SIGNAL ]",
        )

    def sizeHint(self) -> QSize:
        return QSize(640, 360)
