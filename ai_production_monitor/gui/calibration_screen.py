"""
gui/calibration_screen.py
Zone Calibration Screen — draw rectangular ROI zones on a live camera feed.

Features
--------
- Live camera preview (QLabel-based, updated via QTimer)
- Click-drag to draw rectangular zones; rubber-band feedback while dragging
- Dynamic number of zones (not fixed — add as many as needed)
- Per-zone: editable name + colour picker
- Undo (remove last zone) / Clear All
- Save → station_config.json  {"station_id": "...", "zones": [...]}
- Load → reload existing config and redraw all zones
- All coordinates are stored as pixel values relative to the ORIGINAL
  camera frame (not the scaled display), so they stay valid regardless
  of window size.

Coordinate system note
----------------------
The canvas widget shows the camera frame scaled to fit the QLabel.
Mouse events are in widget-pixel space; the class converts them to
frame-pixel space before storing, and converts back to widget-pixel
space before drawing.  This keeps the stored JSON coordinates
resolution-independent (tied to the camera resolution, not the GUI).
"""

from __future__ import annotations

import json
import os
import copy
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import (
    QPoint, QRect, QSize, Qt, QTimer, pyqtSignal,
)
from PyQt5.QtGui import (
    QColor, QFont, QImage, QPainter, QPen, QPixmap,
    QBrush, QCursor,
)
from PyQt5.QtWidgets import (
    QColorDialog, QFileDialog, QFrame, QGroupBox,
    QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy,
    QSpinBox, QVBoxLayout, QWidget,
)

from data.config_handler import ConfigHandler
from vision.camera_handler import CameraHandler


# ---------------------------------------------------------------------------
# Data model for one Zone
# ---------------------------------------------------------------------------

class ZoneRect:
    """
    One calibrated zone.

    Coordinates (x1, y1, x2, y2) are always in *frame* pixel space
    (top-left / bottom-right, normalised so x1<x2, y1<y2).
    """

    _PALETTE = [
        "#1565C0",  # blue
        "#00838F",  # teal
        "#558B2F",  # green
        "#6A1B9A",  # purple
        "#E65100",  # orange
        "#AD1457",  # pink
        "#00695C",  # dark teal
        "#283593",  # indigo
    ]

    def __init__(
        self,
        zone_id: int,
        name: str,
        x1: int, y1: int, x2: int, y2: int,
        color: str | None = None,
    ) -> None:
        self.zone_id = zone_id
        self.name    = name
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        self.color = color or self._PALETTE[(zone_id - 1) % len(self._PALETTE)]

    # ------------------------------------------------------------------ #
    def width(self)  -> int: return self.x2 - self.x1
    def height(self) -> int: return self.y2 - self.y1

    def is_valid(self, min_size: int = 20) -> bool:
        return self.width() >= min_size and self.height() >= min_size

    def contains_point(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def to_dict(self) -> dict:
        return {
            "id":    self.zone_id,
            "name":  self.name,
            "color": self.color,
            "x1":    self.x1,
            "y1":    self.y1,
            "x2":    self.x2,
            "y2":    self.y2,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ZoneRect":
        return cls(
            zone_id = d["id"],
            name    = d.get("name", f"Zone {d['id']}"),
            x1      = d["x1"], y1 = d["y1"],
            x2      = d["x2"], y2 = d["y2"],
            color   = d.get("color"),
        )

    def qcolor(self) -> QColor:
        return QColor(self.color)


# ---------------------------------------------------------------------------
# Drawing Canvas
# ---------------------------------------------------------------------------

class ZoneCanvas(QLabel):
    """
    QLabel that renders the camera frame and all defined zones, and handles
    mouse-drag to draw new rectangular zones.

    Signals
    -------
    zone_drawn(ZoneRect)    — emitted when the user finishes drawing a zone
    zone_hovered(int|None)  — zone_id under cursor, or None
    """

    zone_drawn   = pyqtSignal(object)   # ZoneRect
    zone_hovered = pyqtSignal(object)   # int or None

    MIN_ZONE_PX = 15   # minimum side length in frame pixels

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(480, 270)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self.setStyleSheet("background-color: #0a0a0a;")

        # Stored zones (managed externally via set_zones / clear)
        self._zones: list[ZoneRect] = []

        # State for in-progress drag
        self._drag_start: Optional[QPoint] = None   # widget coords
        self._drag_end:   Optional[QPoint] = None   # widget coords
        self._dragging    = False

        # Latest background pixmap (original resolution stored separately)
        self._bg_pixmap: Optional[QPixmap] = None
        self._frame_w: int = 1280
        self._frame_h: int = 720

        # Hover tracking
        self._hovered_id: Optional[int] = None

        self._show_placeholder()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_background(self, frame: np.ndarray) -> None:
        """Set/update the background image from a BGR numpy frame."""
        h, w = frame.shape[:2]
        self._frame_w = w
        self._frame_h = h
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg  = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self._bg_pixmap = QPixmap.fromImage(qimg)
        self._refresh()

    def set_zones(self, zones: list[ZoneRect]) -> None:
        self._zones = list(zones)
        self._refresh()

    def get_zones(self) -> list[ZoneRect]:
        return list(self._zones)

    def clear_zones(self) -> None:
        self._zones.clear()
        self._dragging    = False
        self._drag_start  = None
        self._drag_end    = None
        self._refresh()

    def remove_last_zone(self) -> Optional[ZoneRect]:
        if self._zones:
            removed = self._zones.pop()
            self._refresh()
            return removed
        return None

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._bg_pixmap:
            self._drag_start = event.pos()
            self._drag_end   = event.pos()
            self._dragging   = True

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            self._drag_end = event.pos()
            self._refresh()   # live rubber-band

        # Hover detection
        if self._bg_pixmap:
            fx, fy = self._widget_to_frame(event.pos())
            hov = None
            for z in reversed(self._zones):   # top-most zone first
                if z.contains_point(fx, fy):
                    hov = z.zone_id
                    break
            if hov != self._hovered_id:
                self._hovered_id = hov
                self.zone_hovered.emit(hov)
                self._refresh()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            if self._drag_start and self._drag_end:
                # Convert both corners to frame coords
                fx1, fy1 = self._widget_to_frame(self._drag_start)
                fx2, fy2 = self._widget_to_frame(self._drag_end)
                self._drag_start = None
                self._drag_end   = None

                # Ignore tiny accidental clicks
                if (abs(fx2 - fx1) >= self.MIN_ZONE_PX and
                        abs(fy2 - fy1) >= self.MIN_ZONE_PX):
                    # Assign next available ID
                    used_ids = {z.zone_id for z in self._zones}
                    new_id   = 1
                    while new_id in used_ids:
                        new_id += 1
                    zone = ZoneRect(
                        zone_id = new_id,
                        name    = f"Zone {new_id}",
                        x1=fx1, y1=fy1, x2=fx2, y2=fy2,
                    )
                    self._zones.append(zone)
                    self.zone_drawn.emit(zone)
                self._refresh()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        """Composite: background → zones → rubber-band → labels."""
        super().paintEvent(event)   # draws the QLabel pixmap

    def _refresh(self) -> None:
        """Re-render everything onto the QLabel pixmap."""
        if self._bg_pixmap is None:
            self._show_placeholder()
            return

        # Scale background to fit label while preserving aspect ratio
        base = self._bg_pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        canvas = QPixmap(self.size())
        canvas.fill(QColor(10, 10, 10))

        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)

        # Letterbox offset
        ox = (self.width()  - base.width())  // 2
        oy = (self.height() - base.height()) // 2
        painter.drawPixmap(ox, oy, base)

        # Scale factors (frame → widget-on-screen)
        sx = base.width()  / self._frame_w
        sy = base.height() / self._frame_h

        # Draw completed zones
        for zone in self._zones:
            self._draw_zone(painter, zone, ox, oy, sx, sy,
                            hovered=(zone.zone_id == self._hovered_id))

        # Draw rubber-band in-progress rect
        if self._dragging and self._drag_start and self._drag_end:
            r = QRect(self._drag_start, self._drag_end).normalized()
            pen = QPen(QColor("#00e5ff"), 1, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(0, 229, 255, 30)))
            painter.drawRect(r)

        painter.end()
        self.setPixmap(canvas)

    @staticmethod
    def _draw_zone(
        painter: QPainter,
        zone: ZoneRect,
        ox: int, oy: int,
        sx: float, sy: float,
        hovered: bool = False,
    ) -> None:
        """Draw one zone rectangle with label onto painter."""
        wx1 = int(zone.x1 * sx) + ox
        wy1 = int(zone.y1 * sy) + oy
        wx2 = int(zone.x2 * sx) + ox
        wy2 = int(zone.y2 * sy) + oy

        qc = zone.qcolor()
        fill = QColor(qc.red(), qc.green(), qc.blue(), 55 if not hovered else 90)

        pen_width = 3 if hovered else 2
        painter.setPen(QPen(qc, pen_width))
        painter.setBrush(QBrush(fill))
        painter.drawRect(QRect(QPoint(wx1, wy1), QPoint(wx2, wy2)))

        # Corner handles (visual affordance)
        handle = 6
        painter.setBrush(QBrush(qc))
        painter.setPen(Qt.NoPen)
        for hx, hy in [(wx1, wy1), (wx2, wy1), (wx1, wy2), (wx2, wy2)]:
            painter.drawRect(hx - handle // 2, hy - handle // 2, handle, handle)

        # Zone ID badge (top-left)
        badge_w, badge_h = 22, 18
        painter.setBrush(QBrush(qc))
        painter.setPen(Qt.NoPen)
        painter.drawRect(wx1, wy1, badge_w, badge_h)
        painter.setPen(QPen(Qt.white))
        painter.setFont(QFont("Consolas", 8, QFont.Bold))
        painter.drawText(QRect(wx1, wy1, badge_w, badge_h),
                         Qt.AlignCenter, str(zone.zone_id))

        # Zone name label below the top edge
        label_rect = QRect(wx1 + badge_w + 2, wy1, wx2 - wx1 - badge_w - 2, badge_h)
        painter.setPen(QPen(Qt.white))
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(label_rect, Qt.AlignVCenter | Qt.AlignLeft,
                         f" {zone.name}")

        # Dimension tooltip (bottom-right corner, shown on hover)
        if hovered:
            dim_text = f"{zone.width()}×{zone.height()}px"
            painter.setFont(QFont("Consolas", 7))
            painter.setPen(QPen(QColor(200, 200, 200)))
            painter.drawText(
                QRect(wx1, wy2 - 14, wx2 - wx1, 14),
                Qt.AlignRight | Qt.AlignVCenter,
                dim_text + " "
            )

    def _show_placeholder(self) -> None:
        pm = QPixmap(self.size() if not self.size().isEmpty()
                     else QSize(640, 360))
        pm.fill(QColor(10, 10, 10))
        p = QPainter(pm)
        p.setPen(QColor(60, 60, 60))
        p.setFont(QFont("Consolas", 11))
        p.drawText(pm.rect(), Qt.AlignCenter,
                   "[ No camera frame ]\n\nClick  ▶ Start Preview  or  📸 Capture Frame")
        p.end()
        self.setPixmap(pm)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _letterbox_origin(self) -> tuple[int, int, float, float]:
        """Return (ox, oy, sx, sy) for the current displayed pixmap."""
        if self._bg_pixmap is None:
            return 0, 0, 1.0, 1.0
        scaled_w = int(self._frame_w * min(self.width()  / self._frame_w,
                                           self.height() / self._frame_h))
        scaled_h = int(self._frame_h * min(self.width()  / self._frame_w,
                                           self.height() / self._frame_h))
        ox = (self.width()  - scaled_w) // 2
        oy = (self.height() - scaled_h) // 2
        sx = scaled_w / self._frame_w
        sy = scaled_h / self._frame_h
        return ox, oy, sx, sy

    def _widget_to_frame(self, pt: QPoint) -> tuple[int, int]:
        """Convert widget pixel (x,y) → frame pixel (x,y)."""
        ox, oy, sx, sy = self._letterbox_origin()
        fx = int(max(0, min((pt.x() - ox) / sx, self._frame_w - 1)))
        fy = int(max(0, min((pt.y() - oy) / sy, self._frame_h - 1)))
        return fx, fy


# ---------------------------------------------------------------------------
# Zone Editor Row  (name + colour + delete button per zone)
# ---------------------------------------------------------------------------

class ZoneEditorRow(QFrame):
    """
    One row in the zone list panel.
    Lets the user rename a zone and change its border colour.
    """

    name_changed   = pyqtSignal(int, str)    # (zone_id, new_name)
    color_changed  = pyqtSignal(int, str)    # (zone_id, hex_color)
    delete_clicked = pyqtSignal(int)         # zone_id

    def __init__(self, zone: ZoneRect, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.zone_id = zone.zone_id
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #152028;
                border: 2px solid {zone.color};
                border-radius: 4px;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(6)

        # Zone ID badge
        badge = QLabel(f"Z{zone.zone_id}")
        badge.setFont(QFont("Consolas", 9, QFont.Bold))
        badge.setFixedWidth(26)
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet(
            f"color: #000; background-color: {zone.color}; "
            "border-radius: 3px; padding: 1px;"
        )
        layout.addWidget(badge)

        # Name editor
        self._name_edit = QLineEdit(zone.name)
        self._name_edit.setFont(QFont("Consolas", 9))
        self._name_edit.setStyleSheet(
            "background:#0d1b2a; color:#eceff1; "
            "border:1px solid #37474f; border-radius:3px; padding:2px 4px;"
        )
        self._name_edit.textChanged.connect(
            lambda txt: self.name_changed.emit(self.zone_id, txt)
        )
        layout.addWidget(self._name_edit, 1)

        # Coords label
        coords = QLabel(
            f"({zone.x1},{zone.y1})\n({zone.x2},{zone.y2})"
        )
        coords.setFont(QFont("Consolas", 7))
        coords.setStyleSheet("color: #546e7a;")
        coords.setFixedWidth(90)
        layout.addWidget(coords)

        # Colour swatch button
        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(22, 22)
        self._color_btn.setStyleSheet(
            f"background-color:{zone.color}; border:1px solid #455a64; border-radius:3px;"
        )
        self._color_btn.setToolTip("Change zone colour")
        self._current_color = zone.color
        self._color_btn.clicked.connect(self._pick_color)
        layout.addWidget(self._color_btn)

        # Delete button
        del_btn = QPushButton("✕")
        del_btn.setFont(QFont("Consolas", 9, QFont.Bold))
        del_btn.setFixedSize(22, 22)
        del_btn.setStyleSheet(
            "background-color:#37474f; color:#ef9a9a; "
            "border:1px solid #455a64; border-radius:3px;"
        )
        del_btn.setToolTip("Remove this zone")
        del_btn.clicked.connect(lambda: self.delete_clicked.emit(self.zone_id))
        layout.addWidget(del_btn)

    def _pick_color(self) -> None:
        dlg = QColorDialog(QColor(self._current_color), self)
        if dlg.exec_():
            chosen = dlg.selectedColor().name()
            self._current_color = chosen
            self._color_btn.setStyleSheet(
                f"background-color:{chosen}; border:1px solid #455a64; border-radius:3px;"
            )
            self.color_changed.emit(self.zone_id, chosen)

    def set_highlighted(self, on: bool) -> None:
        """Highlight this row when zone is hovered on canvas."""
        base_color = self._current_color
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {'#1e3040' if on else '#152028'};
                border: {'3' if on else '2'}px solid {base_color};
                border-radius: 4px;
            }}
        """)


# ---------------------------------------------------------------------------
# Main Calibration Screen
# ---------------------------------------------------------------------------

class CalibrationScreen(QWidget):
    """
    Full zone-calibration screen.

    Signals
    -------
    calibration_saved(dict)  — emitted with the full config dict after saving
    back_requested()         — user clicked Back / Cancel
    """

    calibration_saved = pyqtSignal(dict)
    back_requested    = pyqtSignal()

    # Default JSON save path (relative to project root)
    DEFAULT_SAVE_DIR = Path(__file__).parent.parent / "config"

    def __init__(
        self,
        config: ConfigHandler,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config   = config
        self._zones:   list[ZoneRect] = []
        self._camera:  Optional[CameraHandler] = None
        self._live     = False

        # QTimer drives the live preview
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(33)           # ~30 fps
        self._preview_timer.timeout.connect(self._grab_live_frame)

        # Remember last used save path
        self._last_save_path: Optional[str] = None

        self._build_ui()
        self._load_existing_zones_from_config()

    # ================================================================== #
    # UI construction
    # ================================================================== #

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        # ---- Header ----
        root.addLayout(self._build_header())

        # ---- Body ----
        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_canvas_panel(), stretch=3)
        body.addLayout(self._build_right_panel(), stretch=1)
        root.addLayout(body)

    # ---- Header ----

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()

        title = QLabel("🗺  ZONE CALIBRATION")
        title.setFont(QFont("Consolas", 14, QFont.Bold))
        title.setStyleSheet("color: #00bcd4;")
        row.addWidget(title)
        row.addStretch()

        self._lbl_hint = QLabel(
            "Drag on the preview to draw a zone rectangle.  "
            "Hover to inspect.  Right-panel to rename / recolour."
        )
        self._lbl_hint.setFont(QFont("Consolas", 8))
        self._lbl_hint.setStyleSheet("color: #546e7a;")
        row.addWidget(self._lbl_hint)

        btn_back = self._make_btn("← Back", self._on_back)
        row.addWidget(btn_back)
        return row

    # ---- Canvas panel ----

    def _build_canvas_panel(self) -> QWidget:
        frame = QFrame()
        frame.setStyleSheet(
            "background-color:#0d1b2a; border:1px solid #263238; border-radius:5px;"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._canvas = ZoneCanvas()
        self._canvas.zone_drawn.connect(self._on_zone_drawn)
        self._canvas.zone_hovered.connect(self._on_zone_hovered)
        layout.addWidget(self._canvas, stretch=1)

        # Toolbar below canvas
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        for label, slot, accent in [
            ("📸  Capture Frame",  self._capture_frame,  "#00838f"),
            ("▶  Start Preview",  self._toggle_live,    "#1b5e20"),
            ("↩  Undo Last Zone",  self._undo_zone,      "#37474f"),
            ("🗑  Clear All",       self._clear_all,      "#b71c1c"),
        ]:
            btn = self._make_btn(label, slot, accent=accent)
            toolbar.addWidget(btn)
            # Store preview button ref for text toggling
            if "Preview" in label:
                self._btn_preview = btn

        toolbar.addStretch()

        # Live indicator dot
        self._lbl_live = QLabel("● LIVE")
        self._lbl_live.setFont(QFont("Consolas", 8, QFont.Bold))
        self._lbl_live.setStyleSheet("color: #546e7a;")
        toolbar.addWidget(self._lbl_live)

        layout.addLayout(toolbar)

        # Status bar (zone count + hover info)
        self._lbl_status = QLabel("No zones defined — drag to create one")
        self._lbl_status.setFont(QFont("Consolas", 8))
        self._lbl_status.setStyleSheet(
            "color: #607d8b; border-top:1px solid #152028; padding-top:3px;"
        )
        layout.addWidget(self._lbl_status)

        return frame

    # ---- Right panel ----

    def _build_right_panel(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(10)

        # ---- Zone list ----
        zone_group = QGroupBox("Defined Zones")
        zone_group.setStyleSheet(self._group_style())
        zone_layout = QVBoxLayout(zone_group)
        zone_layout.setContentsMargins(4, 8, 4, 4)
        zone_layout.setSpacing(4)

        self._zone_scroll = QScrollArea()
        self._zone_scroll.setWidgetResizable(True)
        self._zone_scroll.setStyleSheet(
            "background:#0a1520; border:none;"
        )
        self._zone_scroll.setFixedHeight(280)

        self._zone_list_widget = QWidget()
        self._zone_list_layout = QVBoxLayout(self._zone_list_widget)
        self._zone_list_layout.setContentsMargins(0, 0, 0, 0)
        self._zone_list_layout.setSpacing(4)
        self._zone_list_layout.addStretch()

        self._zone_scroll.setWidget(self._zone_list_widget)
        zone_layout.addWidget(self._zone_scroll)
        col.addWidget(zone_group)

        # ---- Add zone manually (without drawing) ----
        add_btn = self._make_btn("＋  Add Zone Manually", self._add_zone_manual,
                                  accent="#263238")
        add_btn.setToolTip(
            "Adds a new zone with zero size — then edit coordinates "
            "or draw over the canvas to resize."
        )
        col.addWidget(add_btn)

        col.addStretch()

        # ---- Save / Load ----
        io_group = QGroupBox("Config File")
        io_group.setStyleSheet(self._group_style())
        io_layout = QVBoxLayout(io_group)
        io_layout.setSpacing(6)

        # Station ID field
        sid_row = QHBoxLayout()
        sid_row.addWidget(self._mklabel("Station ID:", 8))
        self._edit_station_id = QLineEdit(self._config.active_station)
        self._edit_station_id.setFont(QFont("Consolas", 9))
        self._edit_station_id.setStyleSheet(
            "background:#0d1b2a; color:#eceff1; "
            "border:1px solid #37474f; border-radius:3px; padding:2px 6px;"
        )
        sid_row.addWidget(self._edit_station_id)
        io_layout.addLayout(sid_row)

        # Frame resolution readout (informational)
        self._lbl_resolution = QLabel("Frame: — × —")
        self._lbl_resolution.setFont(QFont("Consolas", 8))
        self._lbl_resolution.setStyleSheet("color: #546e7a;")
        io_layout.addWidget(self._lbl_resolution)

        # Buttons
        self._btn_load = self._make_btn("📂  Load Config",  self._load_config,
                                         accent="#263238")
        self._btn_save = self._make_btn("💾  Save Config",  self._save_config,
                                         accent="#0d47a1")
        self._btn_save.setFixedHeight(36)
        io_layout.addWidget(self._btn_load)
        io_layout.addWidget(self._btn_save)
        col.addWidget(io_group)

        # Apply to live session button
        self._btn_apply = self._make_btn(
            "✔  Apply & Continue →", self._apply_to_session,
            accent="#1565c0",
        )
        self._btn_apply.setFixedHeight(42)
        self._btn_apply.setEnabled(False)
        col.addWidget(self._btn_apply)

        return col

    # ================================================================== #
    # Zone management
    # ================================================================== #

    def _on_zone_drawn(self, zone: ZoneRect) -> None:
        """Called when user finishes dragging a new zone on canvas."""
        self._zones.append(zone)
        self._add_zone_row(zone)
        self._sync_canvas()
        self._update_status()
        self._btn_apply.setEnabled(True)

    def _add_zone_row(self, zone: ZoneRect) -> None:
        row = ZoneEditorRow(zone)
        row.name_changed.connect(self._on_zone_name_changed)
        row.color_changed.connect(self._on_zone_color_changed)
        row.delete_clicked.connect(self._delete_zone)
        # Insert before the stretch item
        self._zone_list_layout.insertWidget(
            self._zone_list_layout.count() - 1, row
        )

    def _remove_zone_row(self, zone_id: int) -> None:
        for i in range(self._zone_list_layout.count()):
            item = self._zone_list_layout.itemAt(i)
            if item and item.widget():
                w = item.widget()
                if isinstance(w, ZoneEditorRow) and w.zone_id == zone_id:
                    w.deleteLater()
                    return

    def _rebuild_zone_rows(self) -> None:
        """Clear and rebuild all zone rows (used after load)."""
        # Remove all existing rows
        for i in reversed(range(self._zone_list_layout.count())):
            item = self._zone_list_layout.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), ZoneEditorRow):
                item.widget().deleteLater()
        # Re-add
        for zone in self._zones:
            self._add_zone_row(zone)

    def _on_zone_name_changed(self, zone_id: int, name: str) -> None:
        for z in self._zones:
            if z.zone_id == zone_id:
                z.name = name
                break
        self._sync_canvas()

    def _on_zone_color_changed(self, zone_id: int, color: str) -> None:
        for z in self._zones:
            if z.zone_id == zone_id:
                z.color = color
                break
        self._sync_canvas()
        # Also update the row's border colour
        for i in range(self._zone_list_layout.count()):
            item = self._zone_list_layout.itemAt(i)
            if item and item.widget():
                w = item.widget()
                if isinstance(w, ZoneEditorRow) and w.zone_id == zone_id:
                    w.setStyleSheet(f"""
                        QFrame {{
                            background-color: #152028;
                            border: 2px solid {color};
                            border-radius: 4px;
                        }}
                    """)

    def _delete_zone(self, zone_id: int) -> None:
        self._zones = [z for z in self._zones if z.zone_id != zone_id]
        self._remove_zone_row(zone_id)
        self._sync_canvas()
        self._update_status()
        self._btn_apply.setEnabled(bool(self._zones))

    def _undo_zone(self) -> None:
        removed = self._canvas.remove_last_zone()
        if removed:
            self._zones = [z for z in self._zones if z.zone_id != removed.zone_id]
            self._remove_zone_row(removed.zone_id)
            self._update_status()
            self._btn_apply.setEnabled(bool(self._zones))

    def _clear_all(self) -> None:
        if self._zones and QMessageBox.question(
            self, "Clear All",
            "Remove all defined zones?",
            QMessageBox.Yes | QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        self._zones.clear()
        self._canvas.clear_zones()
        self._rebuild_zone_rows()
        self._update_status()
        self._btn_apply.setEnabled(False)

    def _add_zone_manual(self) -> None:
        """Add a new zone with a default near-center rectangle."""
        used_ids = {z.zone_id for z in self._zones}
        new_id   = 1
        while new_id in used_ids:
            new_id += 1
        cx, cy   = self._canvas._frame_w // 2, self._canvas._frame_h // 2
        size     = min(self._canvas._frame_w, self._canvas._frame_h) // 6
        zone = ZoneRect(new_id, f"Zone {new_id}",
                        cx - size, cy - size,
                        cx + size, cy + size)
        self._zones.append(zone)
        self._add_zone_row(zone)
        self._sync_canvas()
        self._update_status()
        self._btn_apply.setEnabled(True)

    def _on_zone_hovered(self, zone_id) -> None:
        # Highlight the corresponding row
        for i in range(self._zone_list_layout.count()):
            item = self._zone_list_layout.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), ZoneEditorRow):
                item.widget().set_highlighted(item.widget().zone_id == zone_id)

        if zone_id is not None:
            for z in self._zones:
                if z.zone_id == zone_id:
                    self._lbl_status.setText(
                        f"Zone {z.zone_id}: \"{z.name}\"  "
                        f"({z.x1},{z.y1})→({z.x2},{z.y2})  "
                        f"{z.width()}×{z.height()} px"
                    )
                    return
        else:
            self._update_status()

    def _sync_canvas(self) -> None:
        self._canvas.set_zones(self._zones)

    def _update_status(self) -> None:
        n = len(self._zones)
        if n == 0:
            self._lbl_status.setText("No zones defined — drag to create one")
        else:
            names = ",  ".join(f"Z{z.zone_id}:{z.name}" for z in self._zones)
            self._lbl_status.setText(f"{n} zone{'s' if n != 1 else ''}:  {names}")

    # ================================================================== #
    # Camera control
    # ================================================================== #

    def _capture_frame(self) -> None:
        """Grab a single frame and freeze it as background."""
        if self._live:
            self._toggle_live()   # stop live first

        cam_cfg = self._config.get_camera_config()
        cam = CameraHandler(
            camera_type  = cam_cfg.get("type",         "webcam"),
            device_index = cam_cfg.get("device_index", 0),
            url          = cam_cfg.get("url",          ""),
            width        = cam_cfg.get("width",        1280),
            height       = cam_cfg.get("height",       720),
        )
        ok = cam.open()
        if ok:
            ret, frame = cam.read()
            cam.release()
            if ret and frame is not None:
                self._canvas.set_background(frame)
                h, w = frame.shape[:2]
                self._lbl_resolution.setText(f"Frame: {w} × {h} px")
                return

        # Fallback: synthetic grey frame
        w, h = cam_cfg.get("width", 1280), cam_cfg.get("height", 720)
        fallback = np.full((h, w, 3), 35, dtype=np.uint8)
        cv2.putText(
            fallback, "Camera not available",
            (w // 2 - 180, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (80, 80, 80), 2,
        )
        self._canvas.set_background(fallback)
        self._lbl_resolution.setText(f"Frame: {w} × {h} px (fallback)")

    def _toggle_live(self) -> None:
        if self._live:
            self._preview_timer.stop()
            if self._camera:
                self._camera.release()
                self._camera = None
            self._live = False
            self._btn_preview.setText("▶  Start Preview")
            self._lbl_live.setStyleSheet("color: #546e7a;")
        else:
            cam_cfg = self._config.get_camera_config()
            self._camera = CameraHandler(
                camera_type  = cam_cfg.get("type",         "webcam"),
                device_index = cam_cfg.get("device_index", 0),
                url          = cam_cfg.get("url",          ""),
                width        = cam_cfg.get("width",        1280),
                height       = cam_cfg.get("height",       720),
                fps          = cam_cfg.get("fps",          30),
            )
            if self._camera.open():
                self._live = True
                self._preview_timer.start()
                self._btn_preview.setText("⏹  Stop Preview")
                self._lbl_live.setStyleSheet(
                    "color: #d50000; font-family:Consolas; font-size:8px; font-weight:bold;"
                )
                # Read frame size
                w = self._camera.actual_width
                h = self._camera.actual_height
                self._lbl_resolution.setText(f"Frame: {w} × {h} px")
            else:
                self._lbl_live.setText("● OFFLINE")

    def _grab_live_frame(self) -> None:
        if self._camera and self._camera.is_open():
            ok, frame = self._camera.read()
            if ok and frame is not None:
                self._canvas.set_background(frame)

    # ================================================================== #
    # Save / Load
    # ================================================================== #

    def _build_config_dict(self) -> dict:
        return {
            "station_id": self._edit_station_id.text().strip()
                          or self._config.active_station,
            "frame_width":  self._canvas._frame_w,
            "frame_height": self._canvas._frame_h,
            "zones": [z.to_dict() for z in self._zones],
        }

    def _save_config(self) -> None:
        if not self._zones:
            QMessageBox.warning(self, "No Zones",
                                "Define at least one zone before saving.")
            return

        default_path = str(
            self.DEFAULT_SAVE_DIR
            / f"{self._edit_station_id.text().strip() or 'station'}_config.json"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Zone Config", default_path,
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return

        data = self._build_config_dict()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self._last_save_path = path
        self._lbl_status.setText(f"✔ Saved → {Path(path).name}")

        # Also push zones into the live ConfigHandler
        self._push_zones_to_config()

        QMessageBox.information(
            self, "Saved",
            f"Zone configuration saved to:\n{path}"
        )

    def _load_config(self) -> None:
        default_dir = str(self.DEFAULT_SAVE_DIR)
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Zone Config", default_dir,
            "JSON Files (*.json);;All Files (*)"
        )
        if not path or not os.path.exists(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate minimal structure
        if "zones" not in data:
            QMessageBox.critical(self, "Invalid File",
                                  "Selected file has no 'zones' key.")
            return

        # Restore station ID
        sid = data.get("station_id", "")
        if sid:
            self._edit_station_id.setText(sid)

        # Rebuild zone list
        self._zones = [ZoneRect.from_dict(zd) for zd in data["zones"]]
        self._canvas.clear_zones()
        self._rebuild_zone_rows()
        self._sync_canvas()
        self._update_status()
        self._btn_apply.setEnabled(bool(self._zones))

        # Restore frame size if saved
        fw = data.get("frame_width",  self._canvas._frame_w)
        fh = data.get("frame_height", self._canvas._frame_h)
        self._canvas._frame_w = fw
        self._canvas._frame_h = fh
        self._lbl_resolution.setText(f"Frame: {fw} × {fh} px")

        self._lbl_status.setText(
            f"✔ Loaded {len(self._zones)} zones from {Path(path).name}"
        )

    def _load_existing_zones_from_config(self) -> None:
        """On startup, populate from ConfigHandler if zones are already saved."""
        zones_cfg = self._config.get_zones()
        loaded = []
        for z in zones_cfg:
            poly = z.get("polygon", [])
            if len(poly) >= 2:
                # Convert bounding box of polygon to rect
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                loaded.append(ZoneRect(
                    zone_id = z["id"],
                    name    = z.get("name", f"Zone {z['id']}"),
                    x1 = min(xs), y1 = min(ys),
                    x2 = max(xs), y2 = max(ys),
                    color   = "#{:02x}{:02x}{:02x}".format(*z.get("color", [0, 120, 255])),
                ))
            elif all(k in z for k in ("x1", "y1", "x2", "y2")):
                loaded.append(ZoneRect.from_dict(z))

        if loaded:
            self._zones = loaded
            self._rebuild_zone_rows()
            self._sync_canvas()
            self._update_status()
            self._btn_apply.setEnabled(True)

    def _push_zones_to_config(self) -> None:
        """Write current zones back into ConfigHandler (in memory)."""
        zones_for_config = []
        for z in self._zones:
            zones_for_config.append({
                "id":      z.zone_id,
                "name":    z.name,
                "color":   [
                    int(z.qcolor().red()),
                    int(z.qcolor().green()),
                    int(z.qcolor().blue()),
                ],
                "polygon": [
                    [z.x1, z.y1],
                    [z.x2, z.y1],
                    [z.x2, z.y2],
                    [z.x1, z.y2],
                ],
                # Rect fields for easy re-loading
                "x1": z.x1, "y1": z.y1,
                "x2": z.x2, "y2": z.y2,
            })
        self._config.set_zones(zones_for_config)

    def _apply_to_session(self) -> None:
        """Save zones to ConfigHandler and emit signal."""
        self._push_zones_to_config()
        self._config.save()
        self.calibration_saved.emit(self._build_config_dict())

    # ================================================================== #
    # Navigation
    # ================================================================== #

    def _on_back(self) -> None:
        if self._live:
            self._toggle_live()
        self.back_requested.emit()

    def closeEvent(self, event) -> None:
        if self._live:
            self._toggle_live()
        super().closeEvent(event)

    # ================================================================== #
    # Style helpers
    # ================================================================== #

    @staticmethod
    def _make_btn(
        label: str,
        slot=None,
        accent: str = "#263238",
    ) -> QPushButton:
        btn = QPushButton(label)
        btn.setFont(QFont("Consolas", 9))
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {accent};
                color: #eceff1;
                border: 1px solid #455a64;
                border-radius: 4px;
                padding: 5px 10px;
            }}
            QPushButton:hover  {{ background-color: #37474f; border-color: #00bcd4; }}
            QPushButton:pressed {{ background-color: #0d1b2a; }}
            QPushButton:disabled {{ color: #455a64; border-color: #263238; }}
        """)
        if slot:
            btn.clicked.connect(slot)
        return btn

    @staticmethod
    def _mklabel(text: str, size: int = 9) -> QLabel:
        lbl = QLabel(text)
        lbl.setFont(QFont("Consolas", size))
        lbl.setStyleSheet("color: #607d8b;")
        return lbl

    @staticmethod
    def _group_style() -> str:
        return """
            QGroupBox {
                color: #546e7a;
                border: 1px solid #263238;
                border-radius: 5px;
                margin-top: 10px;
                font-family: Consolas;
                font-size: 9px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
        """
