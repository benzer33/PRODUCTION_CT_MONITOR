"""
gui/theme.py
Central typography and colour constants for the factory-floor UI.

Import these instead of hardcoding pixel values in individual screens so
a single change here propagates everywhere.

Usage
-----
    from gui.theme import FONT_SIZE_BODY, FONT_SIZE_AI_TEXT, make_font, F

    lbl.setFont(make_font(FONT_SIZE_LABEL))
    lbl.setFont(F(FONT_SIZE_LABEL, bold=True))
"""

from __future__ import annotations
from PyQt5.QtGui import QFont

# ---------------------------------------------------------------------------
# Font family
# ---------------------------------------------------------------------------
FONT_FAMILY = "Consolas"

# ---------------------------------------------------------------------------
# Semantic size scale  (pt)
# ---------------------------------------------------------------------------

# Large heading displayed at the top of each screen
FONT_SIZE_TITLE   = 11

# GroupBox / section-header titles  (was 9 → now 11)
FONT_SIZE_SECTION = 12

# Button labels, form input text, most body copy  (was 9 → now 11)
FONT_SIZE_BODY    = 14

# Form field labels, column headers  (was 8-9 → now 10)
FONT_SIZE_LABEL   = 10

# Hint text, secondary captions  (was 7-8 → now 9)
FONT_SIZE_CAPTION = 7

# Real-time status indicators (e.g. "P3:ARMED") — needs to be read at a glance
FONT_SIZE_STATUS  = 12

# Numeric metric value inside stat cards  (keep prominent)
FONT_SIZE_METRIC  = 12

# AI Analysis long-form text — the most important reading target
FONT_SIZE_AI_TEXT = 22

# Icon-emoji in alert/status widgets
FONT_SIZE_ICON    = 12

# ---------------------------------------------------------------------------
# Colour palette (keep in one place for easy tweaking)
# ---------------------------------------------------------------------------
CLR_ACCENT      = "#00bcd4"
CLR_TEXT        = "#eceff1"
CLR_SUBTEXT     = "#b0bec5"
CLR_MUTED       = "#607d8b"
CLR_SECTION_HDR = "#546e7a"
CLR_BG_CARD     = "#0d1b2a"
CLR_BG_DEEP     = "#0a1520"
CLR_BORDER      = "#263238"
CLR_BORDER_DIM  = "#1e3040"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_font(size: int, bold: bool = False) -> QFont:
    """Return a QFont using the standard family and the given point size."""
    f = QFont(FONT_FAMILY, size)
    if bold:
        f.setBold(True)
    return f


# Short alias used in tight code
F = make_font


# ---------------------------------------------------------------------------
# Reusable stylesheet snippets (return str so callers can concatenate)
# ---------------------------------------------------------------------------

def btn_stylesheet(accent: str = "#263238") -> str:
    """Dark button stylesheet with accent background."""
    return f"""
        QPushButton {{
            background-color: {accent};
            color: {CLR_TEXT};
            border: 1px solid #455a64;
            border-radius: 4px;
            padding: 6px 14px;
            font-family: {FONT_FAMILY};
            font-size: {FONT_SIZE_BODY}px;
        }}
        QPushButton:hover   {{ background-color: #37474f; border-color: {CLR_ACCENT}; }}
        QPushButton:pressed {{ background-color: #0d1b2a; }}
        QPushButton:disabled {{ color: #455a64; border-color: {CLR_BORDER}; }}
    """


def groupbox_stylesheet(
    title_color: str = CLR_SECTION_HDR,
    border_color: str = CLR_BORDER,
) -> str:
    """Standard dark GroupBox stylesheet."""
    return f"""
        QGroupBox {{
            color: {title_color};
            border: 1px solid {border_color};
            border-radius: 5px;
            margin-top: 12px;
            font-family: {FONT_FAMILY};
            font-size: {FONT_SIZE_SECTION}px;
            font-weight: bold;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }}
    """


def input_stylesheet() -> str:
    """Stylesheet for QComboBox / QLineEdit / QSpinBox."""
    return f"""
        QComboBox, QLineEdit, QSpinBox {{
            background-color: #152028;
            color: {CLR_TEXT};
            border: 1px solid #37474f;
            border-radius: 3px;
            padding: 4px 6px;
            font-family: {FONT_FAMILY};
            font-size: {FONT_SIZE_BODY}px;
        }}
        QComboBox:focus, QLineEdit:focus, QSpinBox:focus {{
            border-color: {CLR_ACCENT};
        }}
        QComboBox::drop-down {{ border: none; }}
        QComboBox QAbstractItemView {{
            background: #152028;
            color: {CLR_TEXT};
            selection-background-color: #1565c0;
        }}
    """
