"""
gui/widgets/zone_widget.py
Zone status panel — shows live elapsed time vs standard per zone.

Displays:
- Zone name + ID
- Elapsed time (updates every 100 ms from stats_updated signal)
- Standard time (from golden cycle)
- % deviation — colour-coded:
    green  = ≤ 0%  (on time or faster)
    yellow = 0%–warning threshold
    red    = > critical threshold
- A small progress bar showing how far into standard time the zone is
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QProgressBar,
    QSizePolicy, QVBoxLayout, QWidget,
)


# ---------------------------------------------------------------------------
# Single zone card
# ---------------------------------------------------------------------------

class ZoneCard(QFrame):
    """One zone's status card in the side panel."""

    COLOR_NORMAL   = "#00c853"   # green
    COLOR_WARNING  = "#ffab00"   # amber
    COLOR_CRITICAL = "#d50000"   # red
    COLOR_INACTIVE = "#37474f"   # grey-blue
    COLOR_ACTIVE   = "#1565c0"   # blue border when active

    def __init__(self, zone_id: int, zone_name: str, parent=None) -> None:
        super().__init__(parent)
        self.zone_id   = zone_id
        self.zone_name = zone_name
        self._is_active = False

        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.setLineWidth(2)
        self._apply_style(active=False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        # Header row
        header = QHBoxLayout()
        self._lbl_name   = QLabel(f"Z{zone_id}  {zone_name}")
        self._lbl_name.setFont(QFont("Consolas", 9, QFont.Bold))
        self._lbl_name.setStyleSheet("color: #b0bec5;")
        self._lbl_active = QLabel("●")
        self._lbl_active.setFont(QFont("Consolas", 10))
        self._lbl_active.setStyleSheet("color: #37474f;")
        header.addWidget(self._lbl_name)
        header.addStretch()
        header.addWidget(self._lbl_active)
        layout.addLayout(header)

        # Time row
        time_row = QHBoxLayout()
        self._lbl_elapsed  = QLabel("0.0s")
        self._lbl_elapsed.setFont(QFont("Consolas", 18, QFont.Bold))
        self._lbl_elapsed.setStyleSheet("color: #eceff1;")
        self._lbl_standard = QLabel("/ 0.0s")
        self._lbl_standard.setFont(QFont("Consolas", 10))
        self._lbl_standard.setStyleSheet("color: #607d8b;")
        time_row.addWidget(self._lbl_elapsed)
        time_row.addWidget(self._lbl_standard, alignment=Qt.AlignBottom)
        time_row.addStretch()
        self._lbl_dev = QLabel("")
        self._lbl_dev.setFont(QFont("Consolas", 11, QFont.Bold))
        time_row.addWidget(self._lbl_dev, alignment=Qt.AlignBottom)
        layout.addLayout(time_row)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(6)
        self._progress.setStyleSheet("""
            QProgressBar {
                background-color: #263238;
                border-radius: 3px;
                border: none;
            }
            QProgressBar::chunk {
                background-color: #00c853;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self._progress)

    # ------------------------------------------------------------------
    # Update methods
    # ------------------------------------------------------------------

    def update_stats(
        self,
        elapsed: float,
        standard: float,
        over_pct: float,
        is_active: bool,
        completed: bool,
        warning_pct: int = 15,
        critical_pct: int = 30,
    ) -> None:
        self._is_active = is_active

        self._lbl_elapsed.setText(f"{elapsed:.1f}s")
        self._lbl_standard.setText(f"/ {standard:.1f}s" if standard else "/ --")

        # Deviation label
        if elapsed > 0 and standard > 0:
            sign = "+" if over_pct >= 0 else ""
            self._lbl_dev.setText(f"{sign}{over_pct:.0f}%")
            if over_pct >= critical_pct:
                color = self.COLOR_CRITICAL
            elif over_pct >= warning_pct:
                color = self.COLOR_WARNING
            else:
                color = self.COLOR_NORMAL
            self._lbl_dev.setStyleSheet(f"color: {color};")
        else:
            self._lbl_dev.setText("")

        # Progress bar
        if standard > 0:
            pct = min(int((elapsed / standard) * 100), 100)
            self._progress.setValue(pct)
            bar_color = (
                self.COLOR_CRITICAL if over_pct >= critical_pct
                else self.COLOR_WARNING if over_pct >= warning_pct
                else self.COLOR_NORMAL
            )
            self._progress.setStyleSheet(f"""
                QProgressBar {{
                    background-color: #263238;
                    border-radius: 3px;
                    border: none;
                }}
                QProgressBar::chunk {{
                    background-color: {bar_color};
                    border-radius: 3px;
                }}
            """)

        # Active indicator
        if is_active:
            self._lbl_active.setStyleSheet("color: #00e5ff;")
        elif completed:
            self._lbl_active.setStyleSheet("color: #00c853;")
        else:
            self._lbl_active.setStyleSheet("color: #37474f;")

        self._apply_style(is_active)

    def _apply_style(self, active: bool) -> None:
        border_color = self.COLOR_ACTIVE if active else "#263238"
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #152028;
                border: 2px solid {border_color};
                border-radius: 6px;
            }}
        """)


# ---------------------------------------------------------------------------
# Zone panel (holds N ZoneCards)
# ---------------------------------------------------------------------------

class ZonePanel(QWidget):
    """Side panel with one ZoneCard per zone."""

    def __init__(
        self,
        zones_config: list[dict],
        warning_pct: int  = 15,
        critical_pct: int = 30,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._warning_pct  = warning_pct
        self._critical_pct = critical_pct
        self._cards: dict[int, ZoneCard] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        for zone in zones_config:
            card = ZoneCard(zone["id"], zone.get("name", f"Zone {zone['id']}"))
            self._cards[zone["id"]] = card
            layout.addWidget(card)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Public update API
    # ------------------------------------------------------------------

    def update_stats(self, stats: dict) -> None:
        """
        Accept stats dict from VisionThread.stats_updated signal.
        Keys are zone_id integers; value is dict with elapsed/standard/etc.
        """
        for zone_id, card in self._cards.items():
            zone_stat = stats.get(zone_id)
            if zone_stat is None:
                continue
            card.update_stats(
                elapsed      = zone_stat.get("elapsed", 0.0),
                standard     = zone_stat.get("standard", 0.0),
                over_pct     = zone_stat.get("over_pct", 0.0),
                is_active    = zone_stat.get("is_active", False),
                completed    = zone_stat.get("completed", False),
                warning_pct  = self._warning_pct,
                critical_pct = self._critical_pct,
            )
