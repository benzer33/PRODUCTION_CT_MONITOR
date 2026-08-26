# -*- coding: utf-8 -*-
"""
gui/summary_screen.py
AI Summary Screen - แสดงผลสรุปหลังกด "Stop & View Summary"

เนื้อหาที่แสดง:
  - สถิติภาพรวม (pass/fail/sequence error %)
  - กราฟ PyQtGraph 3 จอ:
      1. Cycle time vs standard line
      2. Bar chart deviation (%) ต่อ zone
      3. Alert + sequence violation count ต่อ cycle
  - กล่อง AI Analysis (Thai) ทำงานใน background thread
    โดย API จะถูกเรียกเพียงครั้งเดียว ไม่ crash
"""

from __future__ import annotations

import csv
import datetime
import os
from typing import Optional

import pyqtgraph as pg
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ai.gemini_client import AnalysisWorker
from data.config_handler import ConfigHandler
from data.database import DatabaseManager
from gui.theme import (
    FONT_FAMILY, FONT_SIZE_AI_TEXT, FONT_SIZE_BODY, FONT_SIZE_CAPTION,
    FONT_SIZE_LABEL, FONT_SIZE_METRIC, FONT_SIZE_SECTION, FONT_SIZE_TABLE,
    FONT_SIZE_TITLE, make_font, btn_stylesheet, CLR_MUTED,
)

# PyQtGraph dark theme
pg.setConfigOption("background", "#0d1b2a")
pg.setConfigOption("foreground", "#cfd8dc")

_CLR_PASS     = "#00c853"
_CLR_FAIL     = "#d50000"
_CLR_WARN     = "#ffab00"
_CLR_STD      = "#00bcd4"
_CLR_ACCENT   = "#1565c0"
_CLR_GRID     = "#1e3040"
_CLR_BG       = "#0d1b2a"
_CLR_CARD     = "#0f2233"
_CLR_TEXT     = "#cfd8dc"
_CLR_SUBTEXT  = "#607d8b"
_ZONE_PALETTE = ["#1565c0", "#00838f", "#558b2f", "#6a1b9a", "#e65100"]


# ---------------------------------------------------------------------------
# Helper: stat card
# ---------------------------------------------------------------------------

def _stat_card(title: str, value: str, color: str = _CLR_TEXT) -> QFrame:
    card = QFrame()
    card.setFrameShape(QFrame.StyledPanel)
    card.setStyleSheet(f"background:{_CLR_CARD}; border-radius:6px; border:1px solid #152840;")
    lay = QVBoxLayout(card)
    lay.setContentsMargins(10, 6, 10, 6)
    lay.setSpacing(2)
    lbl_t = QLabel(title)
    lbl_t.setFont(make_font(FONT_SIZE_LABEL))
    lbl_t.setStyleSheet(f"color:{_CLR_SUBTEXT}; border:none; background:transparent;")
    lbl_v = QLabel(value)
    lbl_v.setFont(make_font(FONT_SIZE_METRIC, bold=True))
    lbl_v.setStyleSheet(f"color:{color}; border:none; background:transparent;")
    lay.addWidget(lbl_t)
    lay.addWidget(lbl_v)
    return card


# ---------------------------------------------------------------------------
# Chart helpers (PyQtGraph)
# ---------------------------------------------------------------------------

def _make_plot(title: str) -> pg.PlotWidget:
    pw = pg.PlotWidget(title=title)
    pw.setBackground(_CLR_BG)
    pw.getPlotItem().titleLabel.setAttr("color", _CLR_TEXT)
    pw.showGrid(x=True, y=True, alpha=0.25)
    pw.getAxis("bottom").setTextPen(_CLR_TEXT)
    pw.getAxis("left").setTextPen(_CLR_TEXT)
    pw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return pw


def _build_cycle_time_chart(
    plot: pg.PlotWidget,
    cycle_times: list[float],
    standard_time: float,
    statuses: list[str],
) -> None:
    """Bar chart: cycle time per round, coloured by status, + standard dashed line."""
    plot.clear()
    if not cycle_times:
        return

    n = len(cycle_times)
    xs = list(range(n))

    for i, (t, st) in enumerate(zip(cycle_times, statuses)):
        color = _CLR_PASS if st == "pass" else (_CLR_FAIL if "seq" in st else _CLR_WARN)
        bar = pg.BarGraphItem(x=[i], height=[t], width=0.7, brush=color, pen=pg.mkPen(None))
        plot.addItem(bar)

    if standard_time > 0:
        line = pg.InfiniteLine(
            pos=standard_time, angle=0,
            pen=pg.mkPen(color=_CLR_STD, width=2, style=Qt.DashLine),
            label=f"Standard {standard_time:.1f}s",
            labelOpts={"color": _CLR_STD, "position": 0.95},
        )
        plot.addItem(line)

    plot.getAxis("bottom").setTicks([[(i, f"#{i+1}") for i in xs]])
    plot.setLabel("left", "วินาที")
    plot.setLabel("bottom", "รอบที่")
    # กำหนด x-range ให้ bars แยกกันไม่ทับกัน
    plot.setXRange(-0.5, n - 0.5, padding=0.05)
    if cycle_times:
        plot.setYRange(0, max(cycle_times) * 1.15, padding=0)


def _build_zone_deviation_chart(
    plot: pg.PlotWidget,
    zone_avgs: dict[str, float],
    zone_standards: dict[str, float],
    zone_names: dict[str, str] | None = None,
) -> None:
    """Horizontal bar chart of % deviation per zone."""
    plot.clear()
    if not zone_avgs:
        return

    zones  = sorted(zone_avgs.keys(), key=lambda z: int(z) if z.isdigit() else z)
    ys     = list(range(len(zones)))
    devs   = []
    labels = []
    for zid in zones:
        avg = zone_avgs.get(zid, 0.0)
        std = zone_standards.get(zid, zone_standards.get(str(zid), 0.0))
        dev = ((avg - std) / std * 100) if std > 0 else 0.0
        devs.append(dev)
        name = (zone_names or {}).get(zid, f"Zone {zid}")
        labels.append(name)

    colors = [
        _CLR_PASS if abs(d) <= 5 else (_CLR_WARN if abs(d) <= 25 else _CLR_FAIL)
        for d in devs
    ]

    for i, (dev, color) in enumerate(zip(devs, colors)):
        bar = pg.BarGraphItem(x0=0, x1=dev, y=[i], height=0.6, brush=color, pen=pg.mkPen(None))
        plot.addItem(bar)

    line0 = pg.InfiniteLine(
        pos=0, angle=90,
        pen=pg.mkPen(color=_CLR_STD, width=1, style=Qt.DashLine),
    )
    plot.addItem(line0)

    plot.getAxis("left").setTicks([list(zip(ys, labels))])
    plot.setLabel("bottom", "Deviation (%)")
    plot.setLabel("left", "Zone")
    plot.setYRange(-0.5, len(zones) - 0.5, padding=0.1)
    if devs:
        max_abs = max(abs(d) for d in devs) or 1.0
        plot.setXRange(-max_abs * 1.2, max_abs * 1.2, padding=0)


def _build_alert_per_cycle_chart(
    plot: pg.PlotWidget,
    cycles: list,
    per_cycle_alerts: dict,
    per_cycle_violations: dict,
) -> None:
    """Grouped bar: alert count + sequence violation count per cycle."""
    # ล้าง legend เก่าออกก่อน clear
    legend_item = plot.getPlotItem().legend
    if legend_item is not None:
        legend_item.clear()
    plot.clear()
    completed = [c for c in cycles if c.status != "in_progress"]
    if not completed:
        return

    n = len(completed)
    xs = list(range(n))

    alert_counts = [len(per_cycle_alerts.get(c.id, [])) for c in completed]
    viol_counts  = [len(per_cycle_violations.get(c.id, [])) for c in completed]

    bar_a = pg.BarGraphItem(
        x=[x - 0.2 for x in xs], height=alert_counts, width=0.35,
        brush=_CLR_WARN, pen=pg.mkPen(None), name="Alert",
    )
    bar_v = pg.BarGraphItem(
        x=[x + 0.2 for x in xs], height=viol_counts, width=0.35,
        brush=_CLR_FAIL, pen=pg.mkPen(None), name="Seq. Violation",
    )
    plot.addItem(bar_a)
    plot.addItem(bar_v)

    legend = plot.addLegend(offset=(10, 10))
    legend.addItem(bar_a, "\u26a0 Alert")
    legend.addItem(bar_v, "\u274c Seq. Violation")

    plot.getAxis("bottom").setTicks([[(i, f"#{i+1}") for i in xs]])
    plot.setLabel("left", "จำนวน")
    plot.setLabel("bottom", "รอบที่")
    plot.setXRange(-0.5, n - 0.5, padding=0.05)
    max_y = max(max(alert_counts, default=0), max(viol_counts, default=0))
    plot.setYRange(0, max(max_y * 1.2, 1), padding=0)


# ---------------------------------------------------------------------------
# AI Analysis Worker thread (thin wrapper for summary mode)
# ---------------------------------------------------------------------------

class _SummaryAnalysisWorker(QThread):
    """Thin wrapper: ดึง API key จาก env/.env/config แล้วส่งต่อให้ AnalysisWorker."""

    analysis_ready = pyqtSignal(str)
    analysis_error = pyqtSignal(str)

    def __init__(self, session_data: dict, config: ConfigHandler, parent=None) -> None:
        super().__init__(parent)
        self._session_data = session_data
        self._config       = config

    def run(self) -> None:
        # Resolve API key
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            self.analysis_error.emit(
                "ไม่พบ Google API key\n"
                "กรุณาตั้งค่า GOOGLE_API_KEY ใน .env หรือ config/default_config.json"
            )
            return

        worker = AnalysisWorker(
            api_key      = api_key,
            model        = "gemini-3.5-flash",
            session_data = self._session_data,
        )
        worker.analysis_ready.connect(self.analysis_ready)
        worker.analysis_error.connect(self.analysis_error)
        # Run synchronously inside this thread (AnalysisWorker.run() is just a method)
        worker.run()


# ---------------------------------------------------------------------------
# Summary Screen
# ---------------------------------------------------------------------------

class SummaryScreen(QWidget):
    """
    AI Summary Screen - เปิดขึ้นมาหลังรับ session_stopped signal

    ตัวอย่างการใช้งาน
    ---------
        screen = SummaryScreen(config, db)
        stack.addWidget(screen)
        monitor_screen.session_stopped.connect(screen.load_session)
    """

    back_to_monitor = pyqtSignal()  # ส่งสัญญาณกลับไป "Monitor หน้า"

    def __init__(self, config: ConfigHandler, db: DatabaseManager, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._db     = db
        self._worker: _SummaryAnalysisWorker | None = None
        self._session_data: dict | None = None
        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_session(self, session_id: int) -> None:
        """ดึงข้อมูล session จาก DB แล้ว render ทุก section"""
        import sys
        self._session_data = self._db.get_session_full_data(session_id)
        cycles = self._session_data.get("cycles", [])
        print(f"[SUMMARY] load_session({session_id}) -> {len(cycles)} cycles total", file=sys.stderr, flush=True)
        for c in cycles:
            print(f"  cycle #{c.cycle_number} status={c.status} time={c.cycle_time_sec}", file=sys.stderr, flush=True)
        self._render_stats()
        self._render_charts()
        self._render_table()
        self._start_ai_analysis()

    # ------------------------------------------------------------------
    # UI Build
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.setStyleSheet(f"background:{_CLR_BG}; color:{_CLR_TEXT};")
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        # Header
        hdr = QHBoxLayout()
        title = QLabel("\U0001f4ca  AI Summary - สรุปผลการผลิต")
        title.setFont(make_font(FONT_SIZE_TITLE, bold=True))
        title.setStyleSheet("color:#00bcd4;")
        hdr.addWidget(title)
        hdr.addStretch()

        self._btn_back = QPushButton("\u2190  กลับ Monitor")
        self._btn_back.setFixedHeight(36)
        self._btn_back.setStyleSheet(
            f"QPushButton{{background:#1565c0;color:#fff;border-radius:4px;font-size:{FONT_SIZE_BODY}px;}}"
            "QPushButton:hover{background:#1976d2;}"
        )
        self._btn_back.clicked.connect(self.back_to_monitor)
        hdr.addWidget(self._btn_back)
        root.addLayout(hdr)

        # Stat cards row
        cards_row = QHBoxLayout()
        cards_row.setSpacing(8)
        self._card_total  = _stat_card("รอบทั้งหมด",     "-")
        self._card_pass   = _stat_card("ผ่านมาตรฐาน",   "-", _CLR_PASS)
        self._card_fail   = _stat_card("ไม่ผ่าน",        "-", _CLR_FAIL)
        self._card_seq    = _stat_card("ผิดลำดับ",       "-", _CLR_WARN)
        self._card_avg    = _stat_card("เวลาเฉลี่ย (s)", "-")
        self._card_dev    = _stat_card("Deviation เฉลี่ย", "-")
        for card in (
            self._card_total, self._card_pass, self._card_fail,
            self._card_seq, self._card_avg, self._card_dev,
        ):
            cards_row.addWidget(card)

        root.addLayout(cards_row)

        # Charts + AI analysis splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet("QSplitter::handle{background:#1e3040;}")

        charts_panel = QWidget()
        charts_lay = QVBoxLayout(charts_panel)
        charts_lay.setContentsMargins(0, 0, 0, 0)
        charts_lay.setSpacing(6)

        self._plot_cycle  = _make_plot("\U0001f4c8  Cycle Time vs Standard")
        self._plot_zone   = _make_plot("\U0001f4ca  Deviation by Zone (%)")
        self._plot_alerts = _make_plot("\U0001f514  Alert / Sequence Violation by Cycle")

        charts_lay.addWidget(self._plot_cycle,  3)
        charts_lay.addWidget(self._plot_zone,   2)
        charts_lay.addWidget(self._plot_alerts, 2)

        splitter.addWidget(charts_panel)

        # AI analysis panel
        ai_panel = QWidget()
        ai_panel.setMinimumWidth(320)
        ai_lay = QVBoxLayout(ai_panel)
        ai_lay.setContentsMargins(8, 0, 0, 0)
        ai_lay.setSpacing(6)

        ai_header = QHBoxLayout()
        ai_title = QLabel("\U0001f916  AI Analysis")
        ai_title.setFont(make_font(FONT_SIZE_SECTION, bold=True))
        ai_title.setStyleSheet("color:#00bcd4;")
        ai_header.addWidget(ai_title)
        ai_header.addStretch()

        self._ai_status_lbl = QLabel("กำลังวิเคราะห์...")
        self._ai_status_lbl.setFont(make_font(FONT_SIZE_CAPTION))
        self._ai_status_lbl.setStyleSheet(f"color:{_CLR_SUBTEXT};")
        ai_header.addWidget(self._ai_status_lbl)
        ai_lay.addLayout(ai_header)

        self._ai_text = QTextEdit()
        self._ai_text.setReadOnly(True)
        self._ai_text.setWordWrapMode(1)   # WrapAtWordBoundaryOrAnywhere
        self._ai_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._ai_text.setStyleSheet(
            f"background:{_CLR_CARD}; color:{_CLR_TEXT}; "
            "border:1px solid #152840; border-radius:4px; "
            f"font-family:{FONT_FAMILY}; font-size:{FONT_SIZE_AI_TEXT}px; line-height:1.6;"
        )
        self._ai_text.setPlaceholderText("AI กำลังวิเคราะห์...")
        ai_lay.addWidget(self._ai_text)

        self._btn_retry = QPushButton("\U0001f504  วิเคราะห์ใหม่ (AI)")
        self._btn_retry.setFixedHeight(32)
        self._btn_retry.setVisible(False)
        self._btn_retry.setStyleSheet(
            "QPushButton{background:#263238;color:#90a4ae;border-radius:4px;}"
            "QPushButton:hover{background:#37474f;color:#fff;}"
        )
        self._btn_retry.clicked.connect(self._start_ai_analysis)
        ai_lay.addWidget(self._btn_retry)

        splitter.addWidget(ai_panel)
        splitter.setSizes([680, 360])

        root.addWidget(splitter, 1)

        # Raw data table section
        tbl_hdr_row = QHBoxLayout()
        tbl_title = QLabel("\U0001f4cb  ข้อมูลดิบรายรอบ")
        tbl_title.setFont(make_font(FONT_SIZE_SECTION, bold=True))
        tbl_title.setStyleSheet("color:#00bcd4;")
        tbl_hdr_row.addWidget(tbl_title)
        tbl_hdr_row.addStretch()

        self._btn_export = QPushButton("\U0001f4be  Export CSV")
        self._btn_export.setFixedHeight(32)
        self._btn_export.setStyleSheet(
            f"QPushButton{{background:#1b5e20;color:#a5d6a7;border-radius:4px;"
            f"font-size:{FONT_SIZE_BODY}px;padding:0 12px;}}"
            "QPushButton:hover{background:#2e7d32;color:#fff;}"
        )
        self._btn_export.clicked.connect(self._export_csv)
        tbl_hdr_row.addWidget(self._btn_export)
        root.addLayout(tbl_hdr_row)

        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setAlternatingRowColors(False)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.setStyleSheet(
            f"QTableWidget{{background:{_CLR_CARD};color:{_CLR_TEXT};"
            "border:1px solid #152840;border-radius:4px;gridline-color:#152840;"
            f"font-family:{FONT_FAMILY};font-size:{FONT_SIZE_TABLE}pt;}}"
            f"QHeaderView::section{{background:#0a1520;color:{_CLR_TEXT};"
            "border:none;border-bottom:1px solid #152840;"
            f"padding:4px;font-family:{FONT_FAMILY};font-size:{FONT_SIZE_TABLE}pt;font-weight:bold;}}"
            "QTableWidget::item{padding:4px 8px;}"
        )
        self._table.setMinimumHeight(200)
        self._table.setMaximumHeight(320)
        root.addWidget(self._table)

    # ------------------------------------------------------------------
    # Render helpers
    # ------------------------------------------------------------------

    def _render_stats(self) -> None:
        if not self._session_data:
            return
        stats = self._session_data.get("stats", {})

        total   = stats.get("total_cycles", 0)
        passed  = stats.get("pass_cycles", 0)
        failed  = stats.get("fail_cycles", 0)
        seq_err = stats.get("seq_error_cycles", 0)
        avg_t   = stats.get("avg_cycle_time", 0.0)
        avg_dev = stats.get("avg_deviation_pct", 0.0)

        pct_pass = f"{passed/max(total,1)*100:.0f}%"
        self._card_total.findChildren(QLabel)[1].setText(str(total))
        self._card_pass.findChildren(QLabel)[1].setText(f"{passed}  ({pct_pass})")
        self._card_fail.findChildren(QLabel)[1].setText(str(failed))
        self._card_seq.findChildren(QLabel)[1].setText(str(seq_err))
        self._card_avg.findChildren(QLabel)[1].setText(f"{avg_t:.2f}")
        dev_sign = "+" if avg_dev >= 0 else ""
        self._card_dev.findChildren(QLabel)[1].setText(f"{dev_sign}{avg_dev:.1f}%")

    def _render_charts(self) -> None:
        if not self._session_data:
            return

        stats   = self._session_data.get("stats", {})
        cycles  = self._session_data.get("cycles", [])
        per_ca  = self._session_data.get("per_cycle_alerts", {})
        per_cv  = self._session_data.get("per_cycle_violations", {})

        completed   = [c for c in cycles if c.status != "in_progress"]
        cycle_times = [c.cycle_time_sec or 0.0 for c in completed]
        statuses    = [self._cycle_status_csv(c) for c in completed]

        golden = self._config.get_golden_cycle() or {}
        std_times_raw: dict = golden.get("standard_times") or {}

        # ใช้ standard_time_sec จาก CycleLog ตัวแรก — ค่าเดียวกับที่ใช้ตัดสิน pass/fail จริง
        # (รวม travel time แล้ว ต่างจาก sum zone times)
        std_time = next(
            (c.standard_time_sec for c in completed if c.standard_time_sec),
            0.0,
        ) or 0.0

        _build_cycle_time_chart(self._plot_cycle, cycle_times, std_time, statuses)

        # Chart 2: zone deviation
        std_times: dict = std_times_raw
        zone_avgs: dict[str, float] = {}
        zone_counts: dict[str, int] = {}
        for c in completed:
            for zid, t in (c.zone_times or {}).items():
                key = str(zid)
                zone_avgs[key]   = zone_avgs.get(key, 0.0) + (t or 0.0)
                zone_counts[key] = zone_counts.get(key, 0) + 1
        for k in zone_avgs:
            if zone_counts[k] > 0:
                zone_avgs[k] /= zone_counts[k]

        zone_stds = {str(k): float(v) for k, v in std_times.items()}
        zone_names_cfg = {
            str(z.get("id", i)): z.get("name", f"Zone {i}")
            for i, z in enumerate(self._config.get_zones())
        }
        _build_zone_deviation_chart(
            self._plot_zone, zone_avgs, zone_stds, zone_names_cfg
        )

        # Chart 3: alerts + violations per cycle
        _build_alert_per_cycle_chart(
            self._plot_alerts, cycles, per_ca, per_cv
        )

    @staticmethod
    def _cycle_status_ui(cycle) -> str:
        """แปลง status + sequence_errors ของ cycle เป็นข้อความสำหรับ UI แบบกระชับ"""
        has_seq = bool(cycle.sequence_errors)
        is_slow = cycle.status in ("fail", "timeout")
        if has_seq and is_slow:
            return "ช้า+ผิดลำดับ"
        if has_seq:
            return "ผิดลำดับขั้นตอน"
        if is_slow:
            return "ช้าเกินมาตรฐาน"
        return "Pass"

    @staticmethod
    def _cycle_status_csv(cycle) -> str:
        """แปลง status + sequence_errors เป็น English key สำหรับ CSV"""
        has_seq = bool(cycle.sequence_errors)
        is_slow = cycle.status in ("fail", "timeout")
        if has_seq and is_slow:
            return "fail_slow_and_sequence_error"
        if has_seq:
            return "sequence_error"
        if is_slow:
            return "fail_slow"
        return "pass"

    def _build_zone_names(self) -> dict[str, str]:
        """สร้าง dict {str(zone_id): zone_name} จาก config"""
        return {
            str(z.get("id", i)): z.get("name", f"Zone {i}")
            for i, z in enumerate(self._config.get_zones())
        }

    def _render_table(self) -> None:
        """เติมข้อมูล QTableWidget จาก session_data - ข้อมูลดิบรายรอบ + แถวสรุป"""
        tbl = self._table
        tbl.clearContents()
        tbl.setRowCount(0)
        tbl.setColumnCount(0)

        if not self._session_data:
            return

        cycles   = self._session_data.get("cycles", [])
        per_ca   = self._session_data.get("per_cycle_alerts", {})
        per_cv   = self._session_data.get("per_cycle_violations", {})
        completed = [c for c in cycles if c.status != "in_progress"]
        if not completed:
            return

        zone_names = self._build_zone_names()  # {str(id): name}
        # ลำดับ zone จาก config
        zone_ids_ordered: list[str] = [
            str(z.get("id", i))
            for i, z in enumerate(self._config.get_zones())
        ]
        if not zone_ids_ordered:
            # fallback: รวม zone_ids จาก cycle data
            seen: set[str] = set()
            for c in completed:
                for k in (c.zone_times or {}).keys():
                    seen.add(str(k))
            zone_ids_ordered = sorted(seen)

        # สร้าง headers
        fixed_headers = ["รอบที่"]
        zone_col_names = [zone_names.get(zid, f"Zone {zid}") for zid in zone_ids_ordered]
        tail_headers   = ["เวลารวม (s)", "มาตรฐาน (s)", "เบี่ยงเบน (%)", "Alert", "สถานะ"]
        all_headers    = fixed_headers + zone_col_names + tail_headers

        n_data_rows = len(completed)
        tbl.setColumnCount(len(all_headers))
        tbl.setRowCount(n_data_rows + 1)   # +1 = แถวสรุป
        tbl.setHorizontalHeaderLabels(all_headers)

        # Cell background colors
        _BG_PASS    = QColor(10, 30, 10)
        _BG_SLOW    = QColor(50, 20, 0, 180)
        _BG_SEQ     = QColor(40, 35, 0, 180)
        _BG_BOTH    = QColor(55, 10, 10, 190)
        _BG_SUMMARY = QColor(15, 25, 40)

        total_times: list[float] = []

        for row_i, cyc in enumerate(completed):
            has_seq = bool(cyc.sequence_errors)
            is_slow = cyc.status in ("fail", "timeout")
            if has_seq and is_slow:
                row_bg = _BG_BOTH
            elif has_seq:
                row_bg = _BG_SEQ
            elif is_slow:
                row_bg = _BG_SLOW
            else:
                row_bg = _BG_PASS

            col = 0

            def _cell(text: str, align=Qt.AlignCenter, _bg=row_bg) -> QTableWidgetItem:
                item = QTableWidgetItem(text)
                item.setBackground(_bg)
                item.setTextAlignment(align | Qt.AlignVCenter)
                return item

            # col 0: รอบที่
            tbl.setItem(row_i, col, _cell(str(cyc.cycle_number))); col += 1

            # zone columns
            zone_times: dict = cyc.zone_times or {}
            zone_hands: dict = cyc.zone_hands or {}
            for zid in zone_ids_ordered:
                t = zone_times.get(zid) or zone_times.get(int(zid) if zid.isdigit() else zid)
                h = zone_hands.get(zid) or zone_hands.get(int(zid) if zid.isdigit() else zid)
                if t is not None:
                    hand_tag = f" ({h[0]})" if h else ""
                    cell_txt = f"{t:.2f}s{hand_tag}"
                else:
                    cell_txt = "-"
                tbl.setItem(row_i, col, _cell(cell_txt)); col += 1

            # เวลารวม
            total_t = cyc.cycle_time_sec
            if total_t is not None:
                total_times.append(total_t)
                tbl.setItem(row_i, col, _cell(f"{total_t:.2f}")); col += 1
            else:
                tbl.setItem(row_i, col, _cell("-")); col += 1

            # มาตรฐาน
            std_t = cyc.standard_time_sec
            tbl.setItem(row_i, col, _cell(f"{std_t:.2f}" if std_t else "-")); col += 1

            # เบี่ยงเบน (%)
            dev = cyc.deviation_pct
            if dev is not None:
                sign = "+" if dev >= 0 else ""
                tbl.setItem(row_i, col, _cell(f"{sign}{dev:.1f}%")); col += 1
            else:
                tbl.setItem(row_i, col, _cell("-")); col += 1

            # Alert count
            alert_cnt = len(per_ca.get(cyc.id, []))
            tbl.setItem(row_i, col, _cell(str(alert_cnt) if alert_cnt else "0")); col += 1

            # สถานะ
            status_txt = self._cycle_status_ui(cyc)
            status_item = _cell(status_txt)
            if is_slow or has_seq:
                status_item.setForeground(
                    QColor(_CLR_FAIL) if (is_slow and has_seq) else
                    (QColor(_CLR_WARN) if has_seq else QColor(_CLR_FAIL))
                )
            else:
                status_item.setForeground(QColor(_CLR_PASS))
            tbl.setItem(row_i, col, status_item)

        # แถวสรุป (footer)
        summary_row = n_data_rows
        n_cols = len(all_headers)

        def _sum_cell(text: str, align=Qt.AlignCenter) -> QTableWidgetItem:
            item = QTableWidgetItem(text)
            item.setBackground(_BG_SUMMARY)
            item.setTextAlignment(align | Qt.AlignVCenter)
            item.setForeground(QColor("#b0bec5"))
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            return item

        # col 0: label
        tbl.setItem(summary_row, 0, _sum_cell("สรุป"))

        # zone columns: ว่าง
        for ci in range(1, 1 + len(zone_ids_ordered)):
            tbl.setItem(summary_row, ci, _sum_cell(""))

        base_col = 1 + len(zone_ids_ordered)

        if total_times:
            avg_t = sum(total_times) / len(total_times)
            min_t = min(total_times)
            max_t = max(total_times)
            tbl.setItem(summary_row, base_col,
                        _sum_cell(f"avg {avg_t:.2f}  min {min_t:.2f}  max {max_t:.2f}"))
        else:
            tbl.setItem(summary_row, base_col, _sum_cell("-"))

        tbl.setItem(summary_row, base_col + 1, _sum_cell(""))  # มาตรฐาน
        tbl.setItem(summary_row, base_col + 2, _sum_cell(""))  # เบี่ยงเบน

        # pass rate
        pass_cnt = sum(1 for c in completed if c.status == "pass"
                       and not c.sequence_errors)
        pass_pct = pass_cnt / len(completed) * 100
        tbl.setItem(summary_row, base_col + 3, _sum_cell(""))   # alert col
        tbl.setItem(summary_row, base_col + 4,
                    _sum_cell(f"Pass rate: {pass_pct:.0f}%  ({pass_cnt}/{len(completed)})"))

        tbl.resizeRowsToContents()

    # ------------------------------------------------------------------
    # CSV Export
    # ------------------------------------------------------------------

    def _export_csv(self) -> None:
        """Export ข้อมูลดิบรายรอบเป็น CSV (UTF-8 with BOM for Excel compatibility)."""
        if not self._session_data:
            QMessageBox.warning(self, "Export CSV", "ยังไม่มีข้อมูล session ที่จะ export")
            return

        cycles    = self._session_data.get("cycles", [])
        per_ca    = self._session_data.get("per_cycle_alerts", {})
        session   = self._session_data.get("session")
        completed = [c for c in cycles if c.status != "in_progress"]
        if not completed:
            QMessageBox.warning(self, "Export CSV", "ไม่มี cycle ที่สมบูรณ์ใน session นี้")
            return

        zone_names = self._build_zone_names()
        zone_ids_ordered: list[str] = [
            str(z.get("id", i))
            for i, z in enumerate(self._config.get_zones())
        ]
        if not zone_ids_ordered:
            seen: set[str] = set()
            for c in completed:
                for k in (c.zone_times or {}).keys():
                    seen.add(str(k))
            zone_ids_ordered = sorted(seen)

        # สร้าง default filename
        station_id  = self._config.active_station or "station"
        session_id  = session.id if session else "?"
        date_str    = datetime.date.today().strftime("%Y%m%d")
        default_fn  = f"{station_id}_session{session_id}_{date_str}.csv"

        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", default_fn,
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return  # user cancelled

        # Build header row
        headers = ["cycle_number"]
        for zid in zone_ids_ordered:
            zname = zone_names.get(zid, f"zone{zid}")
            headers += [f"{zname}_sec", f"{zname}_hand"]
        headers += ["total_sec", "standard_sec", "deviation_pct", "alert_count", "status"]

        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

                for cyc in completed:
                    row: list = [cyc.cycle_number]

                    zone_times: dict = cyc.zone_times or {}
                    zone_hands: dict = cyc.zone_hands or {}
                    for zid in zone_ids_ordered:
                        t = zone_times.get(zid) or zone_times.get(
                            int(zid) if zid.isdigit() else zid)
                        h = zone_hands.get(zid) or zone_hands.get(
                            int(zid) if zid.isdigit() else zid)
                        row.append(f"{t:.4f}" if t is not None else "")
                        row.append(h[0] if h else "")

                    row.append(f"{cyc.cycle_time_sec:.4f}" if cyc.cycle_time_sec is not None else "")
                    row.append(f"{cyc.standard_time_sec:.4f}" if cyc.standard_time_sec is not None else "")
                    row.append(f"{cyc.deviation_pct:.2f}" if cyc.deviation_pct is not None else "")
                    row.append(len(per_ca.get(cyc.id, [])))
                    row.append(self._cycle_status_csv(cyc))
                    writer.writerow(row)

            QMessageBox.information(
                self, "Export สำเร็จ",
                f"บันทึกไฟล์เรียบร้อยแล้ว:\n{path}"
            )
        except OSError as exc:
            QMessageBox.critical(
                self, "Export ล้มเหลว",
                f"ไม่สามารถเขียนไฟล์ได้:\n{exc.strerror}\n\n{path}"
            )

    # ------------------------------------------------------------------
    # AI Analysis
    # ------------------------------------------------------------------

    def _start_ai_analysis(self) -> None:
        if not self._session_data:
            return

        # Cancel any previous worker
        if self._worker and self._worker.isRunning():
            self._worker.terminate()

        self._ai_text.setPlaceholderText("AI กำลังวิเคราะห์...")
        self._ai_text.clear()
        self._ai_status_lbl.setText("กำลังวิเคราะห์...")
        self._ai_status_lbl.setStyleSheet(f"color:{_CLR_SUBTEXT};")
        self._btn_retry.setVisible(False)

        self._worker = _SummaryAnalysisWorker(self._session_data, self._config, parent=self)
        self._worker.analysis_ready.connect(self._on_ai_ready)
        self._worker.analysis_error.connect(self._on_ai_error)
        self._worker.start()

    def _on_ai_ready(self, text: str) -> None:
        self._ai_text.setMarkdown(text)
        self._ai_status_lbl.setText("\u2705 วิเคราะห์เสร็จสิ้น")
        self._ai_status_lbl.setStyleSheet(f"color:{_CLR_PASS};")
        self._btn_retry.setVisible(False)

    def _on_ai_error(self, error_msg: str) -> None:
        self._ai_text.setPlainText(
            f"\u26a0  AI Summary ล้มเหลว\n\n{error_msg}\n\n"
            "กรุณาตรวจสอบ API key และการเชื่อมต่ออินเทอร์เน็ต\n"
            "กด 'วิเคราะห์ใหม่' เพื่อลองอีกครั้ง หรือตรวจสอบ API key"
        )
        self._ai_status_lbl.setText("\u274c AI ล้มเหลว")
        self._ai_status_lbl.setStyleSheet(f"color:{_CLR_FAIL};")
        self._btn_retry.setVisible(True)
