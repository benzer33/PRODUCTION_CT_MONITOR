"""
gui/summary_screen.py
AI Summary Screen — แสดงหลังกด "Stop & View Summary"

ประกอบด้วย:
  - ส่วนสถิติตัวเลข (pass/fail/sequence error %)
  - กราฟ PyQtGraph 3 ชิ้น:
      1. Cycle time vs standard line
      2. Bar chart deviation (%) ต่อ zone
      3. Alert + sequence violation count ต่อ cycle
  - กล่อง AI Analysis (Thai) — โหลดใน background thread
    ถ้า API ล้มเหลว แสดงข้อความแจ้งเตือน ไม่ทำ crash
"""

from __future__ import annotations

import os
from typing import Optional

import pyqtgraph as pg
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ai.gemini_client import AnalysisWorker
from data.config_handler import ConfigHandler
from data.database import DatabaseManager

# ── PyQtGraph dark theme ─────────────────────────────────────────────────────
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
    lbl_t.setFont(QFont("Consolas", 8))
    lbl_t.setStyleSheet(f"color:{_CLR_SUBTEXT}; border:none; background:transparent;")
    lbl_v = QLabel(value)
    lbl_v.setFont(QFont("Consolas", 16, QFont.Bold))
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
        _CLR_PASS if d <= 5 else (_CLR_WARN if d <= 25 else _CLR_FAIL)
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


def _build_alert_per_cycle_chart(
    plot: pg.PlotWidget,
    cycles: list,
    per_cycle_alerts: dict,
    per_cycle_violations: dict,
) -> None:
    """Grouped bar: alert count + sequence violation count per cycle."""
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
    legend.addItem(bar_a, "⚠ Alert")
    legend.addItem(bar_v, "✗ Seq. Violation")

    plot.getAxis("bottom").setTicks([[(i, f"#{i+1}") for i in xs]])
    plot.setLabel("left", "จำนวน")
    plot.setLabel("bottom", "รอบที่")


# ---------------------------------------------------------------------------
# AI Analysis Worker thread (thin wrapper for summary mode)
# ---------------------------------------------------------------------------

class _SummaryAnalysisWorker(QThread):
    """Thin wrapper: ดึง API key จาก env/.env/config แล้วเรียก AnalysisWorker."""

    analysis_ready = pyqtSignal(str)
    analysis_error = pyqtSignal(str)

    def __init__(self, session_data: dict, config: ConfigHandler, parent=None) -> None:
        super().__init__(parent)
        self._session_data = session_data
        self._config       = config

    def run(self) -> None:
        # ── Resolve API key ────────────────────────────────────────────
        api_key = (
            os.environ.get("GOOGLE_API_KEY", "")
            or self._config.get("google_api_key", "")
        )
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            self.analysis_error.emit(
                "ไม่พบ Google API key\n"
                "กรุณาตั้งค่า GOOGLE_API_KEY ใน .env หรือ config/default_config.json"
            )
            return

        worker = AnalysisWorker(
            api_key      = api_key,
            model        = "gemini-2.0-flash",
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
    AI Summary Screen — แสดงหลัง session_stopped signal

    การใช้งาน
    ---------
        screen = SummaryScreen(config, db)
        stack.addWidget(screen)
        monitor_screen.session_stopped.connect(screen.load_session)
    """

    back_to_monitor = pyqtSignal()  # เมื่อกด "Monitor ต่อ"

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
        """ดึงข้อมูลจาก DB และเรนเดอร์ทุก section"""
        self._session_data = self._db.get_session_full_data(session_id)
        self._render_stats()
        self._render_charts()
        self._start_ai_analysis()

    # ------------------------------------------------------------------
    # UI Build
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.setStyleSheet(f"background:{_CLR_BG}; color:{_CLR_TEXT};")
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        # ── Header ────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        title = QLabel("📊  AI Summary — สรุปผลการผลิต")
        title.setFont(QFont("Consolas", 14, QFont.Bold))
        title.setStyleSheet("color:#00bcd4;")
        hdr.addWidget(title)
        hdr.addStretch()

        self._btn_back = QPushButton("◀  กลับไป Monitor")
        self._btn_back.setFixedHeight(36)
        self._btn_back.setStyleSheet(
            "QPushButton{background:#1565c0;color:#fff;border-radius:4px;font-size:11px;}"
            "QPushButton:hover{background:#1976d2;}"
        )
        self._btn_back.clicked.connect(self.back_to_monitor)
        hdr.addWidget(self._btn_back)
        root.addLayout(hdr)

        # ── Stat cards row ────────────────────────────────────────────
        cards_row = QHBoxLayout()
        cards_row.setSpacing(8)
        self._card_total  = _stat_card("รอบทั้งหมด",     "—")
        self._card_pass   = _stat_card("ผ่านมาตรฐาน",   "—", _CLR_PASS)
        self._card_fail   = _stat_card("ไม่ผ่าน",        "—", _CLR_FAIL)
        self._card_seq    = _stat_card("ผิดลำดับ",       "—", _CLR_WARN)
        self._card_avg    = _stat_card("เวลาเฉลี่ย (s)", "—")
        self._card_dev    = _stat_card("Deviation เฉลี่ย", "—")
        for card in (
            self._card_total, self._card_pass, self._card_fail,
            self._card_seq, self._card_avg, self._card_dev,
        ):
            cards_row.addWidget(card)
        root.addLayout(cards_row)

        # ── Main split: charts | AI text ─────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("QSplitter::handle{background:#152840;}")

        # Charts panel
        charts_panel = QWidget()
        charts_panel.setStyleSheet(f"background:{_CLR_BG};")
        charts_lay = QVBoxLayout(charts_panel)
        charts_lay.setContentsMargins(0, 0, 0, 0)
        charts_lay.setSpacing(6)

        self._plot_cycle  = _make_plot("⏱  Cycle Time vs Standard")
        self._plot_zone   = _make_plot("📊  Deviation ต่อ Zone (%)")
        self._plot_alerts = _make_plot("🔔  Alert / Sequence Violation ต่อรอบ")

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
        ai_title = QLabel("🤖  AI Analysis (Claude)")
        ai_title.setFont(QFont("Consolas", 10, QFont.Bold))
        ai_title.setStyleSheet("color:#00bcd4;")
        ai_header.addWidget(ai_title)
        ai_header.addStretch()

        self._ai_status_lbl = QLabel("กำลังวิเคราะห์...")
        self._ai_status_lbl.setFont(QFont("Consolas", 8))
        self._ai_status_lbl.setStyleSheet(f"color:{_CLR_SUBTEXT};")
        ai_header.addWidget(self._ai_status_lbl)
        ai_lay.addLayout(ai_header)

        self._ai_text = QTextEdit()
        self._ai_text.setReadOnly(True)
        self._ai_text.setStyleSheet(
            f"background:{_CLR_CARD}; color:{_CLR_TEXT}; "
            "border:1px solid #152840; border-radius:4px; "
            "font-family:Consolas; font-size:12px; line-height:1.5;"
        )
        self._ai_text.setPlaceholderText("AI กำลังประมวลผล...")
        ai_lay.addWidget(self._ai_text)

        self._btn_retry = QPushButton("🔄  ลองใหม่ (AI)")
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
        statuses    = [c.status for c in completed]

        # Standard time from golden cycle config
        gc_cfg      = self._config.get_golden_cycle()
        std_times   = gc_cfg.get("standard_times", {})
        std_total   = sum(float(v) for v in std_times.values()) if std_times else 0.0

        # ── Chart 1: cycle times ──────────────────────────────────────
        _build_cycle_time_chart(
            self._plot_cycle, cycle_times, std_total, statuses
        )

        # ── Chart 2: zone deviation ───────────────────────────────────
        zone_avgs = stats.get("zone_avg_times", {})
        zone_stds = {str(k): float(v) for k, v in std_times.items()}
        zone_names_cfg = {
            str(z.get("id", i)): z.get("name", f"Zone {i}")
            for i, z in enumerate(self._config.get_zones())
        }
        _build_zone_deviation_chart(
            self._plot_zone, zone_avgs, zone_stds, zone_names_cfg
        )

        # ── Chart 3: alerts + violations per cycle ────────────────────
        _build_alert_per_cycle_chart(
            self._plot_alerts, cycles, per_ca, per_cv
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

        self._ai_text.setPlaceholderText("AI กำลังประมวลผล...")
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
        self._ai_status_lbl.setText("✅ วิเคราะห์เสร็จสิ้น")
        self._ai_status_lbl.setStyleSheet(f"color:{_CLR_PASS};")
        self._btn_retry.setVisible(False)

    def _on_ai_error(self, error_msg: str) -> None:
        self._ai_text.setPlainText(
            f"⚠  AI Summary ไม่สำเร็จ\n\n{error_msg}\n\n"
            "กราฟและสถิติด้านบนยังคงแสดงข้อมูลครบถ้วน\n"
            "กด 'ลองใหม่' เมื่อมีการเชื่อมต่ออินเทอร์เน็ต หรือตรวจสอบ API key"
        )
        self._ai_status_lbl.setText("❌ AI ไม่สำเร็จ")
        self._ai_status_lbl.setStyleSheet(f"color:{_CLR_FAIL};")
        self._btn_retry.setVisible(True)
