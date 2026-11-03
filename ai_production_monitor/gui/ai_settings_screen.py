# -*- coding: utf-8 -*-
"""
gui/ai_settings_screen.py
หน้าตั้งค่า AI -- API, Prompt Template, Output Format, Context
"""
from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QGroupBox, QFormLayout, QScrollArea,
    QFrame, QLineEdit, QTextEdit, QCheckBox, QRadioButton,
    QButtonGroup, QMessageBox,
)

from gui import theme
from gui.theme import (
    CLR_ACCENT, CLR_TEXT, CLR_SUBTEXT, CLR_BORDER,
    make_font, FONT_SIZE_LABEL, FONT_SIZE_TITLE, FONT_SIZE_CAPTION, FONT_SIZE_BODY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lbl(text: str, bold: bool = False) -> QLabel:
    l = QLabel(text)
    l.setFont(make_font(FONT_SIZE_LABEL, bold=bold))
    l.setStyleSheet(f"color: {CLR_TEXT};")
    return l


def _hint(text: str) -> QLabel:
    l = QLabel(text)
    l.setFont(make_font(FONT_SIZE_CAPTION))
    l.setStyleSheet(f"color: {CLR_SUBTEXT};")
    l.setWordWrap(True)
    return l


def _group(title: str) -> QGroupBox:
    g = QGroupBox(title)
    g.setStyleSheet(theme.groupbox_stylesheet())
    inner = QVBoxLayout()
    inner.setContentsMargins(12, 16, 12, 12)
    inner.setSpacing(8)
    g.setLayout(inner)
    return g


def _reset_btn(tooltip: str = "Reset to Default") -> QPushButton:
    b = QPushButton("Reset Default")
    b.setFixedWidth(120)
    b.setToolTip(tooltip)
    b.setStyleSheet(theme.btn_stylesheet("#263238"))
    return b


# ---------------------------------------------------------------------------
# AISettingsScreen
# ---------------------------------------------------------------------------

class AISettingsScreen(QWidget):
    """
    Standalone widget (ฝังใน main stacked-widget) สำหรับจัดการ AI settings
    """

    settings_saved  = pyqtSignal(dict)
    back_requested  = pyqtSignal()

    # Available Gemini models
    _MODELS = [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-3.5-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]

    _LANGUAGES = ["ภาษาไทย", "English", "ภาษาไทย + English"]

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config
        self._build_ui()
        self._load_current()

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 18)
        root.setSpacing(14)

        # Header
        hdr = QHBoxLayout()
        title = QLabel("🤖  ตั้งค่า AI")
        title.setFont(QFont(theme.FONT_FAMILY, FONT_SIZE_TITLE + 2, QFont.Bold))
        title.setStyleSheet(f"color: {CLR_ACCENT};")
        hdr.addWidget(title)
        hdr.addStretch()

        btn_back = QPushButton("← กลับ")
        btn_back.setFixedWidth(110)
        btn_back.setStyleSheet(theme.btn_stylesheet())
        btn_back.clicked.connect(self.back_requested)
        hdr.addWidget(btn_back)
        root.addLayout(hdr)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {CLR_BORDER};")
        root.addWidget(sep)

        # Scroll body
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        body_w = QWidget()
        body_w.setStyleSheet("background: transparent;")
        body = QVBoxLayout(body_w)
        body.setSpacing(14)

        body.addWidget(self._build_api_group())
        body.addWidget(self._build_prompt_group())
        body.addWidget(self._build_format_group())
        body.addWidget(self._build_context_group())
        body.addStretch()

        scroll.setWidget(body_w)
        root.addWidget(scroll, 1)

        # Footer
        foot = QHBoxLayout()
        foot.addStretch()

        self._btn_test = QPushButton("🔌  Test API Key")
        self._btn_test.setFixedWidth(150)
        self._btn_test.setStyleSheet(theme.btn_stylesheet("#263238"))
        self._btn_test.clicked.connect(self._test_api_key)
        foot.addWidget(self._btn_test)

        btn_save = QPushButton("💾  บันทึก & Apply")
        btn_save.setFixedWidth(160)
        btn_save.setStyleSheet(theme.btn_stylesheet(CLR_ACCENT))
        btn_save.clicked.connect(self._save)
        foot.addWidget(btn_save)
        root.addLayout(foot)

    # ------------------------------------------------------------------
    # Group builders
    # ------------------------------------------------------------------

    def _build_api_group(self) -> QGroupBox:
        g = _group("🔑  API & Model")
        form = QFormLayout()
        form.setSpacing(10)

        # Model
        self._combo_model = QComboBox()
        self._combo_model.addItems(self._MODELS)
        self._combo_model.setStyleSheet(theme.input_stylesheet())
        form.addRow(_lbl("Gemini Model:"), self._combo_model)

        # Timeout
        self._spin_timeout = QSpinBox()
        self._spin_timeout.setRange(10, 300)
        self._spin_timeout.setSuffix("  วินาที")
        self._spin_timeout.setFixedWidth(130)
        self._spin_timeout.setStyleSheet(theme.input_stylesheet())
        form.addRow(_lbl("Timeout:"), self._spin_timeout)

        g.layout().addLayout(form)
        g.layout().addWidget(_hint(
            "API Key อ่านจาก GOOGLE_API_KEY ใน .env -- ไม่เก็บใน config เพื่อความปลอดภัย"
        ))

        rst = _reset_btn()
        rst.clicked.connect(self._reset_api)
        g.layout().addWidget(rst, alignment=Qt.AlignRight)
        return g

    def _build_prompt_group(self) -> QGroupBox:
        g = _group("📝  Prompt Template")

        info_row = QHBoxLayout()
        info_row.addWidget(_hint(
            "Variables ที่ใช้ได้: {station_id} {total_cycles} {pass_rate} {passed} {failed} "
            "{seq_errors} {avg_time} {min_time} {max_time} {std_dev} {avg_deviation} "
            "{zone_table} {alert_total} {alert_warn} {alert_crit} {alert_by_zone} "
            "{viol_total} {viol_skip} {viol_oor} {viol_repeat} {viol_cycles} "
            "{cycle_table} {output_language} {output_length_instruction} {output_format_instruction} {operator}"
        ))
        g.layout().addLayout(info_row)

        self._txt_prompt = QTextEdit()
        self._txt_prompt.setMinimumHeight(280)
        self._txt_prompt.setFont(make_font(theme.FONT_SIZE_TABLE))
        self._txt_prompt.setStyleSheet(
            f"background: #0d1b2a; color: {CLR_TEXT}; "
            f"border: 1px solid {CLR_BORDER}; border-radius: 4px;"
        )
        g.layout().addWidget(self._txt_prompt)

        rst = _reset_btn("กลับเป็น prompt เริ่มต้นที่ใช้งานได้")
        rst.clicked.connect(self._reset_prompt)
        g.layout().addWidget(rst, alignment=Qt.AlignRight)
        return g

    def _build_format_group(self) -> QGroupBox:
        g = _group("🎨  Output Format")
        form = QFormLayout()
        form.setSpacing(10)

        # Language
        self._combo_lang = QComboBox()
        self._combo_lang.addItems(self._LANGUAGES)
        self._combo_lang.setStyleSheet(theme.input_stylesheet())
        form.addRow(_lbl("ภาษา Output:"), self._combo_lang)
        g.layout().addLayout(form)

        # Length
        len_row = QHBoxLayout()
        len_row.addWidget(_lbl("ความยาว:"))
        self._bg_length = QButtonGroup(self)
        for i, (val, txt) in enumerate([
            ("short",  "สั้น (~200 คำ)"),
            ("medium", "ปานกลาง (~400 คำ)"),
            ("long",   "ละเอียด (~800 คำ)"),
        ]):
            rb = QRadioButton(txt)
            rb.setProperty("value", val)
            rb.setStyleSheet(f"color: {CLR_TEXT};")
            self._bg_length.addButton(rb, i)
            len_row.addWidget(rb)
        len_row.addStretch()
        g.layout().addLayout(len_row)

        # Format checkboxes
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(_lbl("รูปแบบ:"))
        self._chk_headers = QCheckBox("ใช้หัวข้อ (##)")
        self._chk_headers.setStyleSheet(f"color: {CLR_TEXT};")
        self._chk_bullets = QCheckBox("ใช้ bullet point")
        self._chk_bullets.setStyleSheet(f"color: {CLR_TEXT};")
        fmt_row.addWidget(self._chk_headers)
        fmt_row.addWidget(self._chk_bullets)
        fmt_row.addStretch()
        g.layout().addLayout(fmt_row)

        rst = _reset_btn()
        rst.clicked.connect(self._reset_format)
        g.layout().addWidget(rst, alignment=Qt.AlignRight)
        return g

    def _build_context_group(self) -> QGroupBox:
        g = _group("📊  Context ที่ส่งให้ AI")

        self._chk_ctx_stats   = QCheckBox("สถิติเฉลี่ย (avg / deviation / zone times)")
        self._chk_ctx_alerts  = QCheckBox("Alert list")
        self._chk_ctx_viols   = QCheckBox("Sequence violation")
        self._chk_ctx_cycles  = QCheckBox("รายละเอียดแต่ละรอบ (cycle detail)")
        self._chk_ctx_raw_ts  = QCheckBox("Raw timestamps  ⚠️ เพิ่ม token มาก")

        spin_row = QHBoxLayout()
        self._spin_max_cycles = QSpinBox()
        self._spin_max_cycles.setRange(5, 500)
        self._spin_max_cycles.setSuffix("  รอบล่าสุด")
        self._spin_max_cycles.setFixedWidth(150)
        self._spin_max_cycles.setStyleSheet(theme.input_stylesheet())
        spin_row.addWidget(_lbl("จำกัดข้อมูลรายรอบ:"))
        spin_row.addWidget(self._spin_max_cycles)
        spin_row.addStretch()

        for w in [self._chk_ctx_stats, self._chk_ctx_alerts, self._chk_ctx_viols,
                  self._chk_ctx_cycles, self._chk_ctx_raw_ts]:
            w.setStyleSheet(f"color: {CLR_TEXT};")
            g.layout().addWidget(w)
        g.layout().addLayout(spin_row)

        rst = _reset_btn()
        rst.clicked.connect(self._reset_context)
        g.layout().addWidget(rst, alignment=Qt.AlignRight)
        return g

    # ------------------------------------------------------------------
    # Load / Save / Reset helpers
    # ------------------------------------------------------------------

    def _load_current(self) -> None:
        s = self._config.get_ai_settings()

        # API group
        idx = self._combo_model.findText(s.get("model", ""))
        self._combo_model.setCurrentIndex(max(idx, 0))
        self._spin_timeout.setValue(int(s.get("timeout", 60)))

        # Prompt
        self._txt_prompt.setPlainText(s.get("prompt_template", ""))

        # Format
        lang = s.get("output_language", "ภาษาไทย")
        li = self._combo_lang.findText(lang)
        self._combo_lang.setCurrentIndex(max(li, 0))

        length = s.get("output_length", "medium")
        for btn in self._bg_length.buttons():
            if btn.property("value") == length:
                btn.setChecked(True)
        if not any(b.isChecked() for b in self._bg_length.buttons()):
            self._bg_length.button(1).setChecked(True)  # medium

        self._chk_headers.setChecked(bool(s.get("output_headers", True)))
        self._chk_bullets.setChecked(bool(s.get("output_bullets", True)))

        # Context
        self._chk_ctx_stats.setChecked(bool(s.get("ctx_stats", True)))
        self._chk_ctx_alerts.setChecked(bool(s.get("ctx_alerts", True)))
        self._chk_ctx_viols.setChecked(bool(s.get("ctx_violations", True)))
        self._chk_ctx_cycles.setChecked(bool(s.get("ctx_cycle_detail", True)))
        self._chk_ctx_raw_ts.setChecked(bool(s.get("ctx_raw_timestamps", False)))
        self._spin_max_cycles.setValue(int(s.get("max_cycles_in_prompt", 30)))

    def _collect(self) -> dict:
        length_val = "medium"
        for btn in self._bg_length.buttons():
            if btn.isChecked():
                length_val = btn.property("value")
                break
        return {
            "model":              self._combo_model.currentText(),
            "timeout":            self._spin_timeout.value(),
            "prompt_template":    self._txt_prompt.toPlainText(),
            "output_language":    self._combo_lang.currentText(),
            "output_length":      length_val,
            "output_headers":     self._chk_headers.isChecked(),
            "output_bullets":     self._chk_bullets.isChecked(),
            "ctx_stats":          self._chk_ctx_stats.isChecked(),
            "ctx_alerts":         self._chk_ctx_alerts.isChecked(),
            "ctx_violations":     self._chk_ctx_viols.isChecked(),
            "ctx_cycle_detail":   self._chk_ctx_cycles.isChecked(),
            "ctx_raw_timestamps": self._chk_ctx_raw_ts.isChecked(),
            "max_cycles_in_prompt": self._spin_max_cycles.value(),
        }

    def _save(self) -> None:
        settings = self._collect()
        self._config.set_ai_settings(settings)
        self.settings_saved.emit(settings)
        QMessageBox.information(self, "บันทึกแล้ว", "บันทึกการตั้งค่า AI เรียบร้อย")

    # -- Reset per group --

    def _reset_api(self) -> None:
        d = self._config._AI_DEFAULTS
        idx = self._combo_model.findText(d["model"])
        self._combo_model.setCurrentIndex(max(idx, 0))
        self._spin_timeout.setValue(int(d["timeout"]))

    def _reset_prompt(self) -> None:
        self._txt_prompt.setPlainText(self._config.PROMPT_DEFAULT)

    def _reset_format(self) -> None:
        d = self._config._AI_DEFAULTS
        li = self._combo_lang.findText(d["output_language"])
        self._combo_lang.setCurrentIndex(max(li, 0))
        length = d["output_length"]
        for btn in self._bg_length.buttons():
            if btn.property("value") == length:
                btn.setChecked(True)
        self._chk_headers.setChecked(bool(d["output_headers"]))
        self._chk_bullets.setChecked(bool(d["output_bullets"]))

    def _reset_context(self) -> None:
        d = self._config._AI_DEFAULTS
        self._chk_ctx_stats.setChecked(bool(d["ctx_stats"]))
        self._chk_ctx_alerts.setChecked(bool(d["ctx_alerts"]))
        self._chk_ctx_viols.setChecked(bool(d["ctx_violations"]))
        self._chk_ctx_cycles.setChecked(bool(d["ctx_cycle_detail"]))
        self._chk_ctx_raw_ts.setChecked(bool(d["ctx_raw_timestamps"]))
        self._spin_max_cycles.setValue(int(d["max_cycles_in_prompt"]))

    # ------------------------------------------------------------------
    # Test API Key
    # ------------------------------------------------------------------

    def _test_api_key(self) -> None:
        import os
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            QMessageBox.warning(self, "ไม่พบ API Key",
                "ไม่พบ GOOGLE_API_KEY ใน .env\n"
                "กรุณาเพิ่ม GOOGLE_API_KEY=... ในไฟล์ .env\n"
                "แล้วรีสตาร์ทแอป")
            return

        self._btn_test.setEnabled(False)
        self._btn_test.setText("กำลังทดสอบ...")

        from PyQt5.QtCore import QThread, pyqtSignal as _sig

        class _Tester(QThread):
            done   = _sig(bool, str)
            def __init__(self, key, model):
                super().__init__()
                self._key   = key
                self._model = model
            def run(self):
                try:
                    from google import genai as _g
                    from google.genai import types as _t
                    c = _g.Client(api_key=self._key)
                    r = c.models.generate_content(
                        model    = self._model,
                        contents = "Reply with exactly: OK",
                        config   = _t.GenerateContentConfig(max_output_tokens=10),
                    )
                    self.done.emit(True, r.text.strip())
                except Exception as e:
                    self.done.emit(False, str(e))

        self._tester = _Tester(api_key, self._combo_model.currentText())

        def _on_done(ok: bool, msg: str):
            self._btn_test.setEnabled(True)
            self._btn_test.setText("🔌  Test API Key")
            if ok:
                QMessageBox.information(self, "สำเร็จ",
                    "API Key ใช้งานได้\nResponse: " + msg)
            else:
                QMessageBox.critical(self, "ล้มเหลว",
                    "เชื่อมต่อไม่ได้:\n" + msg)

        self._tester.done.connect(_on_done)
        self._tester.start()
