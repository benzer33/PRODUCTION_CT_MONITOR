"""
gui/font_settings_screen.py
หน้าตั้งค่าฟ้อนต์ — เลือก font family และขนาดตัวอักษรในแต่ละส่วน
แล้ว apply ทั่วทั้งแอปทันทีผ่าน theme.apply_font_settings()
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QGroupBox, QFormLayout, QScrollArea,
    QFrame, QLineEdit, QTextEdit,
)

from gui import theme
from gui.theme import CLR_ACCENT, CLR_TEXT, CLR_SUBTEXT, CLR_BG_CARD, CLR_BORDER
from utils.font_downloader import download_google_font, load_fonts_from_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label(text: str, bold: bool = False) -> QLabel:
    lbl = QLabel(text)
    lbl.setFont(QFont(theme.FONT_FAMILY, theme.FONT_SIZE_LABEL, QFont.Bold if bold else QFont.Normal))
    lbl.setStyleSheet(f"color: {CLR_TEXT};")
    return lbl


def _spin(value: int, lo: int = 6, hi: int = 36) -> QSpinBox:
    sb = QSpinBox()
    sb.setRange(lo, hi)
    sb.setValue(value)
    sb.setFixedWidth(72)
    sb.setStyleSheet(theme.input_stylesheet())
    return sb


# ---------------------------------------------------------------------------
# Background download worker
# ---------------------------------------------------------------------------

class _DownloadWorker(QThread):
    progress = pyqtSignal(str)          # log line
    finished = pyqtSignal(list, list)   # (saved_paths, errors)

    def __init__(self, family: str):
        super().__init__()
        self._family = family

    def run(self):
        result = download_google_font(
            self._family,
            progress_cb=lambda msg: self.progress.emit(msg),
        )
        self.finished.emit(
            [str(p) for p in result.saved],
            result.errors,
        )


# ---------------------------------------------------------------------------
# FontSettingsScreen
# ---------------------------------------------------------------------------

class FontSettingsScreen(QWidget):
    """
    Standalone widget (embedded in the main stacked-widget) for managing
    font family and size constants used throughout the app.
    """

    # Emitted after saving so the parent can prompt a restart or refresh
    settings_saved = pyqtSignal(dict)
    # Back button
    back_requested = pyqtSignal()

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
        root.setSpacing(16)

        # --- Header ---
        hdr = QHBoxLayout()
        title = QLabel("🔤  ตั้งค่าฟ้อนต์")
        title.setFont(QFont(theme.FONT_FAMILY, theme.FONT_SIZE_TITLE + 2, QFont.Bold))
        title.setStyleSheet(f"color: {CLR_ACCENT};")
        hdr.addWidget(title)
        hdr.addStretch()

        btn_back = QPushButton("← กลับ")
        btn_back.setFixedWidth(110)
        btn_back.setStyleSheet(theme.btn_stylesheet())
        btn_back.clicked.connect(self.back_requested)
        hdr.addWidget(btn_back)

        root.addLayout(hdr)

        # separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {CLR_BORDER};")
        root.addWidget(sep)

        # --- Scrollable body ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        body_widget = QWidget()
        body_widget.setStyleSheet("background: transparent;")
        body = QVBoxLayout(body_widget)
        body.setSpacing(16)

        # ── Font Family ──────────────────────────────────────────────
        grp_family = self._make_group("🔡  Font Family")
        form_family = QFormLayout()
        form_family.setSpacing(10)

        self._combo_family = self._build_family_combo()
        form_family.addRow(_label("ฟ้อนต์หลัก (ภาษาไทย/อังกฤษ):"), self._combo_family)
        grp_family.layout().addLayout(form_family)
        body.addWidget(grp_family)

        # ── Size Scale ───────────────────────────────────────────────
        grp_sizes = self._make_group("📐  ขนาดตัวอักษร (pt)")
        form_sizes = QFormLayout()
        form_sizes.setSpacing(10)

        self._spin_body    = _spin(theme.FONT_SIZE_BODY)
        self._spin_table   = _spin(theme.FONT_SIZE_TABLE)
        self._spin_ai      = _spin(theme.FONT_SIZE_AI_TEXT, lo=10, hi=40)
        self._spin_title   = _spin(theme.FONT_SIZE_TITLE)
        self._spin_section = _spin(theme.FONT_SIZE_SECTION)
        self._spin_label   = _spin(theme.FONT_SIZE_LABEL)
        self._spin_caption = _spin(theme.FONT_SIZE_CAPTION, lo=6, hi=20)

        form_sizes.addRow(_label("Body / ปุ่ม / ช่องกรอก:"),      self._spin_body)
        form_sizes.addRow(_label("ตาราง (Table rows):"),           self._spin_table)
        form_sizes.addRow(_label("AI Analysis (ข้อความยาว):"),     self._spin_ai)
        form_sizes.addRow(_label("หัวข้อหน้า (Title):"),           self._spin_title)
        form_sizes.addRow(_label("GroupBox / Section header:"),    self._spin_section)
        form_sizes.addRow(_label("Label / คำอธิบายฟอร์ม:"),        self._spin_label)
        form_sizes.addRow(_label("Caption / คำบรรยายเล็ก:"),       self._spin_caption)
        grp_sizes.layout().addLayout(form_sizes)
        body.addWidget(grp_sizes)

        # ── Live Preview ─────────────────────────────────────────────
        grp_preview = self._make_group("👁  Preview")
        preview_layout = QVBoxLayout()
        preview_layout.setSpacing(8)

        self._preview_labels: list[tuple[QLabel, str]] = []

        previews = [
            ("Title",    "หัวข้อหน้าจอ — AI Production Monitor",   "title"),
            ("Section",  "📊 สรุปผลการผลิต — Section Header",       "section"),
            ("Body",     "ข้อมูลทั่วไป: ผ่าน 2 / 4 รอบ (50%)",     "body"),
            ("Table",    "1  |  1.22s (L)  |  0.28s (R)  |  Pass", "table"),
            ("AI Text",  "ผลการวิเคราะห์: ควรปรับ SOP โซน 1",      "ai"),
            ("Caption",  "* ค่ามาตรฐาน 4.84 วินาที",               "caption"),
        ]
        for role, text, key in previews:
            row = QHBoxLayout()
            tag = QLabel(f"{role}:")
            tag.setFixedWidth(72)
            tag.setStyleSheet(f"color: {CLR_SUBTEXT}; font-size: 9pt;")
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color: {CLR_TEXT};")
            lbl.setWordWrap(True)
            row.addWidget(tag)
            row.addWidget(lbl)
            preview_layout.addLayout(row)
            self._preview_labels.append((lbl, key))

        grp_preview.layout().addLayout(preview_layout)
        body.addWidget(grp_preview)

        # ── Download from Google Fonts ───────────────────────────────
        grp_dl = self._make_group("☁️  ดาวน์โหลดฟ้อนต์จาก Google Fonts")
        dl_layout = QVBoxLayout()
        dl_layout.setSpacing(8)

        # Input row
        inp_row = QHBoxLayout()
        inp_row.setSpacing(8)
        inp_lbl = _label("ชื่อฟ้อนต์:")
        inp_lbl.setFixedWidth(90)
        self._dl_input = QLineEdit()
        self._dl_input.setPlaceholderText("เช่น  Sarabun,  Kanit,  Noto Sans Thai")
        self._dl_input.setStyleSheet(theme.input_stylesheet())
        self._dl_input.returnPressed.connect(self._start_download)

        self._btn_download = QPushButton("⬇  Download")
        self._btn_download.setFixedWidth(130)
        self._btn_download.setStyleSheet(theme.btn_stylesheet(CLR_ACCENT))
        self._btn_download.clicked.connect(self._start_download)

        inp_row.addWidget(inp_lbl)
        inp_row.addWidget(self._dl_input, 1)
        inp_row.addWidget(self._btn_download)
        dl_layout.addLayout(inp_row)

        # Hint
        hint = QLabel(
            "💡 ชื่อต้องตรงกับ <a href='https://fonts.google.com/?subset=thai&noto.script=Thai' "
            "style='color:#42a5f5;'>Google Fonts</a> เช่น  Sarabun · Kanit · Prompt · Noto Sans Thai · Mitr"
        )
        hint.setOpenExternalLinks(True)
        hint.setStyleSheet(f"color: {CLR_SUBTEXT}; font-size: 9pt;")
        hint.setWordWrap(True)
        dl_layout.addWidget(hint)

        # Log area
        self._dl_log = QTextEdit()
        self._dl_log.setReadOnly(True)
        self._dl_log.setFixedHeight(120)
        self._dl_log.setStyleSheet(
            f"background: #0a1520; color: #80cbc4; font-family: Consolas; font-size: 9pt; "
            f"border: 1px solid {CLR_BORDER}; border-radius: 4px;"
        )
        dl_layout.addWidget(self._dl_log)

        grp_dl.layout().addLayout(dl_layout)
        body.addWidget(grp_dl)

        body.addStretch()
        scroll.setWidget(body_widget)
        root.addWidget(scroll, 1)

        # --- Footer buttons ---
        foot = QHBoxLayout()
        foot.addStretch()

        btn_preview = QPushButton("👁  Preview")
        btn_preview.setStyleSheet(theme.btn_stylesheet("#263238"))
        btn_preview.setFixedWidth(130)
        btn_preview.clicked.connect(self._refresh_preview)
        foot.addWidget(btn_preview)

        btn_reset = QPushButton("↺  Reset Default")
        btn_reset.setStyleSheet(theme.btn_stylesheet("#263238"))
        btn_reset.setFixedWidth(150)
        btn_reset.clicked.connect(self._reset_defaults)
        foot.addWidget(btn_reset)

        btn_save = QPushButton("💾  บันทึก & Apply")
        btn_save.setStyleSheet(theme.btn_stylesheet(CLR_ACCENT))
        btn_save.setFixedWidth(160)
        btn_save.clicked.connect(self._save)
        foot.addWidget(btn_save)

        root.addLayout(foot)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_group(title: str) -> QGroupBox:
        grp = QGroupBox(title)
        grp.setStyleSheet(theme.groupbox_stylesheet())
        inner = QVBoxLayout()
        inner.setContentsMargins(12, 16, 12, 12)
        inner.setSpacing(8)
        grp.setLayout(inner)
        return grp

    def _build_family_combo(self) -> QComboBox:
        """Build combo with Thai-capable font families available on the system."""
        db = QFontDatabase()

        # Preferred Thai-friendly families to show first
        preferred = [
            "Leelawadee UI",
            "Leelawadee",
            "Tahoma",
            "Segoe UI",
            "Angsana New",
            "CordiaUPC",
            "Browallia New",
            "Sarabun",
            "Kanit",
            "Prompt",
            "Noto Sans Thai",
        ]

        available = db.families()
        ordered = [f for f in preferred if f in available]
        # Also add other families that contain "thai" in the name
        extra = sorted(
            f for f in available
            if "thai" in f.lower() and f not in ordered
        )
        ordered += extra

        combo = QComboBox()
        combo.addItems(ordered)
        combo.setStyleSheet(theme.input_stylesheet())
        combo.setMinimumWidth(200)
        combo.currentTextChanged.connect(self._refresh_preview)
        return combo

    # ------------------------------------------------------------------
    # Load / Save / Reset
    # ------------------------------------------------------------------

    def _load_current(self) -> None:
        saved = self._config.get_font_settings()

        family = saved.get("font_family", theme.FONT_FAMILY)
        idx = self._combo_family.findText(family)
        if idx >= 0:
            self._combo_family.setCurrentIndex(idx)
        else:
            # Insert the unknown family at top so it's selectable
            self._combo_family.insertItem(0, family)
            self._combo_family.setCurrentIndex(0)

        self._spin_body.setValue(saved.get("font_size_body",    theme.FONT_SIZE_BODY))
        self._spin_table.setValue(saved.get("font_size_table",  theme.FONT_SIZE_TABLE))
        self._spin_ai.setValue(saved.get("font_size_ai",        theme.FONT_SIZE_AI_TEXT))
        self._spin_title.setValue(saved.get("font_size_title",  theme.FONT_SIZE_TITLE))
        self._spin_section.setValue(saved.get("font_size_section", theme.FONT_SIZE_SECTION))
        self._spin_label.setValue(saved.get("font_size_label",  theme.FONT_SIZE_LABEL))
        self._spin_caption.setValue(saved.get("font_size_caption", theme.FONT_SIZE_CAPTION))

        self._refresh_preview()

    def _collect(self) -> dict:
        return {
            "font_family":       self._combo_family.currentText(),
            "font_size_body":    self._spin_body.value(),
            "font_size_table":   self._spin_table.value(),
            "font_size_ai":      self._spin_ai.value(),
            "font_size_title":   self._spin_title.value(),
            "font_size_section": self._spin_section.value(),
            "font_size_label":   self._spin_label.value(),
            "font_size_caption": self._spin_caption.value(),
        }

    def _save(self) -> None:
        settings = self._collect()
        theme.apply_font_settings(settings)
        self._config.set_font_settings(settings)
        self._refresh_preview()
        self.settings_saved.emit(settings)

    def _reset_defaults(self) -> None:
        defaults = self._config._FONT_DEFAULTS
        idx = self._combo_family.findText(defaults["font_family"])
        if idx >= 0:
            self._combo_family.setCurrentIndex(idx)
        self._spin_body.setValue(defaults["font_size_body"])
        self._spin_table.setValue(defaults["font_size_table"])
        self._spin_ai.setValue(defaults["font_size_ai"])
        self._spin_title.setValue(defaults["font_size_title"])
        self._spin_section.setValue(defaults["font_size_section"])
        self._spin_label.setValue(defaults["font_size_label"])
        self._spin_caption.setValue(defaults["font_size_caption"])
        self._refresh_preview()

    # ------------------------------------------------------------------
    # Live preview
    # ------------------------------------------------------------------

    def _refresh_preview(self) -> None:
        family = self._combo_family.currentText()
        size_map = {
            "title":   self._spin_title.value(),
            "section": self._spin_section.value(),
            "body":    self._spin_body.value(),
            "table":   self._spin_table.value(),
            "ai":      self._spin_ai.value(),
            "caption": self._spin_caption.value(),
        }
        for lbl, key in self._preview_labels:
            sz = size_map.get(key, self._spin_body.value())
            lbl.setFont(QFont(family, sz))

    # ------------------------------------------------------------------
    # Download from Google Fonts
    # ------------------------------------------------------------------

    def _start_download(self) -> None:
        family = self._dl_input.text().strip()
        if not family:
            self._dl_log.append("⚠️  กรอกชื่อฟ้อนต์ก่อน เช่น  Sarabun")
            return

        self._btn_download.setEnabled(False)
        self._dl_log.clear()
        self._dl_log.append(f"🚀  เริ่มดาวน์โหลด '{family}' ...")

        self._worker = _DownloadWorker(family)
        self._worker.progress.connect(self._on_dl_progress)
        self._worker.finished.connect(self._on_dl_finished)
        self._worker.start()

    @pyqtSlot(str)
    def _on_dl_progress(self, msg: str) -> None:
        self._dl_log.append(msg)
        self._dl_log.verticalScrollBar().setValue(
            self._dl_log.verticalScrollBar().maximum()
        )

    @pyqtSlot(list, list)
    def _on_dl_finished(self, saved: list, errors: list) -> None:
        self._btn_download.setEnabled(True)

        if saved:
            # Re-register new fonts with Qt
            newly_loaded = load_fonts_from_dir()
            self._dl_log.append(
                f"\n✅  เสร็จ! โหลด {len(saved)} ไฟล์ — "
                f"family ใหม่: {newly_loaded if newly_loaded else '(รีสตาร์ทเพื่อดู)'}"
            )
            # Refresh combo so new family appears immediately
            current = self._combo_family.currentText()
            self._combo_family.blockSignals(True)
            self._combo_family.clear()
            new_combo = self._build_family_combo()
            for i in range(new_combo.count()):
                self._combo_family.addItem(new_combo.itemText(i))
            new_combo.deleteLater()
            # Also add any newly loaded families not in preferred list
            for fam in newly_loaded:
                if self._combo_family.findText(fam) < 0:
                    self._combo_family.insertItem(0, fam)
            # Restore selection
            idx = self._combo_family.findText(current)
            self._combo_family.setCurrentIndex(max(idx, 0))
            self._combo_family.blockSignals(False)
        else:
            self._dl_log.append("\n❌  ดาวน์โหลดไม่สำเร็จ")

        for err in errors:
            self._dl_log.append(f"   ⚠️  {err}")
