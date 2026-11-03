# -*- coding: utf-8 -*-
"""
utils/font_downloader.py
ดาวน์โหลดฟ้อนต์จาก Google Fonts CSS API แล้วบันทึกลง fonts/ folder
"""

from __future__ import annotations

import re
import urllib.request
import urllib.parse
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _app_dir() -> Path:
    import sys
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


def fonts_dir() -> Path:
    d = _app_dir() / "fonts"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Google Fonts CSS API  (ใช้ UA เก่าเพื่อรับ TTF)
# ---------------------------------------------------------------------------

# Chrome 35 (ก่อน WOFF2 support) --> Google Fonts ส่ง TTF กลับมา
_UA_TTF = (
    "Mozilla/5.0 (Windows NT 6.1; WOW64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/35.0.1916.47 Safari/537.36"
)

_CSS_URL = "https://fonts.googleapis.com/css?family={family}"

# จับ url(...) ในไฟล์ CSS
_URL_RE = re.compile(
    r"url\((https://fonts\.gstatic\.com/[^)]+\.(ttf|otf))\)"
)


def _fetch_text(url: str, ua: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def _fetch_bytes(url: str, ua: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _get_font_urls(family: str) -> list:
    """ดึง list ของ direct TTF URL จาก Google Fonts CSS API"""
    # สร้าง URL: spaces -> + และ variant ครบ
    family_param = urllib.parse.quote(family, safe="+").replace("%20", "+")
    css_url = _CSS_URL.format(family=family_param)
    css = _fetch_text(css_url, ua=_UA_TTF)
    urls = [m.group(1) for m in _URL_RE.finditer(css)]
    return urls


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DownloadResult:
    def __init__(self, family: str, saved: list, errors: list):
        self.family = family
        self.saved  = saved
        self.errors = errors


def download_google_font(family: str, progress_cb=None) -> DownloadResult:
    """
    ดาวน์โหลด Google Font family และบันทึก TTF ลง fonts/<family>/

    Parameters
    ----------
    family      : ชื่อ Google Fonts เช่น Sarabun, Kanit, Noto Sans Thai
    progress_cb : callback(str) สำหรับอัปเดต UI
    """

    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    saved:  list = []
    errors: list = []

    log("กำลังดึง font list ของ '" + family + "' จาก Google Fonts CSS API...")

    try:
        urls = _get_font_urls(family)
    except Exception as exc:
        errors.append("เชื่อมต่อ Google Fonts ไม่ได้: " + str(exc))
        log("ERROR: " + errors[-1])
        return DownloadResult(family, saved, errors)

    if not urls:
        errors.append(
            "ไม่พบไฟล์ .ttf สำหรับ '" + family + "' "
            "-- ตรวจสอบชื่อให้ตรงกับ Google Fonts (case-sensitive)"
        )
        log("ERROR: " + errors[-1])
        return DownloadResult(family, saved, errors)

    log("พบ " + str(len(urls)) + " ไฟล์ -- กำลังดาวน์โหลด...")

    out_dir = fonts_dir() / family.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, url in enumerate(urls, 1):
        filename = url.split("/")[-1].split("?")[0]
        dest = out_dir / filename

        if dest.exists():
            log("(" + str(i) + "/" + str(len(urls)) + ") มีอยู่แล้ว: " + filename)
            saved.append(dest)
            continue

        log("(" + str(i) + "/" + str(len(urls)) + ") ดาวน์โหลด: " + filename)
        try:
            data = _fetch_bytes(url, ua=_UA_TTF)
            dest.write_bytes(data)
            saved.append(dest)
            log("OK: " + filename + " (" + str(len(data)) + " bytes)")
        except Exception as exc:
            errors.append(filename + ": " + str(exc))
            log("ERROR: " + str(exc))

    return DownloadResult(family, saved, errors)


def load_fonts_from_dir(fonts_root=None) -> list:
    """
    โหลดทุกไฟล์ .ttf/.otf ใน fonts_root เข้า QFontDatabase
    ต้องเรียกหลัง QApplication สร้างแล้ว
    """
    from PyQt5.QtGui import QFontDatabase

    root = Path(fonts_root) if fonts_root else fonts_dir()
    loaded: list = []

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in (".ttf", ".otf"):
            fid = QFontDatabase.addApplicationFont(str(path))
            if fid >= 0:
                loaded.extend(QFontDatabase.applicationFontFamilies(fid))

    return loaded
