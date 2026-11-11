"""
data/config_handler.py
Read/write per-station JSON configuration.

All zone polygons, camera settings, alert thresholds, and golden-cycle
references are stored in  config/<station_id>.json  (or a master config
file).  The handler provides typed accessors so the rest of the app
never has to parse raw JSON.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any


# Default config location inside the package tree
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default_config.json"
_USER_CONFIG_DIR = Path(__file__).parent.parent / "config"


class ConfigHandler:
    """Load, mutate, and persist application configuration."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._path: Path = (
            Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        )
        self._data: dict[str, Any] = {}
        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load configuration from JSON file (creates default if missing)."""
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            # Copy the bundled default
            with open(_DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            self.save()

    def save(self) -> None:
        """Persist current configuration to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Top-level accessors
    # ------------------------------------------------------------------

    @property
    def active_station(self) -> str:
        return self._data.get("active_station", "station_01")

    @active_station.setter
    def active_station(self, value: str) -> None:
        self._data["active_station"] = value

    @property
    def anthropic_api_key(self) -> str:
        return self._data.get("anthropic_api_key", "")

    @anthropic_api_key.setter
    def anthropic_api_key(self, value: str) -> None:
        self._data["anthropic_api_key"] = value

    @property
    def anthropic_model(self) -> str:
        return self._data.get("anthropic_model", "claude-sonnet-4-5")

    @property
    def ui(self) -> dict[str, Any]:
        return self._data.get("ui", {})

    # ------------------------------------------------------------------
    # Station accessors
    # ------------------------------------------------------------------

    def station_ids(self) -> list[str]:
        return list(self._data.get("stations", {}).keys())

    def get_station(self, station_id: str | None = None) -> dict[str, Any]:
        sid = station_id or self.active_station
        stations = self._data.setdefault("stations", {})
        if sid not in stations:
            stations[sid] = deepcopy(
                self._data["stations"].get("station_01", {})
            )
        return stations[sid]

    def set_station_name(self, station_id: str, name: str) -> None:
        self.get_station(station_id)["name"] = name

    # ------------------------------------------------------------------
    # Camera config
    # ------------------------------------------------------------------

    def get_camera_config(self, station_id: str | None = None) -> dict[str, Any]:
        return self.get_station(station_id).get("camera", {})

    def set_camera_config(
        self,
        camera_type: str,
        device_index: int = 0,
        url: str = "",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        mirror_horizontal: bool = False,
        file_path: str = "",
        loop: bool = True,
        station_id: str | None = None,
    ) -> None:
        cam: dict = {
            "type":              camera_type,
            "device_index":      device_index,
            "url":               url,
            "width":             width,
            "height":            height,
            "fps":               fps,
            "mirror_horizontal": mirror_horizontal,
        }
        if camera_type == "video_file":
            cam["file_path"] = file_path
            cam["loop"]      = loop
        self.get_station(station_id)["camera"] = cam

    # ------------------------------------------------------------------
    # Zone config
    # ------------------------------------------------------------------

    def get_zones(self, station_id: str | None = None) -> list[dict[str, Any]]:
        return self.get_station(station_id).get("zones", [])

    def set_zone_polygon(
        self,
        zone_id: int,
        polygon: list[list[int]],
        station_id: str | None = None,
    ) -> None:
        """Update the polygon (list of [x,y] points) for a zone."""
        zones = self.get_zones(station_id)
        for zone in zones:
            if zone["id"] == zone_id:
                zone["polygon"] = polygon
                return
        # Zone doesn't exist yet — append
        zones.append({"id": zone_id, "name": f"Zone {zone_id}", "polygon": polygon})

    def set_zones(
        self,
        zones: list[dict[str, Any]],
        station_id: str | None = None,
    ) -> None:
        self.get_station(station_id)["zones"] = zones

    def zones_are_calibrated(self, station_id: str | None = None) -> bool:
        """Return True only when every zone has a non-empty polygon."""
        zones = self.get_zones(station_id)
        return bool(zones) and all(len(z.get("polygon", [])) >= 3 for z in zones)

    # ------------------------------------------------------------------
    # Alert / threshold config
    # ------------------------------------------------------------------

    def get_alert_threshold(self, station_id: str | None = None) -> int:
        return self.get_station(station_id).get("alert_threshold_percent", 30)

    def set_alert_threshold(
        self, value: int, station_id: str | None = None
    ) -> None:
        self.get_station(station_id)["alert_threshold_percent"] = value

    # ------------------------------------------------------------------
    # Golden cycle config
    # ------------------------------------------------------------------

    def get_golden_cycle(self, station_id: str | None = None) -> dict[str, Any]:
        return self.get_station(station_id).get("golden_cycle", {})

    def set_golden_cycle(
        self,
        standard_times: dict,
        trajectory_points: list,
        recorded_cycles: list,
        total_standard_time: float | None = None,
        station_id: str | None = None,
    ) -> None:
        self.get_station(station_id)["golden_cycle"] = {
            "recorded_cycles":    recorded_cycles,
            "standard_times":     standard_times,
            "total_standard_time": total_standard_time,
            "trajectory_points":  trajectory_points,
        }

    def has_golden_cycle(self, station_id: str | None = None) -> bool:
        gc = self.get_golden_cycle(station_id)
        return bool(gc.get("standard_times"))

    def get_min_golden_cycles(self, station_id: str | None = None) -> int:
        return self.get_station(station_id).get("min_golden_cycles", 3)

    def get_max_golden_cycles(self, station_id: str | None = None) -> int:
        return self.get_station(station_id).get("max_golden_cycles", 5)

    # ------------------------------------------------------------------
    # UI preferences
    # ------------------------------------------------------------------

    def get_show_ghost(self) -> bool:
        """Return whether the ghost overlay is enabled (default: True)."""
        return bool(self._data.get("ui", {}).get("show_ghost_overlay", True))

    def set_show_ghost(self, value: bool) -> None:
        """Persist ghost overlay toggle across sessions."""
        self._data.setdefault("ui", {})["show_ghost_overlay"] = bool(value)
        self.save()

    def get_show_skeleton(self) -> bool:
        """Return whether the hand skeleton overlay is enabled (default: True)."""
        return bool(self._data.get("ui", {}).get("show_hand_skeleton", True))

    def set_show_skeleton(self, value: bool) -> None:
        """Persist hand skeleton overlay toggle across sessions."""
        self._data.setdefault("ui", {})["show_hand_skeleton"] = bool(value)
        self.save()

    # ------------------------------------------------------------------
    # RTSP Bookmarks
    # ------------------------------------------------------------------

    def get_rtsp_bookmarks(self) -> list:
        """Return saved RTSP presets.

        Each entry is a dict with keys:
            name, url, username, password, transport
        """
        return list(self._data.get("ui", {}).get("rtsp_bookmarks", []))

    def set_rtsp_bookmarks(self, bookmarks: list) -> None:
        """Persist RTSP bookmark list and save config."""
        self._data.setdefault("ui", {})["rtsp_bookmarks"] = bookmarks
        self.save()

    # ------------------------------------------------------------------
    # Hand Detector config
    # ------------------------------------------------------------------

    _DETECTOR_DEFAULTS: dict = {
        "backend":          "mediapipe",   # "mediapipe" | "yolo_pose"
        # MediaPipe params
        "mp_max_hands":       2,
        "mp_detection_conf":  0.6,
        "mp_tracking_conf":   0.5,
        # YOLO Pose params
        "yolo_model":         "yolov8n-pose.pt",
        "yolo_confidence":    0.4,
        "yolo_device":        "cpu",      # cpu = stable; cuda = needs matching driver
        "yolo_kp_conf_min":   0.3,
        "yolo_hand_extend":   0.35,        # fraction of forearm to extend past wrist
    }

    def get_detector_config(self) -> dict:
        """Return persisted detector config merged with defaults."""
        saved = self._data.get("ui", {}).get("detector", {})
        return {**self._DETECTOR_DEFAULTS, **saved}

    def set_detector_config(self, cfg: dict) -> None:
        """Persist detector config and save."""
        self._data.setdefault("ui", {})["detector"] = cfg
        self.save()

    # ------------------------------------------------------------------
    # Font settings
    # ------------------------------------------------------------------

    _FONT_DEFAULTS: dict = {
        "font_family":      "Leelawadee UI",
        "font_size_body":   14,
        "font_size_table":  10,
        "font_size_ai":     16,
        "font_size_title":  11,
        "font_size_section":12,
        "font_size_label":  10,
        "font_size_caption": 7,
    }

    def get_font_settings(self) -> dict:
        """Return persisted font settings merged with defaults."""
        saved = self._data.get("ui", {}).get("fonts", {})
        return {**self._FONT_DEFAULTS, **saved}

    def set_font_settings(self, settings: dict) -> None:
        """Persist font settings and save config."""
        self._data.setdefault("ui", {})["fonts"] = settings
        self.save()

    # ------------------------------------------------------------------
    # AI settings
    # ------------------------------------------------------------------

    # Built-in default prompt template (Thai factory analysis)
    # Variables available: {station_id} {total_cycles} {pass_rate} {passed}
    # {failed} {seq_errors} {avg_time} {min_time} {max_time} {std_dev}
    # {avg_deviation} {zone_table} {alert_total} {alert_warn} {alert_crit}
    # {alert_by_zone} {viol_total} {viol_skip} {viol_oor} {viol_repeat}
    # {viol_cycles} {cycle_table} {output_language} {output_length_instruction}
    # {output_format_instruction} {operator}
    PROMPT_DEFAULT = (
        "คุณเป็นที่ปรึกษาด้านวิศวกรรมอุตสาหกรรมและการปรับปรุงกระบวนการผลิต\n"
        "โปรดวิเคราะห์ข้อมูลการ monitoring การผลิตต่อไปนี้และตอบเป็น{output_language}ทั้งหมด\n"
        "\n"
        "โครงสร้างคำตอบ (ใช้หัวข้อตามนี้):\n"
        "## 1. ภาพรวมประสิทธิภาพ\n"
        "## 2. จุดที่มีปัญหาบ่อยที่สุด / รุนแรงที่สุด\n"
        "## 3. สาเหตุที่เป็นไปได้\n"
        "## 4. ข้อเสนอแนะที่เป็นรูปธรรม (เรียงตามความสำคัญ)\n"
        "## 5. สรุปและเป้าหมายที่แนะนำ\n"
        "\n"
        "กฎสำคัญ:\n"
        "- อ้างอิงตัวเลขจากข้อมูลด้านล่างทุกครั้งที่เป็นไปได้\n"
        "- ใช้ภาษาที่หัวหน้าไลน์ผลิตที่ไม่ใช่สายเทคนิคเข้าใจได้\n"
        "- ห้ามสร้างข้อมูลที่ไม่มีในข้อมูลด้านล่าง\n"
        "{output_length_instruction}\n"
        "{output_format_instruction}\n"
        "\n"
        "=== ข้อมูล SESSION ===\n"
        "สถานี : {station_id}\n"
        "{operator}\n"
        "\n"
        "สรุปภาพรวม:\n"
        "  รอบทั้งหมด       : {total_cycles} รอบ\n"
        "  ผ่านมาตรฐาน      : {passed} รอบ ({pass_rate})\n"
        "  ไม่ผ่าน (ช้าเกิน) : {failed} รอบ\n"
        "  ผิดลำดับขั้นตอน  : {seq_errors} รอบ\n"
        "\n"
        "สถิติเวลา cycle (วินาที):\n"
        "  เฉลี่ย           : {avg_time}\n"
        "  ต่ำสุด           : {min_time}\n"
        "  สูงสุด           : {max_time}\n"
        "  ค่าเบี่ยงเบน     : {std_dev}\n"
        "  เบี่ยงเบนจาก standard เฉลี่ย: {avg_deviation}\n"
        "\n"
        "เวลาเฉลี่ยต่อโซน:\n"
        "{zone_table}\n"
        "\n"
        "การแจ้งเตือน (Alert):\n"
        "  รวมทั้งหมด       : {alert_total} ครั้ง\n"
        "  ระดับ WARNING    : {alert_warn} ครั้ง\n"
        "  ระดับ CRITICAL   : {alert_crit} ครั้ง\n"
        "  แยกตามโซน:\n"
        "{alert_by_zone}\n"
        "\n"
        "การละเมิดลำดับขั้นตอน (Sequence Violation):\n"
        "  รวม              : {viol_total} ครั้ง ใน {viol_cycles} รอบ\n"
        "  ข้ามขั้น (SKIP)  : {viol_skip} ครั้ง\n"
        "  สลับลำดับ        : {viol_oor} ครั้ง\n"
        "  ทำซ้ำ (REPEAT)   : {viol_repeat} ครั้ง\n"
        "\n"
        "รายละเอียดแต่ละรอบ:\n"
        "{cycle_table}\n"
        "=====================\n"
        "\n"
        "โปรดวิเคราะห์และให้คำแนะนำตามโครงสร้างที่กำหนด"
    )

    _AI_DEFAULTS: dict = {
        "model":           "gemini-2.0-flash",
        "timeout":         60,
        "prompt_template": None,          # None = use PROMPT_DEFAULT
        "output_language": "ภาษาไทย",
        "output_length":   "medium",      # short / medium / long
        "output_headers":  True,
        "output_bullets":  True,
        "ctx_cycle_detail": True,
        "ctx_stats":        True,
        "ctx_alerts":       True,
        "ctx_violations":   True,
        "ctx_raw_timestamps": False,
        "max_cycles_in_prompt": 30,
    }

    def get_ai_settings(self) -> dict:
        """Return persisted AI settings merged with built-in defaults."""
        saved = self._data.get("ai_settings", {})
        merged = {**self._AI_DEFAULTS, **saved}
        # Resolve None prompt -> built-in default string
        if merged["prompt_template"] is None:
            merged["prompt_template"] = self.PROMPT_DEFAULT
        return merged

    def set_ai_settings(self, settings: dict) -> None:
        """Persist AI settings. Saves None for prompt when it matches the default."""
        to_save = dict(settings)
        # Store None when user has reset to built-in default (saves space + allows future default updates)
        if to_save.get("prompt_template") == self.PROMPT_DEFAULT:
            to_save["prompt_template"] = None
        self._data["ai_settings"] = to_save
        self.save()
