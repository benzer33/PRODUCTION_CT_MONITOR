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
        station_id: str | None = None,
    ) -> None:
        self.get_station(station_id)["camera"] = {
            "type": camera_type,
            "device_index": device_index,
            "url": url,
            "width": width,
            "height": height,
            "fps": fps,
        }

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
        station_id: str | None = None,
    ) -> None:
        self.get_station(station_id)["golden_cycle"] = {
            "recorded_cycles": recorded_cycles,
            "standard_times":  standard_times,
            "trajectory_points": trajectory_points,
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
