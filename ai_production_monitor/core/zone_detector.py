"""
core/zone_detector.py
Zone Region-of-Interest (ROI) detection engine.

Responsibilities
----------------
- Store zone polygons (arbitrary convex or concave shapes)
- Determine whether a hand landmark point is inside a zone (point-in-polygon)
- Track entry/exit transitions with hysteresis to suppress jitter
- Expose a simple API used by both the vision thread and GUI overlay painter

Zone definition (from config JSON)
-----------------------------------
    {
        "id": 1,
        "name": "Zone 1 — Pick Part A",
        "color": [0, 120, 255],
        "polygon": [[x0,y0], [x1,y1], ...]  # pixel coords OR normalised [0-1]
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import cv2


@dataclass
class Zone:
    id: int
    name: str
    color: tuple[int, int, int] = (0, 200, 255)
    polygon: list[list[int]] = field(default_factory=list)   # [[x,y], ...]
    # Runtime state (not persisted)
    is_occupied: bool = False
    entry_frame_count: int = 0   # frames hand has been inside (hysteresis)
    exit_frame_count: int  = 0   # frames hand has been outside (hysteresis)

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #

    def polygon_np(self) -> Optional[np.ndarray]:
        """Return polygon as (N,1,2) int32 array for OpenCV, or None."""
        if len(self.polygon) < 3:
            return None
        return np.array(self.polygon, dtype=np.int32).reshape((-1, 1, 2))

    def centroid(self) -> tuple[int, int] | None:
        if not self.polygon:
            return None
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

    def contains_point(self, x: float, y: float) -> bool:
        """Test whether (x, y) lies inside the polygon (pixel coords)."""
        poly = self.polygon_np()
        if poly is None:
            return False
        result = cv2.pointPolygonTest(poly, (float(x), float(y)), measureDist=False)
        return result >= 0

    def draw(
        self,
        frame: np.ndarray,
        alpha: float = 0.25,
        show_label: bool = True,
        highlight: bool = False,
    ) -> np.ndarray:
        """Draw the zone polygon onto *frame* (in-place overlay)."""
        poly = self.polygon_np()
        if poly is None:
            return frame

        overlay = frame.copy()
        color = self.color if not highlight else (255, 255, 255)
        cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)

        if show_label:
            centroid = self.centroid()
            if centroid:
                cv2.putText(
                    frame,
                    self.name,
                    (centroid[0] - 40, centroid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        return frame


# ---------------------------------------------------------------------------
# ZoneDetector
# ---------------------------------------------------------------------------

class ZoneDetector:
    """
    Manages a collection of Zone objects and tracks hand presence
    in each zone with hysteresis filtering to reduce noise.

    Parameters
    ----------
    zones_config  : list of zone dicts from ConfigHandler
    entry_frames  : how many consecutive "inside" frames before entry fires
    exit_frames   : how many consecutive "outside" frames before exit fires
    frame_size    : (width, height) of the video frame; used to convert
                    normalised [0,1] coords to pixels if needed
    """

    def __init__(
        self,
        zones_config: list[dict],
        entry_frames: int = 2,
        exit_frames: int  = 3,
        frame_size: tuple[int, int] = (1280, 720),
    ) -> None:
        self._entry_frames = entry_frames
        self._exit_frames  = exit_frames
        self._frame_size   = frame_size
        self.zones: list[Zone] = []
        self.load_zones(zones_config)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def load_zones(self, zones_config: list[dict]) -> None:
        """(Re)load zone definitions. Resets all runtime state."""
        self.zones = []
        for cfg in zones_config:
            poly = cfg.get("polygon", [])
            poly = self._ensure_pixel_coords(poly)
            color_list = cfg.get("color", [0, 200, 255])
            self.zones.append(
                Zone(
                    id=cfg["id"],
                    name=cfg.get("name", f"Zone {cfg['id']}"),
                    color=tuple(color_list),   # type: ignore[arg-type]
                    polygon=poly,
                )
            )

    def update_zone_polygon(self, zone_id: int, polygon: list[list[int]]) -> None:
        for z in self.zones:
            if z.id == zone_id:
                z.polygon = self._ensure_pixel_coords(polygon)
                return

    def set_frame_size(self, width: int, height: int) -> None:
        self._frame_size = (width, height)

    def _ensure_pixel_coords(self, polygon: list) -> list[list[int]]:
        """Convert [0,1]-normalised polygon to pixel coords if needed."""
        if not polygon:
            return polygon
        # If all values ≤ 1.0 treat as normalised
        flat = [v for pt in polygon for v in pt]
        if flat and max(flat) <= 1.0:
            w, h = self._frame_size
            return [[int(p[0] * w), int(p[1] * h)] for p in polygon]
        return [[int(p[0]), int(p[1])] for p in polygon]

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(
        self,
        hand_x: float,
        hand_y: float,
        hand_detected: bool,
    ) -> list[tuple[int, str]]:
        """
        Call once per video frame with the current hand wrist position.

        Parameters
        ----------
        hand_x, hand_y  : wrist pixel coordinates (or 0,0 if no hand)
        hand_detected   : whether a hand is currently visible

        Returns
        -------
        List of (zone_id, event_type) tuples generated this frame.
        event_type is "enter" or "exit".
        """
        events: list[tuple[int, str]] = []

        for zone in self.zones:
            inside = hand_detected and zone.contains_point(hand_x, hand_y)

            if inside:
                zone.entry_frame_count += 1
                zone.exit_frame_count   = 0
            else:
                zone.exit_frame_count  += 1
                zone.entry_frame_count  = 0

            if not zone.is_occupied and zone.entry_frame_count >= self._entry_frames:
                zone.is_occupied = True
                events.append((zone.id, "enter"))

            elif zone.is_occupied and zone.exit_frame_count >= self._exit_frames:
                zone.is_occupied = False
                events.append((zone.id, "exit"))

        return events

    def reset_all(self) -> None:
        """Reset occupation state for all zones (call on cycle start)."""
        for zone in self.zones:
            zone.is_occupied       = False
            zone.entry_frame_count = 0
            zone.exit_frame_count  = 0

    # ------------------------------------------------------------------
    # Frame drawing
    # ------------------------------------------------------------------

    def draw_zones(
        self,
        frame: np.ndarray,
        alpha: float = 0.20,
        active_zone_ids: set[int] | None = None,
    ) -> np.ndarray:
        """Draw all zones on frame. Active zones are highlighted."""
        active = active_zone_ids or set()
        for zone in self.zones:
            zone.draw(frame, alpha=alpha, highlight=zone.id in active)
        return frame

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_zone(self, zone_id: int) -> Zone | None:
        for z in self.zones:
            if z.id == zone_id:
                return z
        return None

    def occupied_zone_ids(self) -> list[int]:
        return [z.id for z in self.zones if z.is_occupied]

    def all_calibrated(self) -> bool:
        return all(len(z.polygon) >= 3 for z in self.zones)

    @property
    def zone_ids(self) -> list[int]:
        return [z.id for z in self.zones]
