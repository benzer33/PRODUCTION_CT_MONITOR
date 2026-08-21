"""
vision/skeleton_overlay.py
==========================
Pure helper for converting MediaPipe hand landmarks into a list of pixel-space
(x, y) tuples and for defining the HAND_CONNECTIONS edge list.

NOTE: Core trigger logic is NOT touched here.  This module is strictly a
visual layer.  The PointTriggerDetector continues to use only the wrist /
palm-centroid position (WRIST_IDX = 0) for all trigger-state decisions.

Design
------
* We deliberately avoid calling mediapipe.drawing_utils.draw_landmarks()
  because it writes directly onto a BGR numpy frame — we want to draw in the
  Qt paintEvent instead so the overlay stays a transparent QWidget layer and
  does not bake pixels into the emitted frame.
* The module exposes:
    - HAND_CONNECTIONS  : frozenset of (int, int) index pairs  (same as MP)
    - landmarks_to_pixels(landmarks, frame_w, frame_h) -> list[(x, y)]
    - SKELETON_COLOR / JOINT_COLOR constants for the cyan visual style
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# HAND_CONNECTIONS — MediaPipe standard 21-point hand skeleton
# ---------------------------------------------------------------------------
# Thumb:   0-1-2-3-4
# Index:   0-5-6-7-8
# Middle:  0-9-10-11-12
# Ring:    0-13-14-15-16
# Pinky:   0-17-18-19-20
# Palm base (MCP ring): 5-9-13-17, plus 0-17 and 0-5 already above
# This matches mediapipe.python.solutions.hands.HAND_CONNECTIONS exactly.

HAND_CONNECTIONS: frozenset[tuple[int, int]] = frozenset([
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm base ring
    (5, 9), (9, 13), (13, 17),
])

# ---------------------------------------------------------------------------
# Visual style constants  (cyan tone — distinct from green ghost/zone markers)
# ---------------------------------------------------------------------------
# Qt-compatible (R, G, B, A) tuples used by GhostOverlayWidget.paintEvent
SKELETON_LINE_COLOR  = (0, 210, 220, 160)   # cyan, semi-transparent
SKELETON_JOINT_COLOR = (0, 240, 255, 200)   # brighter cyan for joints
SKELETON_LINE_WIDTH  = 1                    # px — thin, non-intrusive
SKELETON_JOINT_RADIUS = 3                   # px


# ---------------------------------------------------------------------------
# Conversion helper
# ---------------------------------------------------------------------------

def landmarks_to_pixels(
    landmarks,          # iterable of mediapipe NormalizedLandmark (x, y in [0,1])
    frame_w: int,
    frame_h: int,
) -> list[tuple[float, float]]:
    """Convert normalised MediaPipe landmarks to frame-pixel coordinates.

    Parameters
    ----------
    landmarks : list/sequence of objects with .x and .y attributes in [0, 1]
    frame_w   : frame width in pixels
    frame_h   : frame height in pixels

    Returns
    -------
    List of 21 (x_px, y_px) float tuples in the same pixel space as the
    wrist/trigger coordinates used by PointTriggerDetector.
    """
    return [(lm.x * frame_w, lm.y * frame_h) for lm in landmarks]
