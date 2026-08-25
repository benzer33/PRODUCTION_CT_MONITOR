"""
vision/hand_tracker.py
MediaPipe Hands wrapper — extracts the wrist landmark position per frame.

Design decisions
----------------
- Only the WRIST landmark (index 0) is used as the hand position proxy.
  This is robust to partial occlusion and is the most stable landmark.
- Both hands are tracked; the tracker returns the "most active" hand
  (nearest to the last known position) to avoid switching.
- All coordinates are returned in PIXEL space (not normalised) so the
  rest of the pipeline deals only with one coordinate system.
- The wrapper exposes a draw() helper that renders landmarks + connections
  directly onto the BGR frame for debug/calibration views.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# MediaPipe ≥ 0.10 moved solutions out of the top-level namespace.
# Import directly from the subpackage to work across all versions.
try:
    from mediapipe.python.solutions import hands as _mp_hands_mod
    from mediapipe.python.solutions import drawing_utils as _mp_drawing_mod
    from mediapipe.python.solutions import drawing_styles as _mp_styles_mod
except ImportError:
    # Fallback for older mediapipe < 0.10 that still has mp.solutions
    import mediapipe as _mp_legacy
    _mp_hands_mod   = _mp_legacy.solutions.hands
    _mp_drawing_mod = _mp_legacy.solutions.drawing_utils
    _mp_styles_mod  = _mp_legacy.solutions.drawing_styles


@dataclass
class HandResult:
    """Result of processing one video frame."""
    detected:   bool
    x:          float   # wrist pixel x
    y:          float   # wrist pixel y
    x_norm:     float   # wrist normalised [0,1]
    y_norm:     float   # wrist normalised [0,1]
    landmarks:  list    # raw MediaPipe landmark list (all 21)
    hand_label: str     # "Left" | "Right" | ""  (backward-compat alias)
    # Phase-1 dual-hand field: explicit handedness string.
    # NOTE (Phase-2 warning): MediaPipe Hands does NOT maintain cross-frame
    # identity when two hands cross/overlap.  The label may swap for 1-3 frames
    # during occlusion.  Phase-2 false-positive filtering MUST smooth/debounce
    # this field before using it for anomaly detection.
    handedness: str = ""  # "Left" | "Right" | "" — set equal to hand_label on construction

    def __post_init__(self):
        # Keep handedness in sync with hand_label if not explicitly provided
        if not self.handedness:
            self.handedness = self.hand_label


# ---------------------------------------------------------------------------
# HandTracker
# ---------------------------------------------------------------------------

class HandTracker:
    """
    Thin wrapper around mediapipe.solutions.hands.Hands.

    Parameters
    ----------
    max_num_hands        : how many hands to detect simultaneously (1 or 2)
    detection_confidence : min confidence for hand detection
    tracking_confidence  : min confidence for landmark tracking
    static_image_mode    : set True for single images, False (default) for video
    """

    WRIST_IDX = 0   # MediaPipe landmark index for wrist

    def __init__(
        self,
        max_num_hands: int = 2,
        detection_confidence: float = 0.6,
        tracking_confidence: float  = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        self._mp_hands   = _mp_hands_mod
        self._mp_drawing = _mp_drawing_mod
        self._mp_styles  = _mp_styles_mod

        self._hands = self._mp_hands.Hands(
            static_image_mode        = static_image_mode,
            max_num_hands            = max_num_hands,
            min_detection_confidence = detection_confidence,
            min_tracking_confidence  = tracking_confidence,
        )

        # Track last known position to pick the "most active" hand
        self._last_x: float = -1.0
        self._last_y: float = -1.0

    # ------------------------------------------------------------------
    # Main processing method
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> HandResult:
        """
        Detect hands in *frame* and return the **primary** (most-active) hand.

        Backward-compatible single-hand API.
        Use process_all() for dual-hand tracking (Phase-1+).

        Parameters
        ----------
        frame : BGR uint8 frame from OpenCV

        Returns
        -------
        HandResult with detected=True and pixel coords when a hand is found.
        """
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            return HandResult(
                detected=False, x=0, y=0,
                x_norm=0, y_norm=0,
                landmarks=[], hand_label=""
            )

        # Pick best hand
        best_lm, best_label = self._pick_best_hand(
            results.multi_hand_landmarks,
            results.multi_handedness,
            w, h,
        )

        wrist    = best_lm.landmark[self.WRIST_IDX]
        px       = wrist.x * w
        py       = wrist.y * h
        self._last_x = px
        self._last_y = py

        return HandResult(
            detected   = True,
            x          = px,
            y          = py,
            x_norm     = wrist.x,
            y_norm     = wrist.y,
            landmarks  = best_lm.landmark,
            hand_label = best_label,
        )

    def process_all(self, frame: np.ndarray) -> list:
        """
        Detect hands in *frame* and return ALL detected hands (0, 1, or 2).

        Returns a list[HandResult], one per detected hand.  Each HandResult has
        .handedness set to "Left" or "Right" from MediaPipe's classification.

        NOTE (Phase-2 warning): When two hands cross or overlap, MediaPipe may
        swap the Left/Right label for 1-3 frames due to lack of true cross-frame
        identity tracking.  Phase-2 anomaly detection MUST apply temporal
        smoothing/debouncing before flagging wrong-hand events to avoid false
        alarms from this known MediaPipe limitation.

        Parameters
        ----------
        frame : BGR uint8 frame from OpenCV

        Returns
        -------
        list[HandResult] — empty when no hand is detected.
        """
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            return []

        hands: list = []
        for idx, lm_set in enumerate(results.multi_hand_landmarks):
            label = (
                results.multi_handedness[idx].classification[0].label
                if results.multi_handedness else ""
            )
            wrist  = lm_set.landmark[self.WRIST_IDX]
            px     = wrist.x * w
            py     = wrist.y * h
            hands.append(HandResult(
                detected   = True,
                x          = px,
                y          = py,
                x_norm     = wrist.x,
                y_norm     = wrist.y,
                landmarks  = lm_set.landmark,
                hand_label = label,
                handedness = label,
            ))
        return hands

    # ------------------------------------------------------------------
    # Drawing helper
    # ------------------------------------------------------------------

    def draw_landmarks(
        self,
        frame: np.ndarray,
        hand_result: HandResult,
    ) -> np.ndarray:
        """
        Draw MediaPipe hand landmarks on frame (for debug/calibration views).
        Returns the annotated frame (modifies in-place).
        """
        if not hand_result.detected:
            return frame

        # Reconstruct landmark list in MediaPipe format for drawing
        # We reprocess to get the full multi_hand_landmarks object; instead,
        # draw manually from stored landmarks for efficiency.
        h, w = frame.shape[:2]
        for lm in hand_result.landmarks:
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # Draw wrist highlight
        if hand_result.detected:
            cv2.circle(
                frame,
                (int(hand_result.x), int(hand_result.y)),
                10, (0, 200, 255), 2
            )
        return frame

    def draw_wrist_only(
        self,
        frame: np.ndarray,
        hand_result: HandResult,
        color: tuple[int, int, int] = (0, 200, 255),
        radius: int = 12,
    ) -> np.ndarray:
        if hand_result.detected:
            cv2.circle(
                frame,
                (int(hand_result.x), int(hand_result.y)),
                radius, color, -1,
            )
        return frame

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _pick_best_hand(
        self,
        multi_lm,
        multi_handedness,
        w: int,
        h: int,
    ):
        """
        When multiple hands detected, choose the one closest to last wrist pos.
        Falls back to the first hand if no history.
        """
        if len(multi_lm) == 1:
            label = (
                multi_handedness[0].classification[0].label
                if multi_handedness else ""
            )
            return multi_lm[0], label

        best_idx  = 0
        best_dist = float("inf")
        for i, lm in enumerate(multi_lm):
            wx = lm.landmark[self.WRIST_IDX].x * w
            wy = lm.landmark[self.WRIST_IDX].y * h
            if self._last_x >= 0:
                dist = (wx - self._last_x) ** 2 + (wy - self._last_y) ** 2
            else:
                dist = 0.0
            if dist < best_dist:
                best_dist = dist
                best_idx  = i

        label = (
            multi_handedness[best_idx].classification[0].label
            if multi_handedness else ""
        )
        return multi_lm[best_idx], label

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._hands.close()

    def __del__(self) -> None:
        try:
            self._hands.close()
        except Exception:
            pass
