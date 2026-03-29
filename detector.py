"""
detector.py — Eye-Blink-to-Speech Assistive Technology
========================================================
Handles all blink detection logic:
  • EAR calculation via MediaPipe Face Mesh landmarks
  • Smoothed EAR buffer (deque)
  • Auto-calibration phase
  • State-machine for Short / Long blink classification
  • Async TTS via pyttsx3 on a daemon thread

Author : <Your Name>
Version: 2.0.0  (production-ready rebuild)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, List, Optional, Tuple

import numpy as np
import pyttsx3

# ---------------------------------------------------------------------------
# Types & constants
# ---------------------------------------------------------------------------

# MediaPipe Face Mesh landmark indices for the left and right eyes
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py
_LEFT_EYE_IDX: Tuple[int, ...] = (362, 385, 387, 263, 373, 380)
_RIGHT_EYE_IDX: Tuple[int, ...] = (33, 160, 158, 133, 153, 144)

# Calibration defaults (overridden during auto-calibration)
_DEFAULT_EAR_OPEN: float = 0.30
_DEFAULT_EAR_CLOSED: float = 0.18

# Blink duration thresholds (seconds)
SHORT_BLINK_MAX: float = 0.45     # < 0.45s  → natural reflex, ignore
LONG_BLINK_MIN: float = 0.50      # 0.50 – 1.20s → "Selection" event
LONG_BLINK_MAX: float = 1.20


class DetectorState(Enum):
    CALIBRATING = auto()
    ACTIVE = auto()


class BlinkEvent(Enum):
    SHORT = auto()    # reflexive blink — ignored by app
    LONG = auto()     # intentional selection blink
    NONE = auto()


# ---------------------------------------------------------------------------
# Calibration result
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    ear_open: float = _DEFAULT_EAR_OPEN
    ear_closed: float = _DEFAULT_EAR_CLOSED
    threshold: float = field(init=False)

    def __post_init__(self) -> None:
        self._compute_threshold()

    def _compute_threshold(self) -> None:
        # Threshold sits 40 % of the way down from open → closed
        self.threshold = self.ear_open - 0.40 * (self.ear_open - self.ear_closed)

    def update(self, ear_open: float, ear_closed: float) -> None:
        self.ear_open = ear_open
        self.ear_closed = ear_closed
        self._compute_threshold()


# ---------------------------------------------------------------------------
# Async TTS engine
# ---------------------------------------------------------------------------

class AsyncTTS:
    """
    Wraps pyttsx3 so speech runs in a daemon thread, keeping the
    video pipeline non-blocking.
    """

    def __init__(self, rate: int = 165, volume: float = 1.0) -> None:
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)
        self._engine.setProperty("volume", volume)
        self._lock = threading.Lock()
        self._busy = False

    @property
    def is_busy(self) -> bool:
        return self._busy

    def speak(self, text: str) -> None:
        """Non-blocking call — fires and forgets on a daemon thread."""
        if self._busy:
            return  # drop overlapping requests gracefully
        thread = threading.Thread(target=self._run, args=(text,), daemon=True)
        thread.start()

    def _run(self, text: str) -> None:
        with self._lock:
            self._busy = True
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            finally:
                self._busy = False


# ---------------------------------------------------------------------------
# EAR calculator
# ---------------------------------------------------------------------------

def compute_ear(landmarks: List, indices: Tuple[int, ...], img_w: int, img_h: int) -> float:
    """
    Calculate Eye Aspect Ratio for one eye.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Args:
        landmarks : MediaPipe NormalizedLandmarkList points.
        indices   : 6 landmark indices [p1..p6] for the eye.
        img_w     : Frame pixel width  (for de-normalising).
        img_h     : Frame pixel height (for de-normalising).

    Returns:
        EAR as a float in [0, 1].
    """
    def _pt(idx: int) -> np.ndarray:
        lm = landmarks[idx]
        return np.array([lm.x * img_w, lm.y * img_h], dtype=np.float64)

    p1, p2, p3, p4, p5, p6 = (_pt(i) for i in indices)

    vertical_a = np.linalg.norm(p2 - p6)
    vertical_b = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal < 1e-6:          # guard division by zero
        return 0.0
    return (vertical_a + vertical_b) / (2.0 * horizontal)


# ---------------------------------------------------------------------------
# Core blink detector
# ---------------------------------------------------------------------------

class BlinkDetector:
    """
    Stateful blink detector that operates as a simple state machine.

    States
    ------
    CALIBRATING  — Collects EAR samples for ``calibration_duration`` seconds.
    ACTIVE       — Monitors blinks and fires BlinkEvent callbacks.

    Usage
    -----
    >>> detector = BlinkDetector()
    >>> detector.on_long_blink = lambda: tts.speak("Hello")
    >>> while cap.isOpened():
    ...     ear = detector.process_frame(landmarks, w, h)
    """

    def __init__(
        self,
        buffer_size: int = 7,
        calibration_duration: float = 5.0,
    ) -> None:
        self._buffer: Deque[float] = deque(maxlen=buffer_size)
        self._calibration_duration = calibration_duration
        self._calibration_samples: List[float] = []
        self._calibration_start: Optional[float] = None

        self.calibration: CalibrationResult = CalibrationResult()
        self.state: DetectorState = DetectorState.CALIBRATING

        # Blink-timing state
        self._eye_closed: bool = False
        self._blink_start: Optional[float] = None

        # Public callbacks (set by app layer)
        self.on_short_blink: Optional[callable] = None
        self.on_long_blink: Optional[callable] = None

        # Diagnostics (read by HUD)
        self.current_ear: float = 0.0
        self.last_event: BlinkEvent = BlinkEvent.NONE
        self.calibration_progress: float = 0.0   # 0.0 → 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self,
        landmarks: List,
        img_w: int,
        img_h: int,
    ) -> float:
        """
        Main per-frame entry point.

        Args:
            landmarks : MediaPipe face landmark list for a single face.
            img_w     : Frame width in pixels.
            img_h     : Frame height in pixels.

        Returns:
            Smoothed EAR value for the current frame.
        """
        left_ear = compute_ear(landmarks, _LEFT_EYE_IDX, img_w, img_h)
        right_ear = compute_ear(landmarks, _RIGHT_EYE_IDX, img_w, img_h)
        raw_ear = (left_ear + right_ear) / 2.0

        self._buffer.append(raw_ear)
        smoothed = float(np.mean(self._buffer))
        self.current_ear = smoothed

        if self.state == DetectorState.CALIBRATING:
            self._run_calibration(smoothed)
        else:
            self._run_state_machine(smoothed)

        return smoothed

    def reset_calibration(self) -> None:
        """Restart the calibration phase (e.g., on environment change)."""
        self._calibration_samples.clear()
        self._calibration_start = None
        self.calibration_progress = 0.0
        self.state = DetectorState.CALIBRATING

    # ------------------------------------------------------------------
    # Internal: calibration
    # ------------------------------------------------------------------

    def _run_calibration(self, ear: float) -> None:
        now = time.monotonic()

        if self._calibration_start is None:
            self._calibration_start = now

        elapsed = now - self._calibration_start
        self.calibration_progress = min(elapsed / self._calibration_duration, 1.0)
        self._calibration_samples.append(ear)

        if elapsed >= self._calibration_duration:
            self._finalise_calibration()

    def _finalise_calibration(self) -> None:
        samples = np.array(self._calibration_samples)
        # Open-eye baseline: upper 60th-percentile of samples
        ear_open = float(np.percentile(samples, 60))
        # Closed-eye estimate: lower 10th-percentile
        ear_closed = float(np.percentile(samples, 10))

        self.calibration.update(ear_open, ear_closed)
        self.state = DetectorState.ACTIVE
        self.calibration_progress = 1.0

    # ------------------------------------------------------------------
    # Internal: state machine
    # ------------------------------------------------------------------

    def _run_state_machine(self, ear: float) -> None:
        threshold = self.calibration.threshold
        now = time.monotonic()
        is_below = ear < threshold

        if is_below and not self._eye_closed:
            # Eye just closed → start timing
            self._eye_closed = True
            self._blink_start = now

        elif not is_below and self._eye_closed:
            # Eye just opened → classify the blink
            self._eye_closed = False
            if self._blink_start is not None:
                duration = now - self._blink_start
                self._classify_blink(duration)
            self._blink_start = None

    def _classify_blink(self, duration: float) -> None:
        if duration < SHORT_BLINK_MAX:
            self.last_event = BlinkEvent.SHORT
            if self.on_short_blink:
                self.on_short_blink()

        elif LONG_BLINK_MIN <= duration <= LONG_BLINK_MAX:
            self.last_event = BlinkEvent.LONG
            if self.on_long_blink:
                self.on_long_blink()
        # durations > LONG_BLINK_MAX are ignored (extended closure)