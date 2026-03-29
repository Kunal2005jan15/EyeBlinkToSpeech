"""
app.py — Eye-Blink-to-Speech Assistive Technology
===================================================
Entry point + OpenCV HUD.

HUD panels
----------
  • Top-left  : Status bar  (state, EAR value, last spoken phrase)
  • Bottom     : Live EAR waveform graph
  • Overlay    : Calibration progress bar

Run
---
    python app.py

Keyboard shortcuts
------------------
    R  — restart calibration
    Q  — quit

Author : <Your Name>
Version: 2.0.0
"""

from __future__ import annotations

import sys
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from detector import AsyncTTS, BlinkDetector, BlinkEvent, DetectorState

# ---------------------------------------------------------------------------
# MediaPipe compatibility shim
# mediapipe >= 0.10.10 moved FaceMesh out of mp.solutions.
# This shim supports both the legacy API (0.10.x) and the new Tasks API.
# ---------------------------------------------------------------------------

def _build_face_mesh():
    """Return a FaceMesh-compatible object regardless of mediapipe version."""
    try:
        # Legacy path — mediapipe <= 0.10.9
        fm = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        return fm, "legacy"
    except AttributeError:
        pass

    # New Tasks API path — mediapipe >= 0.10.10
    try:
        import urllib.request, os, tempfile
        model_path = os.path.join(tempfile.gettempdir(), "face_landmarker.task")
        if not os.path.exists(model_path):
            print("[INFO] Downloading face_landmarker.task model (~5 MB) ...")
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
            urllib.request.urlretrieve(url, model_path)
            print("[INFO] Model downloaded.")
        from mediapipe.tasks import python as _mp_py
        from mediapipe.tasks.python import vision as _mp_vision
        opts = _mp_vision.FaceLandmarkerOptions(
            base_options=_mp_py.BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        fm = _mp_vision.FaceLandmarker.create_from_options(opts)
        return fm, "tasks"
    except Exception as exc:
        print(f"[ERROR] Could not initialise MediaPipe FaceMesh: {exc}")
        import sys; sys.exit(1)


class _FaceMeshWrapper:
    """Uniform interface over both MediaPipe API generations."""

    def __init__(self):
        self._model, self._mode = _build_face_mesh()

    def process(self, rgb_frame: np.ndarray):
        if self._mode == "legacy":
            return self._model.process(rgb_frame)
        # Tasks API: wrap result to match legacy attribute names
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._model.detect(mp_image)
        return _TasksResultAdapter(result)

    def close(self):
        self._model.close()


class _TasksResultAdapter:
    """Makes Tasks API result look like legacy FaceMesh result."""

    def __init__(self, result):
        self._result = result

    @property
    def multi_face_landmarks(self):
        if not self._result.face_landmarks:
            return None
        return [_LandmarkListAdapter(lms) for lms in self._result.face_landmarks]


class _LandmarkListAdapter:
    def __init__(self, landmarks):
        self.landmark = landmarks  # already a list of NormalizedLandmark-like objects

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720

# Phrase cycling — each long-blink speaks the next phrase in the list
PHRASES: List[str] = [
    "Hello",
    "I need help",
    "Thank you",
    "Yes",
    "No",
    "Water please",
    "I am in pain",
    "Call the nurse",
]

# HUD colours  (BGR)
CLR_BG: Tuple[int, int, int] = (15, 15, 20)
CLR_PANEL: Tuple[int, int, int] = (28, 28, 38)
CLR_ACCENT: Tuple[int, int, int] = (0, 215, 130)       # teal-green
CLR_WARN: Tuple[int, int, int] = (30, 180, 255)         # amber
CLR_DANGER: Tuple[int, int, int] = (50, 60, 230)        # red
CLR_TEXT: Tuple[int, int, int] = (220, 220, 220)
CLR_DIM: Tuple[int, int, int] = (100, 100, 110)
CLR_GRAPH: Tuple[int, int, int] = (0, 215, 130)
CLR_THRESH: Tuple[int, int, int] = (50, 160, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

# EAR graph history length (frames)
EAR_HISTORY: int = 200


# ---------------------------------------------------------------------------
# HUD helpers
# ---------------------------------------------------------------------------

def _filled_rect(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    color: Tuple[int, int, int],
    alpha: float = 1.0,
    radius: int = 6,
) -> None:
    """Draw a (optionally semi-transparent) rounded rectangle."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), color, -1)
    for cx, cy in [(x + radius, y + radius), (x + w - radius, y + radius),
                   (x + radius, y + h - radius), (x + w - radius, y + h - radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, -1)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    else:
        frame[:] = overlay


def _text(
    frame: np.ndarray,
    text: str,
    x: int, y: int,
    color: Tuple[int, int, int] = CLR_TEXT,
    scale: float = 0.55,
    thickness: int = 1,
    font=FONT,
) -> Tuple[int, int]:
    """Draw text and return (width, height) of the rendered string."""
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(frame, text, (x, y + th), font, scale, color, thickness, cv2.LINE_AA)
    return tw, th


def _draw_status_panel(
    frame: np.ndarray,
    detector: BlinkDetector,
    last_phrase: str,
    phrase_index: int,
    fps: float,
) -> None:
    """Draws the top-left status HUD panel."""
    px, py, pw, ph = 12, 12, 420, 128
    _filled_rect(frame, px, py, pw, ph, CLR_PANEL, alpha=0.85)

    # State indicator
    if detector.state == DetectorState.CALIBRATING:
        state_label = "  CALIBRATING"
        state_color = CLR_WARN
    else:
        state_label = "  ACTIVE"
        state_color = CLR_ACCENT

    _text(frame, state_label, px + 10, py + 8, state_color, scale=0.65, thickness=2, font=FONT_BOLD)

    # EAR value
    ear_str = f"EAR : {detector.current_ear:.4f}"
    _text(frame, ear_str, px + 10, py + 38, CLR_TEXT, scale=0.55)

    # Threshold
    thr_str = f"THR : {detector.calibration.threshold:.4f}"
    _text(frame, thr_str, px + 180, py + 38, CLR_THRESH, scale=0.55)

    # FPS
    fps_str = f"FPS : {fps:.1f}"
    _text(frame, fps_str, px + 320, py + 38, CLR_DIM, scale=0.50)

    # Last spoken
    phrase_disp = f'"{last_phrase}"' if last_phrase else "—"
    _text(frame, "LAST SPOKEN:", px + 10, py + 66, CLR_DIM, scale=0.44)
    _text(frame, phrase_disp, px + 10, py + 90, CLR_ACCENT, scale=0.60, thickness=2, font=FONT_BOLD)

    # Next phrase preview
    next_phrase = PHRASES[phrase_index % len(PHRASES)]
    _text(frame, f"NEXT: {next_phrase}", px + 10, py + 112, CLR_DIM, scale=0.40)


def _draw_calibration_bar(
    frame: np.ndarray,
    detector: BlinkDetector,
    fh: int, fw: int,
) -> None:
    """Full-width calibration progress overlay."""
    if detector.state != DetectorState.CALIBRATING:
        return

    bar_h = 44
    bx, by = 0, (fh - bar_h) // 2
    _filled_rect(frame, bx, by, fw, bar_h, CLR_PANEL, alpha=0.80)

    prog = detector.calibration_progress
    fill_w = int(prog * fw)
    cv2.rectangle(frame, (bx, by + bar_h - 6), (bx + fill_w, by + bar_h), CLR_ACCENT, -1)

    cal_text = f"AUTO-CALIBRATING ...  {int(prog * 100):3d}%   Keep eyes open and blink naturally"
    (tw, th), _ = cv2.getTextSize(cal_text, FONT_BOLD, 0.60, 2)
    _text(frame, cal_text, (fw - tw) // 2, by + 10,
          CLR_TEXT, scale=0.60, thickness=2, font=FONT_BOLD)


def _draw_ear_graph(
    frame: np.ndarray,
    ear_history: Deque[float],
    threshold: float,
    fh: int, fw: int,
) -> None:
    """Draws a live EAR waveform at the bottom of the frame."""
    gh = 90        # graph height (px)
    gx, gy = 0, fh - gh
    gw = fw

    # Dark background panel
    _filled_rect(frame, gx, gy, gw, gh, CLR_PANEL, alpha=0.78, radius=0)

    if len(ear_history) < 2:
        return

    ear_arr = np.array(ear_history)
    lo, hi = max(0.0, ear_arr.min() - 0.02), min(0.55, ear_arr.max() + 0.04)
    span = hi - lo if hi > lo else 0.1

    def _y(val: float) -> int:
        norm = (val - lo) / span
        return int(gy + gh - norm * (gh - 10))

    # Threshold line
    ty = _y(threshold)
    cv2.line(frame, (gx, ty), (gx + gw, ty), CLR_THRESH, 1, cv2.LINE_AA)
    _text(frame, f"THR {threshold:.3f}", gx + gw - 90, ty - 14, CLR_THRESH, scale=0.38)

    # EAR waveform
    pts = []
    step = gw / max(len(ear_arr) - 1, 1)
    for i, val in enumerate(ear_arr):
        pts.append((int(gx + i * step), _y(val)))

    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i + 1], CLR_GRAPH, 2, cv2.LINE_AA)

    # Current EAR dot
    cv2.circle(frame, pts[-1], 5, CLR_ACCENT, -1)

    # Label
    _text(frame, "EAR WAVEFORM", gx + 8, gy + 6, CLR_DIM, scale=0.38)


def _draw_landmark_overlay(
    frame: np.ndarray,
    landmarks,
    img_w: int,
    img_h: int,
) -> None:
    """Draw subtle eye-region landmark dots."""
    from detector import _LEFT_EYE_IDX, _RIGHT_EYE_IDX  # re-use constants

    for idx in (*_LEFT_EYE_IDX, *_RIGHT_EYE_IDX):
        lm = landmarks[idx]
        cx, cy = int(lm.x * img_w), int(lm.y * img_h)
        cv2.circle(frame, (cx, cy), 2, CLR_ACCENT, -1, cv2.LINE_AA)


def _draw_event_flash(
    frame: np.ndarray,
    event: BlinkEvent,
    event_time: Optional[float],
    fh: int, fw: int,
) -> None:
    """Brief full-frame flash when a long blink is detected."""
    if event != BlinkEvent.LONG or event_time is None:
        return
    elapsed = time.monotonic() - event_time
    if elapsed > 0.35:
        return
    alpha = max(0.0, 0.45 * (1.0 - elapsed / 0.35))
    overlay = frame.copy()
    overlay[:] = CLR_ACCENT
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    _text(frame, "SELECTION", fw // 2 - 70, fh // 2 - 20,
          (255, 255, 255), scale=1.2, thickness=3, font=FONT_BOLD)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def run() -> None:
    # --- MediaPipe setup (version-safe wrapper) ---
    face_mesh = _FaceMeshWrapper()

    # --- Camera ---
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        sys.exit(1)

    # --- Core objects ---
    tts = AsyncTTS(rate=165)
    detector = BlinkDetector(buffer_size=7, calibration_duration=5.0)

    phrase_index: int = 0
    last_phrase: str = ""
    last_event_time: Optional[float] = None
    ear_history: Deque[float] = deque(maxlen=EAR_HISTORY)

    # Wire callbacks
    def _on_long_blink() -> None:
        nonlocal phrase_index, last_phrase, last_event_time
        phrase = PHRASES[phrase_index % len(PHRASES)]
        phrase_index += 1
        last_phrase = phrase
        last_event_time = time.monotonic()
        tts.speak(phrase)

    detector.on_long_blink = _on_long_blink

    # --- FPS tracking ---
    fps_deque: Deque[float] = deque(maxlen=30)
    prev_time = time.monotonic()

    print("[INFO] Eye-Blink-to-Speech starting …  Press R to recalibrate, Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Empty frame received.")
            continue

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        # --- FPS ---
        now = time.monotonic()
        fps_deque.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        fps = float(np.mean(fps_deque))

        # --- MediaPipe inference ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = face_mesh.process(rgb)
        rgb.flags.writeable = True

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            ear = detector.process_frame(landmarks, fw, fh)
            ear_history.append(ear)
            _draw_landmark_overlay(frame, landmarks, fw, fh)
        else:
            # No face detected — show subtle warning
            _text(frame, "NO FACE DETECTED", fw // 2 - 120, fh // 2,
                  CLR_DANGER, scale=0.80, thickness=2)

        # --- HUD layers (back → front) ---
        _draw_ear_graph(frame, ear_history, detector.calibration.threshold, fh, fw)
        _draw_status_panel(frame, detector, last_phrase, phrase_index, fps)
        _draw_calibration_bar(frame, detector, fh, fw)
        _draw_event_flash(frame, detector.last_event, last_event_time, fh, fw)

        # --- Keyboard ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            detector.reset_calibration()
            print("[INFO] Calibration restarted.")

        cv2.imshow("Eye-Blink-to-Speech  |  Q=Quit  R=Recalibrate", frame)

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()
    print("[INFO] Application terminated.")


if __name__ == "__main__":
    run()
