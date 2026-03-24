"""
Heart CV-gnal — Main Application Runner
=========================================
Orchestrates the real-time webcam loop, MediaPipe Holistic inference,
feature extraction, affection scoring, and overlay rendering.

Run via::

    PYTHONPATH=src python apps/run_heart_cvgnal.py

Controls
--------
Q / ESC — quit
"""

from __future__ import annotations

import time
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

from ..pipelines.vision.affection_engine import AffectionEngine, AffectionOutput
from ..pipelines.vision.feature_extractor import FeatureExtractor, FrameFeatures

# ---------------------------------------------------------------------------
# UI constants
# ---------------------------------------------------------------------------
_PANEL_W   = 250          # right-side score panel width (px)
_PANEL_PAD = 10           # outer margin

_EVENT_DISPLAY_SECS = 2.5  # how long an event banner stays visible

# Score-band colour stops (BGR)  — maps score 0→100 to blue→green→red
_COLOUR_STOPS = [
    (0,   (200,  70,  30)),   # 0   — cool blue
    (40,  ( 60, 200,  80)),   # 40  — lime green
    (65,  ( 30, 210, 200)),   # 65  — amber
    (85,  ( 30,  80, 240)),   # 85  — orange-red
    (100, ( 20,  30, 255)),   # 100 — hot red
]

# Mood labels for the score
_MOOD_LABELS = [
    (0,  "Not Feeling It"),
    (20, "Mildly Curious"),
    (40, "Warming Up"),
    (60, "Interested"),
    (75, "Smitten"),
    (88, "Head Over Heels"),
]


# ---------------------------------------------------------------------------
# Helper: score → BGR colour (linear interpolation between stops)
# ---------------------------------------------------------------------------

def _score_color(score: float) -> tuple[int, int, int]:
    score = max(0.0, min(100.0, score))
    for i in range(len(_COLOUR_STOPS) - 1):
        s0, c0 = _COLOUR_STOPS[i]
        s1, c1 = _COLOUR_STOPS[i + 1]
        if s0 <= score <= s1:
            t = (score - s0) / (s1 - s0)
            return tuple(int(c0[j] + t * (c1[j] - c0[j])) for j in range(3))  # type: ignore[return-value]
    return _COLOUR_STOPS[-1][1]


def _mood_label(score: float) -> str:
    label = _MOOD_LABELS[0][1]
    for threshold, text in _MOOD_LABELS:
        if score >= threshold:
            label = text
    return label


# ---------------------------------------------------------------------------
# Application class
# ---------------------------------------------------------------------------

class HeartCVgnalApp:
    """
    Top-level application.  Instantiate once; call ``run()`` to start.
    """

    def __init__(self, camera_index: int = 0) -> None:
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {camera_index}.  "
                "Check macOS System Settings → Privacy & Security → Camera."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

        self._extractor = FeatureExtractor()
        self._engine    = AffectionEngine()

        self._mp_holistic = mp.solutions.holistic
        self._mp_drawing  = mp.solutions.drawing_utils

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        print("=" * 55)
        print("  Heart CV-gnal  |  Press Q or ESC to quit")
        print("=" * 55)

        holistic_cfg = dict(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        with self._mp_holistic.Holistic(**holistic_cfg) as holistic:
            while self._cap.isOpened():
                ret, frame = self._cap.read()
                if not ret:
                    print("[WARN] Frame read failed — retrying …")
                    continue

                # Mirror horizontally for natural selfie view.
                # MediaPipe processes this flipped frame, so its left/right
                # labels align with the displayed image.
                frame = cv2.flip(frame, 1)

                # ── Inference ─────────────────────────────────────────────
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = holistic.process(rgb)
                rgb.flags.writeable = True

                # ── Feature extraction ─────────────────────────────────────
                features = self._extractor.extract(results)

                # ── Score update ──────────────────────────────────────────
                ts     = time.monotonic()
                output = self._engine.update(features, ts)

                # ── Optional: draw MediaPipe skeleton (subtle) ─────────────
                if results.pose_landmarks:
                    self._mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        self._mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=self._mp_drawing.DrawingSpec(
                            color=(60, 60, 80), thickness=1, circle_radius=2
                        ),
                        connection_drawing_spec=self._mp_drawing.DrawingSpec(
                            color=(60, 60, 80), thickness=1
                        ),
                    )

                # ── Render overlay ────────────────────────────────────────
                if not output.calibrated:
                    frame = self._render_calibrating(frame, output)
                else:
                    frame = self._render_score_panel(frame, features, output)
                    frame = self._render_event_banner(frame, output, ts)
                    frame = self._render_status_dots(frame, features)
                    frame = self._render_active_signals(frame, output)

                # ── FPS counter ───────────────────────────────────────────
                self._draw_fps(frame, ts)

                cv2.imshow("Heart CV-gnal", frame)

                key = cv2.waitKey(5) & 0xFF
                if key in (ord('q'), 27):  # Q or ESC
                    break

        self._cleanup()

    # ------------------------------------------------------------------
    # Render: calibration screen
    # ------------------------------------------------------------------

    def _render_calibrating(
        self, frame: np.ndarray, output: AffectionOutput
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        # Darken the camera feed
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        cx, cy = w // 2, h // 2

        # Title
        cv2.putText(
            frame, "HEART  CV-GNAL",
            (cx - 190, cy - 90),
            cv2.FONT_HERSHEY_DUPLEX, 1.4, (100, 200, 255), 2, cv2.LINE_AA,
        )

        # Sub-title
        cv2.putText(
            frame, "Affection / Interest Analyzer",
            (cx - 175, cy - 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 180), 1, cv2.LINE_AA,
        )

        # Instruction
        cv2.putText(
            frame, "Look straight at the camera to calibrate your baseline.",
            (cx - 240, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
        )

        # Progress bar
        bar_w, bar_h = 420, 22
        bx = cx - bar_w // 2
        by = cy + 20
        fill = int(bar_w * output.calib_progress)

        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (40, 40, 50), -1)
        if fill > 0:
            cv2.rectangle(frame, (bx, by), (bx + fill, by + bar_h), (100, 200, 255), -1)
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (80, 80, 110), 1)

        pct_text = f"{int(output.calib_progress * 100)}%"
        cv2.putText(
            frame, pct_text,
            (cx - 18, by + bar_h - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA,
        )

        # Hint
        cv2.putText(
            frame, "Q / ESC to quit",
            (cx - 60, cy + 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 110), 1, cv2.LINE_AA,
        )

        return frame

    # ------------------------------------------------------------------
    # Render: right-side score panel
    # ------------------------------------------------------------------

    def _render_score_panel(
        self,
        frame: np.ndarray,
        features: FrameFeatures,
        output: AffectionOutput,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        pw  = _PANEL_W
        px  = w - pw - _PANEL_PAD
        py  = _PANEL_PAD
        ph  = h - _PANEL_PAD * 2

        # Semi-transparent panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (px - 5, py), (px + pw, py + ph), (12, 12, 22), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (px - 5, py), (px + pw, py + ph), (55, 55, 80), 1)

        score = output.score
        color = _score_color(score)

        # ── Header ────────────────────────────────────────────────────────
        cv2.putText(
            frame, "AFFECTION  METER",
            (px + 4, py + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 200), 1, cv2.LINE_AA,
        )

        # ── Big score number ───────────────────────────────────────────────
        score_str = f"{int(score):3d}"
        cv2.putText(
            frame, score_str,
            (px + 10, py + 85),
            cv2.FONT_HERSHEY_DUPLEX, 2.8, color, 3, cv2.LINE_AA,
        )
        cv2.putText(
            frame, "/ 100",
            (px + 158, py + 85),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (110, 110, 130), 1, cv2.LINE_AA,
        )

        # ── Mood label ────────────────────────────────────────────────────
        cv2.putText(
            frame, _mood_label(score),
            (px + 4, py + 107),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA,
        )

        # ── Gradient gauge bar ────────────────────────────────────────────
        gx   = px + 4
        gy   = py + 118
        gw   = pw - 8
        gh   = 20
        fill = int(gw * score / 100.0)

        cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (35, 35, 45), -1)
        # Render gradient by drawing vertical slices
        for i in range(fill):
            t     = (i / max(1, gw - 1)) * 100.0
            pixel = _score_color(t)
            cv2.line(frame, (gx + i, gy + 1), (gx + i, gy + gh - 1), pixel, 1)
        cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (75, 75, 100), 1)

        # ── Divider ───────────────────────────────────────────────────────
        dy0 = py + 148
        cv2.line(frame, (px, dy0), (px + pw, dy0), (50, 50, 75), 1)

        # ── Metric readouts ───────────────────────────────────────────────
        def _row(label: str, val_str: str, sub: str, y: int,
                 val_color=(200, 200, 210)) -> None:
            cv2.putText(frame, label,   (px + 4,   y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (110, 110, 140), 1, cv2.LINE_AA)
            cv2.putText(frame, val_str, (px + 82,  y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, val_color,        1, cv2.LINE_AA)
            if sub:
                cv2.putText(frame, sub, (px + 130, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (70, 70, 95), 1, cv2.LINE_AA)

        if features.face_detected:
            _row("Mouth ratio",
                 f"{features.mouth_ratio:.3f}",
                 f"(b {output.baseline_mouth:.3f})",
                 py + 168)
            _row("Eye ratio",
                 f"{features.eye_ratio:.3f}",
                 f"(b {output.baseline_eye:.3f})",
                 py + 188)
            _row("Roll angle",
                 f"{features.roll_angle_deg:+.1f} deg",
                 "",
                 py + 208)
        else:
            cv2.putText(frame, "No face detected", (px + 4, py + 188),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 120), 1, cv2.LINE_AA)

        if features.pose_detected:
            _row("Shoulder",
                 f"{features.shoulder_ratio:.3f}",
                 f"(b {output.baseline_shoulder:.3f})",
                 py + 230)
            arm_val   = "CROSSED" if features.wrists_crossed else "free"
            arm_color = (50, 50, 220) if features.wrists_crossed else (80, 160, 80)
            _row("Arms",
                 arm_val, "",
                 py + 250, val_color=arm_color)
        else:
            cv2.putText(frame, "No pose detected", (px + 4, py + 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 120), 1, cv2.LINE_AA)

        # ── Divider ───────────────────────────────────────────────────────
        dy1 = py + 268
        cv2.line(frame, (px, dy1), (px + pw, dy1), (50, 50, 75), 1)

        # ── Event log (last 4 events) ─────────────────────────────────────
        cv2.putText(frame, "Recent events", (px + 4, dy1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (90, 90, 120), 1, cv2.LINE_AA)
        for i, evt in enumerate(output.event_log[:4]):
            alpha = 1.0 - i * 0.22
            col   = tuple(int(c * alpha) for c in (180, 180, 220))  # type: ignore[misc]
            cv2.putText(frame, evt, (px + 4, dy1 + 34 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    # Render: event banner (top-centre, fades after _EVENT_DISPLAY_SECS)
    # ------------------------------------------------------------------

    def _render_event_banner(
        self,
        frame: np.ndarray,
        output: AffectionOutput,
        ts: float,
    ) -> np.ndarray:
        age = ts - output.last_event_time
        if age > _EVENT_DISPLAY_SECS or not output.last_event_text:
            return frame

        # Classify event for colour
        text = output.last_event_text
        if "+" in text:
            bg  = (30,  80, 200)   # BGR: reddish-orange → positive
            fg  = (255, 255, 255)
        elif "-" in text or "Barrier" in text:
            bg  = (60,  30, 160)   # BGR: deep blue → negative
            fg  = (200, 200, 255)
        else:
            bg  = (60,  60,  60)
            fg  = (230, 230, 230)

        h, w = frame.shape[:2]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
        bx = (w - tw) // 2 - 16
        by = 18
        bw = tw + 32
        bh = th + 18

        # Pill background
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), bg, -1)
        alpha_val = max(0.0, 1.0 - age / _EVENT_DISPLAY_SECS)
        cv2.addWeighted(overlay, alpha_val * 0.88, frame, 1 - alpha_val * 0.88, 0, frame)

        # Border
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), fg, 1)

        # Text
        cv2.putText(
            frame, text,
            ((w - tw) // 2, by + th + 8),
            cv2.FONT_HERSHEY_DUPLEX, 0.75, fg, 2, cv2.LINE_AA,
        )
        return frame

    # ------------------------------------------------------------------
    # Render: detection status dots (top-left)
    # ------------------------------------------------------------------

    def _render_status_dots(
        self, frame: np.ndarray, features: FrameFeatures
    ) -> np.ndarray:
        items = [
            ("Face", features.face_detected),
            ("Pose", features.pose_detected),
        ]
        for i, (label, ok) in enumerate(items):
            color = (50, 200, 80) if ok else (50, 50, 160)
            x = _PANEL_PAD + i * 90
            y = _PANEL_PAD + 14
            cv2.circle(frame, (x, y), 6, color, -1)
            cv2.putText(frame, label, (x + 10, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 180), 1, cv2.LINE_AA)
        return frame

    # ------------------------------------------------------------------
    # Render: active-signal badges (below status dots)
    # ------------------------------------------------------------------

    def _render_active_signals(
        self, frame: np.ndarray, output: AffectionOutput
    ) -> np.ndarray:
        signals: list[tuple[str, tuple[int, int, int]]] = []
        if output.is_leaning:
            signals.append(("LEAN IN",  (50, 200, 100)))
        if output.is_syncing:
            signals.append(("SYNC",     (50, 220, 220)))
        if output.is_barrier:
            signals.append(("BARRIER",  (50,  50, 220)))

        for i, (sig, col) in enumerate(signals):
            cv2.putText(
                frame, f">> {sig}",
                (_PANEL_PAD, _PANEL_PAD + 40 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA,
            )
        return frame

    # ------------------------------------------------------------------
    # Render: FPS counter (bottom-left)
    # ------------------------------------------------------------------

    _prev_ts: float = 0.0

    def _draw_fps(self, frame: np.ndarray, ts: float) -> None:
        dt = ts - self._prev_ts
        if dt > 0:
            fps = 1.0 / dt
            h   = frame.shape[0]
            cv2.putText(
                frame, f"{fps:.0f} fps",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 70, 90), 1, cv2.LINE_AA,
            )
        self._prev_ts = ts

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        cap = getattr(self, "_cap", None)
        if cap and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
