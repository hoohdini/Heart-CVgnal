"""
Heart CV-gnal — Affection Score Engine
========================================
Rule-based evaluation of four psychologically-grounded signals.

  Signal              | Direction | Trigger
  --------------------|-----------|--------------------------------------------
  Duchenne Smile      | +5 pts    | mouth_ratio spikes AND eye_ratio drops
  Leaning In          | +2 pts/s  | shoulder_ratio > 1.15 × baseline
  Synchrony           | +1.5 pts/s| sustained head tilt (5-25°) OR nodding
  Barrier Signal      | -3 pts/s  | both wrists cross opposite shoulders > 3 s

Global score initialises at 50 and is clamped to [0, 100].

Usage
-----
    engine = AffectionEngine()
    ...
    output = engine.update(frame_features, timestamp_seconds)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .feature_extractor import FrameFeatures


# ---------------------------------------------------------------------------
# Output data model (consumed by the runner / UI layer)
# ---------------------------------------------------------------------------

@dataclass
class AffectionOutput:
    """Snapshot of the engine's state returned after each frame update."""

    score: float
    """Current affection score, always in [0, 100]."""

    # ── Calibration ──────────────────────────────────────────────────────
    calibrated: bool
    calib_progress: float
    """Fraction of the calibration window elapsed, 0.0 → 1.0."""

    # ── Baselines (published so UI can show deltas) ───────────────────
    baseline_mouth:    float
    baseline_eye:      float
    baseline_shoulder: float

    # ── Active-signal flags ───────────────────────────────────────────
    is_leaning:  bool
    """Subject is currently leaning in."""
    is_barrier:  bool
    """Barrier signal is actively penalising (3-second threshold exceeded)."""
    is_syncing:  bool
    """Synchrony (head tilt or nodding) is contributing to score."""

    # ── Event notification ────────────────────────────────────────────
    last_event_text: str
    last_event_time: float
    """Wall-clock timestamp of the most recent event (for fade-out calc)."""

    # ── Recent event log ──────────────────────────────────────────────
    event_log: List[str]
    """Rolling list of the last N event strings (newest first)."""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AffectionEngine:
    """
    Stateful rule-based affection scorer.

    Maintain one instance per session and call ``update()`` on every frame.
    """

    # ── Calibration ──────────────────────────────────────────────────────
    CALIB_SECS: float = 3.0

    # ── Rule 1: Duchenne Smile ────────────────────────────────────────────
    MOUTH_SPIKE_FACTOR: float = 1.12   # mouth must be ≥12 % wider than baseline
    EYE_SQUINT_FACTOR:  float = 0.90   # eyes must be ≤10 % narrower than baseline
    DUCHENNE_DELTA:     float = 5.0
    DUCHENNE_COOLDOWN:  float = 2.5    # seconds between awards

    # ── Rule 2: Leaning In ───────────────────────────────────────────────
    LEAN_FACTOR:        float = 1.15   # 15 % wider shoulders than baseline
    LEAN_DELTA_PER_SEC: float = 2.0

    # ── Rule 3: Synchrony ────────────────────────────────────────────────
    SYNC_TILT_MIN_DEG:  float = 5.0    # degrees
    SYNC_TILT_MAX_DEG:  float = 25.0   # degrees
    SYNC_TILT_WARMUP:   float = 2.0    # seconds sustained before awarding
    SYNC_DELTA_PER_SEC: float = 1.5
    NODDING_STD:        float = 0.007  # nose-y std threshold (2-sec window)

    # ── Rule 4: Barrier Signal ───────────────────────────────────────────
    BARRIER_WAIT_SECS:    float = 3.0
    BARRIER_DELTA_PER_SEC: float = -3.0

    # ── Score bounds ─────────────────────────────────────────────────────
    SCORE_MIN: float = 0.0
    SCORE_MAX: float = 100.0

    # ── Misc ─────────────────────────────────────────────────────────────
    EVENT_LOG_SIZE: int = 8
    _FPS: float = 30.0   # assumed frame-rate for per-frame delta conversion

    def __init__(self) -> None:
        # ── Calibration state ────────────────────────────────────────────
        self._start_time:  Optional[float] = None
        self._calib_buf:   list[FrameFeatures] = []

        self._calibrated:         bool  = False
        self._baseline_mouth:     float = 0.40
        self._baseline_eye:       float = 0.06
        self._baseline_shoulder:  float = 0.30

        # ── Score ────────────────────────────────────────────────────────
        self._score: float = 50.0

        # ── Rule-1 (Duchenne) ────────────────────────────────────────────
        self._duchenne_last: float = 0.0

        # ── Rule-2 (Lean-in) ─────────────────────────────────────────────
        self._was_leaning: bool = False

        # ── Rule-3 (Synchrony) ───────────────────────────────────────────
        self._sync_tilt_start: Optional[float] = None
        self._nose_y_buf: deque[float] = deque(maxlen=int(self._FPS * 2))  # 2-sec
        self._is_syncing: bool = False

        # ── Rule-4 (Barrier) ─────────────────────────────────────────────
        self._barrier_start:   Optional[float] = None
        self._barrier_alerted: bool = False
        self._is_barrier:      bool = False

        # ── Events ───────────────────────────────────────────────────────
        self._last_event_text: str   = ""
        self._last_event_time: float = 0.0
        self._event_log: deque[str]  = deque(maxlen=self.EVENT_LOG_SIZE)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, features: FrameFeatures, ts: float) -> AffectionOutput:
        """
        Process one frame of features and return the current engine state.

        Args:
            features : FrameFeatures from FeatureExtractor.extract()
            ts       : current wall-clock time (``time.monotonic()`` recommended)
        Returns:
            AffectionOutput snapshot
        """
        if self._start_time is None:
            self._start_time = ts

        # ── Calibration phase ─────────────────────────────────────────────
        if not self._calibrated:
            self._calibrate(features, ts)
            return self._build_output(ts)

        # ── Evaluation phase ──────────────────────────────────────────────
        if features.face_detected:
            self._rule_duchenne(features, ts)
            self._rule_synchrony(features, ts)

        if features.pose_detected:
            self._rule_lean_in(features, ts)
            self._rule_barrier(features, ts)
        else:
            # Skeleton lost — reset pose-dependent timers
            self._barrier_start  = None
            self._is_barrier     = False
            self._was_leaning    = False

        # Clamp score
        self._score = max(self.SCORE_MIN, min(self.SCORE_MAX, self._score))

        return self._build_output(ts)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _calibrate(self, features: FrameFeatures, ts: float) -> None:
        if features.face_detected and features.pose_detected:
            self._calib_buf.append(features)

        elapsed = ts - self._start_time  # type: ignore[operator]
        if elapsed >= self.CALIB_SECS:
            buf = self._calib_buf
            if buf:
                self._baseline_mouth    = float(np.mean([f.mouth_ratio    for f in buf]))
                self._baseline_eye      = float(np.mean([f.eye_ratio      for f in buf]))
                self._baseline_shoulder = float(np.mean([f.shoulder_ratio for f in buf]))
            # If nothing was captured (no face/pose during entire window),
            # keep the hardcoded sensible defaults.
            self._calibrated = True
            self._calib_buf.clear()
            self._fire_event("Baseline locked — go!", ts)

    # ------------------------------------------------------------------
    # Rule 1 — Duchenne Smile
    # ------------------------------------------------------------------

    def _rule_duchenne(self, features: FrameFeatures, ts: float) -> None:
        """
        Award +5 when mouth widens AND eyes narrow simultaneously.
        The combination of a wide smile (zygomaticus major) + eyelid squint
        (orbicularis oculi) is the hallmark of a genuine (Duchenne) smile.
        """
        mouth_up   = features.mouth_ratio > self._baseline_mouth * self.MOUTH_SPIKE_FACTOR
        eyes_squint = features.eye_ratio  < self._baseline_eye   * self.EYE_SQUINT_FACTOR

        if mouth_up and eyes_squint:
            if ts - self._duchenne_last >= self.DUCHENNE_COOLDOWN:
                self._score          += self.DUCHENNE_DELTA
                self._duchenne_last   = ts
                self._fire_event("Duchenne Smile!  +5", ts)

    # ------------------------------------------------------------------
    # Rule 2 — Leaning In
    # ------------------------------------------------------------------

    def _rule_lean_in(self, features: FrameFeatures, ts: float) -> None:
        """
        Award +2 pts/s while the subject's shoulders span > 1.15× baseline.
        In normalised coordinates a wider shoulder span means the subject has
        moved physically closer to the camera — a classic approach signal.
        """
        leaning = features.shoulder_ratio > self._baseline_shoulder * self.LEAN_FACTOR

        if leaning:
            self._score += self.LEAN_DELTA_PER_SEC / self._FPS
            if not self._was_leaning:
                self._fire_event("Leaning In!  +2/s", ts)
        self._was_leaning = leaning

    # ------------------------------------------------------------------
    # Rule 3 — Synchrony
    # ------------------------------------------------------------------

    def _rule_synchrony(self, features: FrameFeatures, ts: float) -> None:
        """
        Award +1.5 pts/s for either:
          a) Sustained optimal head tilt (roll angle 5–25°)  — engaged listening
          b) Head nodding detected via high nose-y variance over 2 seconds
        """
        roll = abs(features.roll_angle_deg)
        tilt_ok = self.SYNC_TILT_MIN_DEG <= roll <= self.SYNC_TILT_MAX_DEG

        # (a) Tilt-based synchrony — requires 2-second warm-up
        if tilt_ok:
            if self._sync_tilt_start is None:
                self._sync_tilt_start = ts
            sustained = (ts - self._sync_tilt_start) >= self.SYNC_TILT_WARMUP
        else:
            self._sync_tilt_start = None
            sustained = False

        # (b) Nodding — nose-y variance over last 2 seconds
        self._nose_y_buf.append(features.nose_y)
        nodding = (
            len(self._nose_y_buf) == self._nose_y_buf.maxlen
            and float(np.std(self._nose_y_buf)) > self.NODDING_STD
        )

        active = sustained or nodding
        if active:
            self._score += self.SYNC_DELTA_PER_SEC / self._FPS
        self._is_syncing = active

    # ------------------------------------------------------------------
    # Rule 4 — Barrier Signal
    # ------------------------------------------------------------------

    def _rule_barrier(self, features: FrameFeatures, ts: float) -> None:
        """
        Penalise -3 pts/s after both wrists have remained crossed over the
        opposite shoulders for more than 3 seconds (defensive arm-cross).
        """
        if features.wrists_crossed:
            if self._barrier_start is None:
                self._barrier_start   = ts
                self._barrier_alerted = False

            elapsed = ts - self._barrier_start
            if elapsed >= self.BARRIER_WAIT_SECS:
                self._score      += self.BARRIER_DELTA_PER_SEC / self._FPS
                self._is_barrier  = True
                if not self._barrier_alerted:
                    self._fire_event("Barrier Signal!  -3/s", ts)
                    self._barrier_alerted = True
            else:
                self._is_barrier = False
        else:
            self._barrier_start   = None
            self._barrier_alerted = False
            self._is_barrier      = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fire_event(self, text: str, ts: float) -> None:
        self._last_event_text = text
        self._last_event_time = ts
        self._event_log.appendleft(text)

    def _build_output(self, ts: float) -> AffectionOutput:
        calib_progress = 0.0
        if not self._calibrated and self._start_time is not None:
            calib_progress = min(1.0, (ts - self._start_time) / self.CALIB_SECS)

        return AffectionOutput(
            score              = self._score,
            calibrated         = self._calibrated,
            calib_progress     = calib_progress,
            baseline_mouth     = self._baseline_mouth,
            baseline_eye       = self._baseline_eye,
            baseline_shoulder  = self._baseline_shoulder,
            is_leaning         = self._was_leaning,
            is_barrier         = self._is_barrier,
            is_syncing         = self._is_syncing,
            last_event_text    = self._last_event_text,
            last_event_time    = self._last_event_time,
            event_log          = list(self._event_log),
        )
