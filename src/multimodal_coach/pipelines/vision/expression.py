"""
Facial Expression Analyzer — MediaPipe FaceMesh based
======================================================
Operates on the face_landmarks returned by MediaPipe Holistic (468 points)
or a standalone FaceMesh result.  No additional models or dependencies.

Metrics produced per frame
--------------------------
smile_score     : 0 (neutral/frown) → 1 (clear smile)
brow_tension    : 0 (relaxed brows) → 1 (heavily furrowed)
eye_openness    : 0 (eyes nearly closed) → 1 (wide open)
confidence_score: weighted combination — "confident speaker" composite
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# FaceMesh landmark indices (MediaPipe 468-point canonical model)
# ---------------------------------------------------------------------------

# Mouth
_MOUTH_LEFT  = 61    # left corner
_MOUTH_RIGHT = 291   # right corner
_MOUTH_TOP   = 13    # upper inner lip centre
_MOUTH_BOTTOM = 14   # lower inner lip centre

# Face geometry references
_NOSE_TIP = 4
_CHIN     = 152

# Inner-most brow points (closest to glabella)
_L_BROW_INNER = 105
_R_BROW_INNER = 334

# Eye — 6-point EAR layout: [outer, upper1, upper2, inner, lower1, lower2]
# (These match the pilot-test facemesh_realtime_pilot_test.py ordering.)
_LEFT_EYE  = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Eye upper-lid point (for brow-to-eye gap)
_L_EYE_TOP = 159
_R_EYE_TOP = 386


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ExpressionMetrics:
    """
    Normalised per-frame expression measurements.
    All values in [0.0, 1.0].
    """
    smile_score: float       # 0 = neutral/frown, 1 = clear smile
    brow_tension: float      # 0 = relaxed,       1 = heavily furrowed
    eye_openness: float      # 0 = nearly closed,  1 = wide open
    confidence_score: float  # overall "confident speaker" composite


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class ExpressionAnalyzer:
    """
    Lightweight facial expression analyser.

    Usage::

        analyzer = ExpressionAnalyzer()
        # holistic_results.face_landmarks is a NormalizedLandmarkList or None
        metrics = analyzer.analyze(holistic_results.face_landmarks)
        if metrics:
            print(f"smile={metrics.smile_score:.2f}  confidence={metrics.confidence_score:.2f}")

    All computations are normalised against face geometry so that
    camera distance does not affect scores.
    """

    # Calibration clip ranges (raw ratio → 0–1)
    _SMILE_RAW_MIN: float =  -0.02   # slight frown
    _SMILE_RAW_MAX: float =   0.07   # clear smile

    _BROW_GAP_MIN:  float =   0.04   # heavily furrowed
    _BROW_GAP_MAX:  float =   0.16   # relaxed / raised

    _EAR_MIN:       float =   0.10   # nearly closed
    _EAR_MAX:       float =   0.32   # wide open

    def analyze(self, face_landmarks) -> Optional[ExpressionMetrics]:
        """
        Args:
            face_landmarks: MediaPipe NormalizedLandmarkList from Holistic
                            (``results.face_landmarks``) — or ``None``.
        Returns:
            ExpressionMetrics on success, None if landmarks unavailable.
        """
        if face_landmarks is None:
            return None

        lm = face_landmarks.landmark
        if len(lm) < 400:          # guard against truncated results
            return None

        smile    = self._compute_smile(lm)
        brow_t   = self._compute_brow_tension(lm)
        eye_open = self._compute_eye_openness(lm)
        conf     = self._compute_confidence(smile, brow_t, eye_open)

        return ExpressionMetrics(
            smile_score=smile,
            brow_tension=brow_t,
            eye_openness=eye_open,
            confidence_score=conf,
        )

    # ------------------------------------------------------------------
    # Private metric computations
    # ------------------------------------------------------------------

    def _compute_smile(self, lm) -> float:
        """
        Measures how much mouth corners are raised above the mouth centre.
        In normalised coords (y=0 is top), smile → corners.y < centre.y.

        Returns 0 (frown / neutral) → 1 (clear smile).
        """
        left   = lm[_MOUTH_LEFT]
        right  = lm[_MOUTH_RIGHT]
        top    = lm[_MOUTH_TOP]
        bottom = lm[_MOUTH_BOTTOM]
        nose   = lm[_NOSE_TIP]
        chin   = lm[_CHIN]

        centre_y = (top.y + bottom.y) / 2.0
        corner_y = (left.y + right.y) / 2.0
        # Positive when corners sit ABOVE (smaller y) the lip centre → smile
        raw      = centre_y - corner_y

        face_h = abs(chin.y - nose.y) or 1e-4
        ratio  = raw / face_h

        return float(np.clip(
            (ratio - self._SMILE_RAW_MIN) /
            (self._SMILE_RAW_MAX - self._SMILE_RAW_MIN),
            0.0, 1.0,
        ))

    def _compute_brow_tension(self, lm) -> float:
        """
        Measures brow proximity to eyes.
        Small brow-to-eye gap → furrowed → high tension.

        Returns 0 (relaxed) → 1 (heavily furrowed).
        """
        l_brow   = lm[_L_BROW_INNER]
        r_brow   = lm[_R_BROW_INNER]
        l_eye_t  = lm[_L_EYE_TOP]
        r_eye_t  = lm[_R_EYE_TOP]
        nose     = lm[_NOSE_TIP]
        chin     = lm[_CHIN]

        face_h = abs(chin.y - nose.y) or 1e-4

        # Positive gap means brow sits above the eye lid
        l_gap = (l_eye_t.y - l_brow.y) / face_h
        r_gap = (r_eye_t.y - r_brow.y) / face_h
        gap   = (l_gap + r_gap) / 2.0

        # Normalise: high gap (relaxed) → low tension; invert
        normalised = np.clip(
            (gap - self._BROW_GAP_MIN) / (self._BROW_GAP_MAX - self._BROW_GAP_MIN),
            0.0, 1.0,
        )
        return float(1.0 - normalised)

    def _compute_eye_openness(self, lm) -> float:
        """
        Eye Aspect Ratio (EAR) averaged across both eyes.
        Uses the 6-point formula from the dlib / FaceMesh convention.

        Returns 0 (nearly closed) → 1 (wide open).
        """
        ear_l = self._ear(lm, _LEFT_EYE)
        ear_r = self._ear(lm, _RIGHT_EYE)
        raw   = (ear_l + ear_r) / 2.0

        return float(np.clip(
            (raw - self._EAR_MIN) / (self._EAR_MAX - self._EAR_MIN),
            0.0, 1.0,
        ))

    def _compute_confidence(
        self,
        smile: float,
        brow_tension: float,
        eye_openness: float,
    ) -> float:
        """
        Weighted combination into a "confident speaker" score.
        Smile and open eyes are positive signals;
        furrowed brows (tension) is a negative signal.
        """
        return float(np.clip(
            0.40 * smile + 0.35 * eye_openness + 0.25 * (1.0 - brow_tension),
            0.0, 1.0,
        ))

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ear(lm, eye_idx: list) -> float:
        """
        6-point Eye Aspect Ratio:
            EAR = (||p1–p5|| + ||p2–p4||) / (2 * ||p0–p3||)

        eye_idx order: [outer, upper1, upper2, inner, lower1, lower2]
        """
        p = [lm[i] for i in eye_idx]

        def d(a, b):
            return np.hypot(a.x - b.x, a.y - b.y)

        v1 = d(p[1], p[5])
        v2 = d(p[2], p[4])
        h  = d(p[0], p[3])
        return (v1 + v2) / (2.0 * h) if h > 1e-6 else 0.0
