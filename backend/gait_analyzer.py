"""
gait_analyzer.py
Neuro-motor biomarker extraction via MediaPipe Pose Landmarker.

Analyses:
  - Posture: cervical / thoracic / pelvic alignment
  - Gait:    stride, cadence, symmetry, balance, velocity
  - Tremor:  frequency + amplitude via landmark jitter analysis

No cv2.imshow or VideoCapture — cloud-safe.
"""

from __future__ import annotations

import collections
import math
import time
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from scipy import signal as sp_signal


# ─────────────────────────────────────────────
#  MediaPipe Pose setup
# ─────────────────────────────────────────────

_mp_pose = mp.solutions.pose

POSE_MODEL = _mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,         # 0=lite, 1=full, 2=heavy; 1 balances speed/accuracy
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# MediaPipe Pose landmark indices
LM_NOSE          = 0
LM_LEFT_EYE      = 2
LM_RIGHT_EYE     = 5
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_ELBOW    = 13
LM_RIGHT_ELBOW   = 14
LM_LEFT_WRIST    = 15
LM_RIGHT_WRIST   = 16
LM_LEFT_HIP      = 23
LM_RIGHT_HIP     = 24
LM_LEFT_KNEE     = 25
LM_RIGHT_KNEE    = 26
LM_LEFT_ANKLE    = 27
LM_RIGHT_ANKLE   = 28
LM_LEFT_HEEL     = 29
LM_RIGHT_HEEL    = 30
LM_LEFT_FOOT_INDEX  = 31
LM_RIGHT_FOOT_INDEX = 32


# ─────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────

def _dist_2d(p1, p2, h, w) -> float:
    """Euclidean distance between two normalised landmarks in pixel space."""
    return math.sqrt(((p1.x - p2.x) * w) ** 2 + ((p1.y - p2.y) * h) ** 2)


def _angle_3pts(a, b, c, h, w) -> float:
    """
    Angle at point B formed by segments BA and BC (degrees).
    Used for joint angle calculation.
    """
    ax, ay = (a.x - b.x) * w, (a.y - b.y) * h
    cx, cy = (c.x - b.x) * w, (c.y - b.y) * h
    dot = ax * cx + ay * cy
    mag = math.sqrt(ax**2 + ay**2) * math.sqrt(cx**2 + cy**2)
    if mag < 1e-6:
        return 0.0
    return math.degrees(math.acos(np.clip(dot / mag, -1.0, 1.0)))


def _midpoint(p1, p2):
    """Return a pseudo-landmark at the midpoint of two landmarks."""
    class _Mid:
        x = (p1.x + p2.x) / 2
        y = (p1.y + p2.y) / 2
        z = (p1.z + p2.z) / 2
    return _Mid()


# ─────────────────────────────────────────────
#  Posture analysis
# ─────────────────────────────────────────────

def analyze_posture(landmarks, h: int, w: int) -> Dict:
    """
    Compute cervical, thoracic, and pelvic alignment metrics.
    """
    lm = landmarks.landmark

    # ── Head tilt ──────────────────────────────────────────────────
    # Angle of the ear-to-eye line vs. horizontal
    l_eye, r_eye = lm[LM_LEFT_EYE], lm[LM_RIGHT_EYE]
    head_tilt = math.degrees(
        math.atan2((l_eye.y - r_eye.y) * h, (l_eye.x - r_eye.x) * w)
    )

    # ── Shoulder asymmetry ─────────────────────────────────────────
    l_sh, r_sh   = lm[LM_LEFT_SHOULDER], lm[LM_RIGHT_SHOULDER]
    shoulder_asym = abs(l_sh.y - r_sh.y) * h   # pixel height difference

    # ── Forward head posture ────────────────────────────────────────
    #  Ear x-offset from shoulder midpoint
    ear_mid_x   = (lm[LM_LEFT_EYE].x + lm[LM_RIGHT_EYE].x) / 2
    sh_mid_x    = (l_sh.x + r_sh.x) / 2
    fhp_mm_px   = (ear_mid_x - sh_mid_x) * w   # positive = head forward

    # ── Spinal lateral deviation ────────────────────────────────────
    #  Mid-shoulder vs mid-hip horizontal offset
    l_hip, r_hip = lm[LM_LEFT_HIP], lm[LM_RIGHT_HIP]
    sh_mid_y    = (l_sh.y + r_sh.y) / 2
    hip_mid_y   = (l_hip.y + r_hip.y) / 2
    spinal_dev  = abs((l_sh.x + r_sh.x) / 2 - (l_hip.x + r_hip.x) / 2) * w

    # ── Pelvic tilt ─────────────────────────────────────────────────
    pelvic_tilt = math.degrees(
        math.atan2((l_hip.y - r_hip.y) * h, (l_hip.x - r_hip.x) * w)
    )

    # ── Scores (0-1; 1 = perfect alignment) ──────────────────────
    cervical_score  = float(np.clip(1.0 - abs(head_tilt) / 20.0, 0, 1))
    thoracic_score  = float(np.clip(1.0 - spinal_dev / (w * 0.05), 0, 1))
    pelvic_score    = float(np.clip(1.0 - abs(pelvic_tilt) / 15.0, 0, 1))

    return {
        "head_tilt_deg":               float(head_tilt),
        "shoulder_asymmetry_deg":      float(shoulder_asym),
        "spinal_lateral_deviation":    float(spinal_dev),
        "forward_head_posture_mm":     float(fhp_mm_px),
        "pelvic_tilt_deg":             float(pelvic_tilt),
        "cervical_score":              cervical_score,
        "thoracic_score":              thoracic_score,
        "pelvic_score":                pelvic_score,
    }


# ─────────────────────────────────────────────
#  Tremor analysis (hand landmark jitter)
# ─────────────────────────────────────────────

class TremorDetector:
    """
    Accumulates wrist landmark positions over time.
    Computes dominant tremor frequency via FFT on the position time series.

    Sliding buffer (maxlen) prevents memory leak.
    """
    BUFFER_SIZE = 300

    def __init__(self, fps: float = 30.0):
        self.fps     = fps
        self._lx_buf: collections.deque = collections.deque(maxlen=self.BUFFER_SIZE)
        self._ly_buf: collections.deque = collections.deque(maxlen=self.BUFFER_SIZE)
        self._rx_buf: collections.deque = collections.deque(maxlen=self.BUFFER_SIZE)
        self._ry_buf: collections.deque = collections.deque(maxlen=self.BUFFER_SIZE)
        self._cached: Dict = {}
        self._frame_count = 0

    def update(self, landmarks, h: int, w: int):
        lm = landmarks.landmark
        self._lx_buf.append(lm[LM_LEFT_WRIST].x  * w)
        self._ly_buf.append(lm[LM_LEFT_WRIST].y  * h)
        self._rx_buf.append(lm[LM_RIGHT_WRIST].x * w)
        self._ry_buf.append(lm[LM_RIGHT_WRIST].y * h)
        self._frame_count += 1
        if self._frame_count % 30 == 0:
            self._recompute()

    def _recompute(self):
        n = len(self._lx_buf)
        if n < 60:
            return

        def _tremor_freq(pos_buf):
            arr = np.array(pos_buf, dtype=np.float64)
            arr = sp_signal.detrend(arr)           # remove gross movement
            # High-pass at 2 Hz to isolate tremor (not voluntary movement)
            b, a = sp_signal.butter(2, 2.0 / (self.fps / 2), btype="high")
            filtered = sp_signal.filtfilt(b, a, arr)
            freqs    = np.fft.rfftfreq(len(filtered), d=1.0 / self.fps)
            power    = np.abs(np.fft.rfft(filtered)) ** 2
            # Tremor range: 2–12 Hz
            mask = (freqs >= 2.0) & (freqs <= 12.0)
            if not np.any(mask):
                return None, 0.0
            dom_freq  = float(freqs[mask][np.argmax(power[mask])])
            amplitude = float(np.sqrt(np.mean(filtered ** 2)))
            return dom_freq, amplitude

        # Combine both wrists
        freqs, amps = [], []
        for buf in [self._lx_buf, self._ly_buf, self._rx_buf, self._ry_buf]:
            f, a = _tremor_freq(buf)
            if f is not None:
                freqs.append(f)
                amps.append(a)

        if not freqs:
            return

        dominant_hz = float(np.mean(freqs))
        amplitude   = float(np.mean(amps))

        # Severity classification
        if amplitude < 1.0:
            severity = "none"
        elif amplitude < 3.0:
            severity = "mild"
        elif amplitude < 7.0:
            severity = "moderate"
        else:
            severity = "severe"

        self._cached = {
            "dominant_tremor_hz":  dominant_hz,
            "tremor_amplitude":    amplitude,
            "tremor_severity":     severity,
        }

    def get_metrics(self) -> Dict:
        return self._cached.copy() if self._cached else {
            "dominant_tremor_hz": None,
            "tremor_amplitude":   None,
            "tremor_severity":    None,
        }


# ─────────────────────────────────────────────
#  Gait analysis
# ─────────────────────────────────────────────

class GaitAnalyzer:
    """
    Tracks ankle + foot landmark positions to derive gait parameters.
    Requires the subject to walk in front of the camera.
    """
    BUFFER_SIZE = 300

    def __init__(self, fps: float = 30.0):
        self.fps  = fps
        self._la_y: collections.deque = collections.deque(maxlen=self.BUFFER_SIZE)
        self._ra_y: collections.deque = collections.deque(maxlen=self.BUFFER_SIZE)
        self._la_x: collections.deque = collections.deque(maxlen=self.BUFFER_SIZE)
        self._ra_x: collections.deque = collections.deque(maxlen=self.BUFFER_SIZE)
        self._hip_y: collections.deque = collections.deque(maxlen=self.BUFFER_SIZE)
        self._frame_count = 0
        self._cached: Dict = {}

    def update(self, landmarks, h: int, w: int):
        lm = landmarks.landmark
        self._la_y.append(lm[LM_LEFT_ANKLE].y  * h)
        self._ra_y.append(lm[LM_RIGHT_ANKLE].y * h)
        self._la_x.append(lm[LM_LEFT_ANKLE].x  * w)
        self._ra_x.append(lm[LM_RIGHT_ANKLE].x * w)
        mid_hip_y = (lm[LM_LEFT_HIP].y + lm[LM_RIGHT_HIP].y) / 2 * h
        self._hip_y.append(mid_hip_y)
        self._frame_count += 1
        if self._frame_count % 30 == 0:
            self._recompute(h, w, landmarks)

    def _recompute(self, h: int, w: int, landmarks):
        n = len(self._la_y)
        if n < 60:
            return

        la_y = np.array(self._la_y)
        ra_y = np.array(self._ra_y)
        la_x = np.array(self._la_x)
        ra_x = np.array(self._ra_x)

        # ── Step detection via vertical ankle oscillations ──────────
        def _detect_steps(ankle_y_arr):
            filtered = sp_signal.savgol_filter(ankle_y_arr, 11, 3)
            # Find local minima (foot lifts)
            peaks, _ = sp_signal.find_peaks(-filtered, distance=int(self.fps * 0.3))
            return peaks

        l_steps = _detect_steps(la_y)
        r_steps = _detect_steps(ra_y)
        total_steps = len(l_steps) + len(r_steps)

        if total_steps < 2:
            return

        duration_sec = n / self.fps
        cadence      = total_steps / duration_sec * 60.0   # steps/min

        # ── Stride length proxy (horizontal ankle travel) ─────────
        l_stride = float(np.std(la_x)) * 2.0 if len(la_x) > 1 else 0.0
        r_stride = float(np.std(ra_x)) * 2.0 if len(ra_x) > 1 else 0.0
        stride_len = (l_stride + r_stride) / 2.0

        # ── Step width (lateral ankle separation) ─────────────────
        step_width = float(np.mean(np.abs(la_x - ra_x)))

        # ── Symmetry (ratio of L vs R step intervals) ─────────────
        if len(l_steps) >= 2 and len(r_steps) >= 2:
            l_intervals = np.diff(l_steps) / self.fps
            r_intervals = np.diff(r_steps) / self.fps
            sym_ratio   = min(np.mean(l_intervals), np.mean(r_intervals)) / \
                          max(np.mean(l_intervals), np.mean(r_intervals))
            symmetry_pct = float(sym_ratio * 100.0)
        else:
            symmetry_pct = None

        # ── Balance score via hip vertical variance ───────────────
        hip_y = np.array(self._hip_y)
        hip_var = float(np.var(sp_signal.detrend(hip_y))) if len(hip_y) > 10 else 0.0
        balance_score = float(np.clip(1.0 - hip_var / (h * 0.02) ** 2, 0.0, 1.0))

        # ── Velocity (steps × stride_len / time) ──────────────────
        velocity = float(stride_len * cadence / 60.0) if cadence > 0 else None

        self._cached = {
            "stride_length_cm":       float(stride_len),
            "cadence_steps_per_min":  float(cadence),
            "gait_symmetry_pct":      symmetry_pct,
            "balance_score":          balance_score,
            "velocity_cm_per_sec":    velocity,
            "step_width_cm":          float(step_width),
        }

    def get_metrics(self) -> Dict:
        return self._cached.copy() if self._cached else {
            "stride_length_cm":      None,
            "cadence_steps_per_min": None,
            "gait_symmetry_pct":     None,
            "balance_score":         0.0,
            "velocity_cm_per_sec":   None,
            "step_width_cm":         None,
        }


# ─────────────────────────────────────────────
#  BodyAnalyzer: session-level orchestrator
# ─────────────────────────────────────────────

class BodyAnalyzer:
    """
    Processes BGR frames for full-body neuro-motor analysis.
    No camera / display access — cloud-safe.
    """

    def __init__(self, fps: float = 30.0):
        self._tremor  = TremorDetector(fps=fps)
        self._gait    = GaitAnalyzer(fps=fps)
        self._posture_history: List[Dict] = []
        self._frame_count = 0

    def process_frame(self, frame: np.ndarray, timestamp_ms: float = 0.0) -> Dict:
        h, w = frame.shape[:2]
        self._frame_count += 1

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = POSE_MODEL.process(rgb)

        if not result.pose_landmarks:
            return {"landmarks_found": False, "frames_processed": self._frame_count}

        lm = result.pose_landmarks

        # ── Posture ────────────────────────────────────────────────
        posture = analyze_posture(lm, h, w)
        self._posture_history.append(posture)
        # Keep last 300 entries to avoid unbounded growth
        if len(self._posture_history) > 300:
            self._posture_history = self._posture_history[-300:]

        # ── Tremor ────────────────────────────────────────────────
        self._tremor.update(lm, h, w)

        # ── Gait ──────────────────────────────────────────────────
        self._gait.update(lm, h, w)

        # ── Assemble output ───────────────────────────────────────
        tremor_m = self._tremor.get_metrics()
        gait_m   = self._gait.get_metrics()

        # Session-averaged posture scores
        cs = np.mean([p["cervical_score"]  for p in self._posture_history])
        ts = np.mean([p["thoracic_score"]  for p in self._posture_history])
        ps = np.mean([p["pelvic_score"]    for p in self._posture_history])

        return {
            "landmarks_found":   True,
            "frames_processed":  self._frame_count,

            # Posture (latest frame + session avg)
            "head_tilt_deg":               posture["head_tilt_deg"],
            "shoulder_asymmetry_deg":      posture["shoulder_asymmetry_deg"],
            "spinal_lateral_deviation":    posture["spinal_lateral_deviation"],
            "forward_head_posture_mm":     posture["forward_head_posture_mm"],
            "pelvic_tilt_deg":             posture["pelvic_tilt_deg"],
            "cervical_score":              float(cs),
            "thoracic_score":              float(ts),
            "pelvic_score":                float(ps),

            # Tremor
            "dominant_tremor_hz":          tremor_m.get("dominant_tremor_hz"),
            "tremor_amplitude":            tremor_m.get("tremor_amplitude"),
            "tremor_severity":             tremor_m.get("tremor_severity"),

            # Gait
            "stride_length_cm":            gait_m.get("stride_length_cm"),
            "cadence_steps_per_min":       gait_m.get("cadence_steps_per_min"),
            "gait_symmetry_pct":           gait_m.get("gait_symmetry_pct"),
            "balance_score":               gait_m.get("balance_score", 0.0),
            "velocity_cm_per_sec":         gait_m.get("velocity_cm_per_sec"),
            "step_width_cm":               gait_m.get("step_width_cm"),
        }

    def get_final_summary(self) -> Dict:
        if not self._posture_history:
            return {}
        return {
            "head_tilt_deg":               float(np.mean([p["head_tilt_deg"] for p in self._posture_history])),
            "shoulder_asymmetry_deg":      float(np.mean([p["shoulder_asymmetry_deg"] for p in self._posture_history])),
            "cervical_score":              float(np.mean([p["cervical_score"] for p in self._posture_history])),
            "thoracic_score":              float(np.mean([p["thoracic_score"] for p in self._posture_history])),
            "pelvic_score":                float(np.mean([p["pelvic_score"]   for p in self._posture_history])),
            **self._tremor.get_metrics(),
            **self._gait.get_metrics(),
        }

    def reset(self):
        self._tremor  = TremorDetector()
        self._gait    = GaitAnalyzer()
        self._posture_history.clear()
        self._frame_count = 0
