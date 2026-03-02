"""
face_analyzer.py
Main face analysis pipeline integrating:
  - MediaPipe Face Mesh (468 landmarks)
  - rPPG extraction (cardio/respiratory biomarkers)
  - EAR (Eye Aspect Ratio) blink & fatigue detection
  - Skin texture hydration proxy
  - 3D facial asymmetry analysis

No cv2.VideoCapture, cv2.imshow, or cv2.waitKey — cloud safe.
"""

from __future__ import annotations

import collections
import math
import time
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from rppg_extractor import rPPGExtractor, compute_skin_texture_metrics


# ─────────────────────────────────────────────
#  MediaPipe initialisation (module-level, reused across frames)
# ─────────────────────────────────────────────

_mp_face_mesh = mp.solutions.face_mesh

FACE_MESH = _mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,          # enables iris + detailed eye points
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ─────────────────────────────────────────────
#  EAR landmark indices (as specified in requirements)
# ─────────────────────────────────────────────
#
# Left eye:   p1=33,  p2=160, p3=158, p4=133, p5=153, p6=144
# Right eye:  p1=362, p2=385, p3=387, p4=263, p5=373, p6=380
#
# EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

EAR_LEFT_IDX  = (33,  133, 160, 158, 153, 144)
EAR_RIGHT_IDX = (362, 263, 385, 387, 373, 380)

EAR_BLINK_THRESHOLD    = 0.25
EAR_BLINK_MIN_FRAMES   = 2    # consecutive frames below threshold = blink
EAR_PROLONGED_FRAMES   = 9    # ≈300 ms at 30 fps → fatigue indicator

# Facial asymmetry landmark pairs (left / right mirror points)
ASYMMETRY_PAIRS: List[Tuple[int, int]] = [
    (234, 454),   # cheeks
    (127, 356),   # jaw sides
    (93,  323),   # mouth corners region
    (33,  263),   # eye corners
    (70,  300),   # eyebrow outer
    (105, 334),   # eyebrow inner
    (4,   4),     # nose tip (mirror itself for size)
]

MUSCLE_TONE_PAIRS: List[Tuple[int, int]] = [
    (61,  291),   # lip corners
    (13,  14),    # lip top/bottom
    (9,   200),   # upper/lower face mid
]


# ─────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────

def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _lm_px(landmark, h: int, w: int) -> Tuple[float, float]:
    """Convert normalised landmark to pixel coordinates."""
    return landmark.x * w, landmark.y * h


def compute_ear(landmarks, indices: Tuple[int, ...], h: int, w: int) -> float:
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    indices order: (p1, p4, p2, p3, p5, p6)
    Per spec: left=(33,133,160,158,153,144), right=(362,263,385,387,373,380)
    """
    lm = landmarks.landmark
    p1 = _lm_px(lm[indices[0]], h, w)
    p4 = _lm_px(lm[indices[1]], h, w)
    p2 = _lm_px(lm[indices[2]], h, w)
    p3 = _lm_px(lm[indices[3]], h, w)
    p5 = _lm_px(lm[indices[4]], h, w)
    p6 = _lm_px(lm[indices[5]], h, w)

    vertical_1 = _euclidean(p2, p6)
    vertical_2 = _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)

    if horizontal < 1e-6:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


# ─────────────────────────────────────────────
#  3D Facial asymmetry
# ─────────────────────────────────────────────

def compute_facial_asymmetry(landmarks, h: int, w: int) -> Dict:
    """
    Measure left-right spatial asymmetry using mirrored landmark pairs.
    Uses 3D coordinates (x, y, z) from MediaPipe Face Mesh.
    """
    lm = landmarks.landmark
    # Face centre line: mid-x of nose bridge (1, 168) and chin (152)
    nose_x = lm[1].x * w

    asymmetry_deltas = []
    for left_idx, right_idx in ASYMMETRY_PAIRS:
        if left_idx == right_idx:
            continue
        lx = lm[left_idx].x * w
        rx = lm[right_idx].x * w
        # Each landmark should be equidistant from midline
        dist_l = abs(lx - nose_x)
        dist_r = abs(rx - nose_x)
        if dist_l + dist_r > 0:
            delta = abs(dist_l - dist_r) / ((dist_l + dist_r) / 2.0)
            asymmetry_deltas.append(delta)

    # Vertical asymmetry (y-level differences)
    ly_vals = [lm[p[0]].y * h for p in ASYMMETRY_PAIRS if p[0] != p[1]]
    ry_vals = [lm[p[1]].y * h for p in ASYMMETRY_PAIRS if p[0] != p[1]]
    y_deltas = [abs(ly - ry) / max(abs(ly + ry) / 2, 1.0)
                for ly, ry in zip(ly_vals, ry_vals)]

    all_deltas = asymmetry_deltas + y_deltas
    if not all_deltas:
        return {"facial_asymmetry_score": 0.0, "left_right_ratio": None}

    asymmetry_score = float(np.clip(np.mean(all_deltas) * 5.0, 0.0, 1.0))

    # L/R ratio via cheek-landmark distances to nose tip
    cheek_l_d = _euclidean(_lm_px(lm[234], h, w), _lm_px(lm[1], h, w))
    cheek_r_d = _euclidean(_lm_px(lm[454], h, w), _lm_px(lm[1], h, w))
    lr_ratio  = cheek_l_d / cheek_r_d if cheek_r_d > 0 else None

    return {
        "facial_asymmetry_score": asymmetry_score,
        "left_right_ratio":       lr_ratio,
    }


def compute_muscle_tone_imbalance(landmarks, h: int, w: int) -> float:
    """
    Estimate muscle tone imbalance via mouth and mid-face landmark geometry.
    Returns score 0-1 (0 = perfectly balanced).
    """
    lm = landmarks.landmark
    imbalances = []
    for l_idx, r_idx in MUSCLE_TONE_PAIRS:
        ly = lm[l_idx].y * h
        ry = lm[r_idx].y * h
        lx = lm[l_idx].x * w
        rx = lm[r_idx].x * w
        h_diff = abs(ly - ry) / max(h * 0.01, 1.0)
        imbalances.append(min(h_diff, 1.0))
    return float(np.clip(np.mean(imbalances) if imbalances else 0.0, 0.0, 1.0))


def compute_stress_structural_score(
    asymmetry_score: float,
    muscle_tone: float,
    ear_avg: float,
) -> float:
    """
    Composite stress indicator from structural facial markers.
    Low EAR + high asymmetry + muscle imbalance → elevated stress score.
    """
    ear_stress = max(0.0, 1.0 - ear_avg / 0.3)    # below 0.3 → stress signal
    score = 0.4 * asymmetry_score + 0.3 * muscle_tone + 0.3 * ear_stress
    return float(np.clip(score, 0.0, 1.0))


def compute_emotional_load(landmarks, h: int, w: int) -> float:
    """
    Baseline emotional load heuristic:
    Uses brow lowering (102, 331) and lip compression (61, 291, 13, 14).
    """
    lm = landmarks.landmark
    # Brow-to-eye vertical distance (furrowed = lower)
    brow_eye_l = abs(lm[105].y - lm[159].y) * h
    brow_eye_r = abs(lm[334].y - lm[386].y) * h
    brow_norm  = np.clip((brow_eye_l + brow_eye_r) / (2.0 * h * 0.08), 0.0, 1.0)
    brow_load  = 1.0 - float(brow_norm)    # lower brows → higher load

    # Lip compression (narrow mouth opening)
    mouth_h    = abs(lm[13].y - lm[14].y) * h
    mouth_w    = abs(lm[61].x - lm[291].x) * w
    lip_ratio  = mouth_h / max(mouth_w * 0.3, 1e-3)
    lip_load   = float(np.clip(1.0 - lip_ratio, 0.0, 1.0))

    return float(np.clip(0.6 * brow_load + 0.4 * lip_load, 0.0, 1.0))


# ─────────────────────────────────────────────
#  FaceAnalyzer: stateful per-session processor
# ─────────────────────────────────────────────

class FaceAnalyzer:
    """
    Processes individual BGR frames (as np.ndarray) coming from a WebSocket
    session and accumulates face biomarkers using a sliding window.

    No camera access, no display windows — cloud-safe.
    """

    def __init__(self):
        self.rppg = rPPGExtractor()

        # EAR blink tracking
        self._blink_count          = 0
        self._prolonged_blink_count = 0
        self._blink_frame_counter  = 0          # frames with EAR < threshold
        self._in_blink             = False
        self._ear_history: collections.deque = collections.deque(maxlen=300)
        self._blink_start_frame    = 0

        # 3D face accumulation (average over session)
        self._asym_scores: List[float]  = []
        self._muscle_scores: List[float] = []
        self._stress_scores: List[float] = []
        self._emotional_loads: List[float] = []

        # Skin texture
        self._skin_history: collections.deque = collections.deque(maxlen=60)

        self._frame_idx    = 0
        self._session_start = time.time()

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: float = 0.0,
    ) -> Dict:
        """
        Run full face analysis pipeline on one BGR frame.
        Returns a dict with all current metrics (safe for JSON serialisation).
        """
        h, w = frame.shape[:2]
        self._frame_idx += 1

        # ── 1. MediaPipe Face Mesh ──────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = FACE_MESH.process(rgb)

        if not result.multi_face_landmarks:
            return self._build_output(landmarks_found=False)

        landmarks = result.multi_face_landmarks[0]

        # ── 2. rPPG ────────────────────────────────────────────────
        rppg_metrics = self.rppg.process_frame(frame, landmarks, timestamp_ms)

        # ── 3. EAR & Blink Detection ────────────────────────────────
        ear_left  = compute_ear(landmarks, EAR_LEFT_IDX,  h, w)
        ear_right = compute_ear(landmarks, EAR_RIGHT_IDX, h, w)
        ear_avg   = (ear_left + ear_right) / 2.0
        self._ear_history.append(ear_avg)
        blink_metrics = self._update_blink_state(ear_avg)

        # ── 4. Skin Texture Proxy ───────────────────────────────────
        skin_metrics = compute_skin_texture_metrics(frame, landmarks)
        if skin_metrics["cheek_rgb_variance"] is not None:
            self._skin_history.append(skin_metrics["hydration_proxy_score"])
        skin_avg_score = float(np.mean(self._skin_history)) if self._skin_history else None

        # ── 5. 3D Structural Analysis ───────────────────────────────
        asym   = compute_facial_asymmetry(landmarks, h, w)
        muscle = compute_muscle_tone_imbalance(landmarks, h, w)
        stress = compute_stress_structural_score(
            asym["facial_asymmetry_score"], muscle, ear_avg
        )
        emo_load = compute_emotional_load(landmarks, h, w)

        # Accumulate for session averages
        self._asym_scores.append(asym["facial_asymmetry_score"])
        self._muscle_scores.append(muscle)
        self._stress_scores.append(stress)
        self._emotional_loads.append(emo_load)

        # ── 6. Landmark confidence (fraction of visible face landmarks) ─
        visible = sum(1 for lm in landmarks.landmark if lm.visibility > 0.5
                      ) if hasattr(landmarks.landmark[0], "visibility") else 468
        lm_confidence = visible / 468.0

        return self._build_output(
            landmarks_found=True,
            rppg=rppg_metrics,
            ear_left=ear_left,
            ear_right=ear_right,
            ear_avg=ear_avg,
            blink=blink_metrics,
            skin=skin_metrics,
            skin_avg_score=skin_avg_score,
            asym=asym,
            muscle_tone=muscle,
            stress=stress,
            emo_load=emo_load,
            lm_confidence=lm_confidence,
        )

    def _update_blink_state(self, ear: float) -> Dict:
        """
        Simple blink FSM:
          - EAR < threshold for ≥ MIN_FRAMES → register blink
          - Duration ≥ PROLONGED_FRAMES      → register prolonged blink (fatigue)
        """
        if ear < EAR_BLINK_THRESHOLD:
            if not self._in_blink:
                self._in_blink = True
                self._blink_start_frame = self._frame_idx
            self._blink_frame_counter += 1
        else:
            if self._in_blink and self._blink_frame_counter >= EAR_BLINK_MIN_FRAMES:
                self._blink_count += 1
                if self._blink_frame_counter >= EAR_PROLONGED_FRAMES:
                    self._prolonged_blink_count += 1
            self._in_blink = False
            self._blink_frame_counter = 0

        elapsed_min = (time.time() - self._session_start) / 60.0
        blink_rate  = (self._blink_count / elapsed_min) if elapsed_min > 0.01 else None

        # Fatigue score: combine blink rate and prolonged blink ratio
        if blink_rate is not None:
            # Normal blink rate ≈ 15-20/min; >25 or <5 → potential issues
            rate_norm    = min(blink_rate / 30.0, 1.0)
            prolong_norm = min(self._prolonged_blink_count / max(self._blink_count, 1), 1.0)
            fatigue      = float(np.clip(0.4 * rate_norm + 0.6 * prolong_norm, 0.0, 1.0))
        else:
            fatigue = 0.0

        return {
            "blink_count":           self._blink_count,
            "prolonged_blink_count": self._prolonged_blink_count,
            "blink_rate_per_min":    blink_rate,
            "fatigue_score":         fatigue,
        }

    def _build_output(
        self,
        landmarks_found: bool = False,
        rppg: Optional[Dict] = None,
        ear_left: float = 0.0,
        ear_right: float = 0.0,
        ear_avg: float = 0.0,
        blink: Optional[Dict] = None,
        skin: Optional[Dict] = None,
        skin_avg_score: Optional[float] = None,
        asym: Optional[Dict] = None,
        muscle_tone: float = 0.0,
        stress: float = 0.0,
        emo_load: float = 0.0,
        lm_confidence: float = 0.0,
    ) -> Dict:
        """Assemble final output dict, safe for JSON serialisation."""
        rppg   = rppg   or {}
        blink  = blink  or {}
        skin   = skin   or {}
        asym   = asym   or {}

        session_asym   = float(np.mean(self._asym_scores))   if self._asym_scores   else 0.0
        session_muscle = float(np.mean(self._muscle_scores)) if self._muscle_scores else 0.0
        session_stress = float(np.mean(self._stress_scores)) if self._stress_scores else 0.0
        session_emo    = float(np.mean(self._emotional_loads)) if self._emotional_loads else 0.0

        return {
            # Status
            "landmarks_found":      landmarks_found,
            "frames_processed":     self._frame_idx,
            "landmark_confidence":  lm_confidence,

            # rPPG / cardio-respiratory
            "heart_rate_bpm":         rppg.get("heart_rate_bpm"),
            "hrv_sdnn_ms":            rppg.get("hrv_sdnn_ms"),
            "hrv_rmssd_ms":           rppg.get("hrv_rmssd_ms"),
            "respiratory_rate_bpm":   rppg.get("respiratory_rate_bpm"),
            "spo2_estimate_pct":      rppg.get("spo2_estimate_pct"),
            "pulse_wave_samples":     rppg.get("pulse_wave_samples", []),
            "rppg_quality_score":     rppg.get("rppg_quality_score", 0.0),

            # Ocular / EAR
            "ear_left":               ear_left,
            "ear_right":              ear_right,
            "ear_average":            ear_avg,
            "blink_count":            blink.get("blink_count", self._blink_count),
            "prolonged_blink_count":  blink.get("prolonged_blink_count", self._prolonged_blink_count),
            "blink_rate_per_min":     blink.get("blink_rate_per_min"),
            "fatigue_score":          blink.get("fatigue_score", 0.0),

            # Skin texture proxy
            "cheek_rgb_variance":     skin.get("cheek_rgb_variance"),
            "specular_ratio":         skin.get("specular_ratio"),
            "hydration_proxy_score":  skin_avg_score,
            "alert_dehydration":      skin.get("alert_dehydration", False),
            "experimental_confidence_low": True,

            # 3D structural (per-frame + session mean)
            "facial_asymmetry_score":         asym.get("facial_asymmetry_score", session_asym),
            "left_right_ratio":               asym.get("left_right_ratio"),
            "muscle_tone_imbalance_score":    muscle_tone,
            "stress_structural_score":        stress,
            "emotional_load_baseline":        emo_load,

            # Session averages (stable over time)
            "session_avg_asymmetry":      session_asym,
            "session_avg_muscle_tone":    session_muscle,
            "session_avg_stress":         session_stress,
            "session_avg_emotional_load": session_emo,
        }

    def get_final_summary(self) -> Dict:
        """Return stable session-level averages for final report."""
        rppg_final = self.rppg._current_metrics()
        return {
            "heart_rate_bpm":             rppg_final.get("heart_rate_bpm"),
            "hrv_sdnn_ms":                rppg_final.get("hrv_sdnn_ms"),
            "hrv_rmssd_ms":               rppg_final.get("hrv_rmssd_ms"),
            "respiratory_rate_bpm":       rppg_final.get("respiratory_rate_bpm"),
            "spo2_estimate_pct":          rppg_final.get("spo2_estimate_pct"),
            "rppg_quality_score":         rppg_final.get("rppg_quality_score", 0.0),
            "pulse_wave_samples":         rppg_final.get("pulse_wave_samples", []),
            "blink_count":                self._blink_count,
            "prolonged_blink_count":      self._prolonged_blink_count,
            "blink_rate_per_min":         self._blink_count / max((time.time() - self._session_start) / 60, 0.01),
            "fatigue_score":              min(self._prolonged_blink_count / max(self._blink_count, 1), 1.0),
            "hydration_proxy_score":      float(np.mean(self._skin_history)) if self._skin_history else None,
            "alert_dehydration":          float(np.mean(self._skin_history)) < 0.2 if self._skin_history else False,
            "facial_asymmetry_score":     float(np.mean(self._asym_scores))   if self._asym_scores   else 0.0,
            "muscle_tone_imbalance_score": float(np.mean(self._muscle_scores)) if self._muscle_scores else 0.0,
            "stress_structural_score":    float(np.mean(self._stress_scores)) if self._stress_scores else 0.0,
            "emotional_load_baseline":    float(np.mean(self._emotional_loads)) if self._emotional_loads else 0.0,
            "frames_processed":           self._frame_idx,
            "experimental_confidence_low": True,
        }

    def reset(self):
        self.rppg.reset()
        self._blink_count = 0
        self._prolonged_blink_count = 0
        self._blink_frame_counter = 0
        self._in_blink = False
        self._ear_history.clear()
        self._asym_scores.clear()
        self._muscle_scores.clear()
        self._stress_scores.clear()
        self._emotional_loads.clear()
        self._skin_history.clear()
        self._frame_idx = 0
        self._session_start = time.time()
