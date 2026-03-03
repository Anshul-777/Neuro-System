"""
face_analyzer.py
Face biomarker pipeline — FaceMesh + rPPG + EAR + 3D asymmetry.

CRITICAL FIX: MediaPipe FaceMesh is initialised INSIDE __init__(),
NOT at module level. Module-level init crashed Render before port binding.
"""

from __future__ import annotations

import collections
import math
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from rppg_extractor import rPPGExtractor, compute_skin_texture

# ── EAR landmark indices (per spec) ──────────────────────────────────────
# Left eye:   p1=33,  p4=133, p2=160, p3=158, p5=153, p6=144
# Right eye:  p1=362, p4=263, p2=385, p3=387, p5=373, p6=380
EAR_L = (33,  133, 160, 158, 153, 144)
EAR_R = (362, 263, 385, 387, 373, 380)

EAR_BLINK_THRESH    = 0.25
EAR_BLINK_MIN_FRAME = 2     # consecutive sub-threshold frames = blink
EAR_PROLONGED_FRAME = 9     # ≈300 ms at 30 fps = fatigue marker

ASYM_PAIRS = [(234,454),(127,356),(93,323),(33,263),(70,300),(105,334)]
TONE_PAIRS = [(61,291),(13,14)]


# ── Geometry helpers ──────────────────────────────────────────────────────

def _px(lm, idx: int, h: int, w: int) -> Tuple[float, float]:
    return lm[idx].x * w, lm[idx].y * h

def _dist(a, b) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def ear(landmarks, idx: Tuple[int,...], h: int, w: int) -> float:
    """EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)"""
    lm = landmarks.landmark
    p1,p4,p2,p3,p5,p6 = [_px(lm, i, h, w) for i in idx]
    horiz = _dist(p1, p4)
    return (_dist(p2,p6) + _dist(p3,p5)) / (2.0*horiz) if horiz > 1e-6 else 0.0


# ── 3D structural helpers ─────────────────────────────────────────────────

def facial_asymmetry(landmarks, h: int, w: int) -> Dict:
    lm = landmarks.landmark
    nose_x = lm[1].x * w
    deltas = []
    for li, ri in ASYM_PAIRS:
        lx = lm[li].x * w; rx = lm[ri].x * w
        dl = abs(lx - nose_x); dr = abs(rx - nose_x)
        if dl + dr > 0:
            deltas.append(abs(dl - dr) / ((dl+dr)/2.0))
    score = float(np.clip(np.mean(deltas)*5, 0, 1)) if deltas else 0.0
    cl = _px(lm,234,h,w); cr = _px(lm,454,h,w); nt = _px(lm,1,h,w)
    lr = _dist(cl,nt)/_dist(cr,nt) if _dist(cr,nt)>0 else None
    return {"facial_asymmetry_score": score, "left_right_ratio": lr}

def muscle_tone(landmarks, h: int, w: int) -> float:
    lm = landmarks.landmark
    vals = []
    for li, ri in TONE_PAIRS:
        vals.append(min(abs(lm[li].y - lm[ri].y)*h / max(h*0.01,1), 1.0))
    return float(np.clip(np.mean(vals), 0, 1)) if vals else 0.0

def stress_score(asym: float, tone: float, ear_avg: float) -> float:
    return float(np.clip(0.4*asym + 0.3*tone + 0.3*max(0,1-ear_avg/0.3), 0, 1))

def emotional_load(landmarks, h: int, w: int) -> float:
    lm = landmarks.landmark
    be_l = abs(lm[105].y - lm[159].y)*h
    be_r = abs(lm[334].y - lm[386].y)*h
    brow = 1.0 - float(np.clip((be_l+be_r)/(2*h*0.08), 0, 1))
    mh   = abs(lm[13].y - lm[14].y)*h
    mw   = abs(lm[61].x - lm[291].x)*w
    lip  = float(np.clip(1 - mh/max(mw*0.3,1e-3), 0, 1))
    return float(np.clip(0.6*brow + 0.4*lip, 0, 1))


# ── FaceAnalyzer ──────────────────────────────────────────────────────────

class FaceAnalyzer:
    """
    Per-session face analyzer. Heavy objects (MediaPipe) created in __init__
    only when a session starts — NOT at module import time.
    """

    def __init__(self):
        # ── Lazy import of mediapipe inside constructor ──────────────
        import mediapipe as mp
        self._fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.rppg = rPPGExtractor()

        # Blink FSM
        self._blink_total      = 0
        self._prolonged_total  = 0
        self._blink_frames     = 0
        self._in_blink         = False
        self._ear_buf: collections.deque = collections.deque(maxlen=300)

        # 3D structural accumulators
        self._asym:   List[float] = []
        self._muscle: List[float] = []
        self._stress: List[float] = []
        self._emo:    List[float] = []
        self._skin_buf: collections.deque = collections.deque(maxlen=60)

        self._n = 0
        self._t0 = time.time()

    def process_frame(self, frame: np.ndarray,
                      timestamp_ms: float = 0.0) -> Dict:
        h, w = frame.shape[:2]
        self._n += 1

        import cv2
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._fm.process(rgb)

        if not result.multi_face_landmarks:
            return self._out(found=False)

        lm = result.multi_face_landmarks[0]

        # rPPG
        rppg = self.rppg.process_frame(frame, lm, timestamp_ms)

        # EAR
        el  = ear(lm, EAR_L, h, w)
        er  = ear(lm, EAR_R, h, w)
        ea  = (el + er) / 2.0
        self._ear_buf.append(ea)
        blink = self._update_blink(ea)

        # Skin
        skin = compute_skin_texture(frame, lm)
        if skin["hydration_proxy_score"] is not None:
            self._skin_buf.append(skin["hydration_proxy_score"])

        # 3D structural
        asym  = facial_asymmetry(lm, h, w)
        tone  = muscle_tone(lm, h, w)
        ss    = stress_score(asym["facial_asymmetry_score"], tone, ea)
        emo   = emotional_load(lm, h, w)

        self._asym.append(asym["facial_asymmetry_score"])
        self._muscle.append(tone)
        self._stress.append(ss)
        self._emo.append(emo)

        return self._out(
            found=True, rppg=rppg,
            el=el, er=er, ea=ea, blink=blink,
            skin=skin, asym=asym, tone=tone, ss=ss, emo=emo,
        )

    def _update_blink(self, ea: float) -> Dict:
        if ea < EAR_BLINK_THRESH:
            if not self._in_blink:
                self._in_blink = True
            self._blink_frames += 1
        else:
            if self._in_blink and self._blink_frames >= EAR_BLINK_MIN_FRAME:
                self._blink_total += 1
                if self._blink_frames >= EAR_PROLONGED_FRAME:
                    self._prolonged_total += 1
            self._in_blink = False
            self._blink_frames = 0

        elapsed_min = max((time.time() - self._t0) / 60.0, 1e-3)
        rate  = self._blink_total / elapsed_min
        prolong_r = self._prolonged_total / max(self._blink_total, 1)
        fatigue   = float(np.clip(0.4*min(rate/30,1) + 0.6*prolong_r, 0, 1))
        return {"blink_count": self._blink_total,
                "prolonged_blink_count": self._prolonged_total,
                "blink_rate_per_min": rate,
                "fatigue_score": fatigue}

    def _out(self, found=False, rppg=None, el=0.0, er=0.0, ea=0.0,
             blink=None, skin=None, asym=None, tone=0.0, ss=0.0, emo=0.0) -> Dict:
        rppg  = rppg  or {}
        blink = blink or {}
        skin  = skin  or {}
        asym  = asym  or {}
        sa = float(np.mean(self._asym))   if self._asym   else 0.0
        sm = float(np.mean(self._muscle)) if self._muscle else 0.0
        ss_avg = float(np.mean(self._stress)) if self._stress else 0.0
        se = float(np.mean(self._emo))    if self._emo    else 0.0
        hydration = float(np.mean(self._skin_buf)) if self._skin_buf else None
        return {
            "landmarks_found":          found,
            "frames_processed":         self._n,
            # rPPG
            "heart_rate_bpm":           rppg.get("heart_rate_bpm"),
            "hrv_sdnn_ms":              rppg.get("hrv_sdnn_ms"),
            "hrv_rmssd_ms":             rppg.get("hrv_rmssd_ms"),
            "respiratory_rate_bpm":     rppg.get("respiratory_rate_bpm"),
            "spo2_estimate_pct":        rppg.get("spo2_estimate_pct"),
            "pulse_wave_samples":       rppg.get("pulse_wave_samples", []),
            "rppg_quality_score":       rppg.get("rppg_quality_score", 0.0),
            # EAR / ocular
            "ear_left":                 el,
            "ear_right":                er,
            "ear_average":              ea,
            "blink_count":              blink.get("blink_count", self._blink_total),
            "prolonged_blink_count":    blink.get("prolonged_blink_count", self._prolonged_total),
            "blink_rate_per_min":       blink.get("blink_rate_per_min"),
            "fatigue_score":            blink.get("fatigue_score", 0.0),
            # Skin
            "hydration_proxy_score":    hydration,
            "alert_dehydration":        (hydration is not None and hydration < 0.2),
            "experimental_confidence_low": True,
            # 3D structural
            "facial_asymmetry_score":   asym.get("facial_asymmetry_score", sa),
            "left_right_ratio":         asym.get("left_right_ratio"),
            "muscle_tone_imbalance_score": tone,
            "stress_structural_score":  ss,
            "emotional_load_baseline":  emo,
            "session_avg_asymmetry":    sa,
            "session_avg_muscle_tone":  sm,
            "session_avg_stress":       ss_avg,
            "session_avg_emotional_load": se,
        }

    def get_final_summary(self) -> Dict:
        snap = self.rppg._snapshot()
        elapsed_min = max((time.time()-self._t0)/60.0, 1e-3)
        hydration = float(np.mean(self._skin_buf)) if self._skin_buf else None
        return {
            **snap,
            "blink_count":              self._blink_total,
            "prolonged_blink_count":    self._prolonged_total,
            "blink_rate_per_min":       self._blink_total / elapsed_min,
            "fatigue_score":            self._prolonged_total / max(self._blink_total,1),
            "hydration_proxy_score":    hydration,
            "alert_dehydration":        (hydration is not None and hydration < 0.2),
            "facial_asymmetry_score":   float(np.mean(self._asym))   if self._asym   else 0.0,
            "muscle_tone_imbalance_score": float(np.mean(self._muscle)) if self._muscle else 0.0,
            "stress_structural_score":  float(np.mean(self._stress)) if self._stress else 0.0,
            "emotional_load_baseline":  float(np.mean(self._emo))    if self._emo    else 0.0,
            "frames_processed":         self._n,
            "experimental_confidence_low": True,
        }

    def reset(self):
        self.rppg.reset()
        self._blink_total = self._prolonged_total = self._blink_frames = 0
        self._in_blink = False
        self._ear_buf.clear(); self._asym.clear(); self._muscle.clear()
        self._stress.clear(); self._emo.clear(); self._skin_buf.clear()
        self._n = 0; self._t0 = time.time()

    def __del__(self):
        try:
            self._fm.close()
        except Exception:
            pass
