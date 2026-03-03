"""
rppg_extractor.py
Remote Photoplethysmography signal extraction.

FIX: No module-level MediaPipe initialization.
All heavy objects are created inside the class constructor (lazy, per-session).
"""

from __future__ import annotations

import collections
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import signal as sp_signal

# ── Constants ──────────────────────────────────────────────────────────────
BUFFER_SIZE = 300       # max frames (~10 s at 30 fps) — enforced by deque maxlen
MIN_FRAMES  = 90        # need ~3 s before first HR estimate
FPS_DEFAULT = 30.0

BP_LOW  = 0.7           # Hz (42 BPM)
BP_HIGH = 3.5           # Hz (210 BPM)
FILTER_ORDER = 4

SPO2_A = 110.0          # Beer-Lambert calibration constants
SPO2_B = 25.0

# MediaPipe Face Mesh landmark indices used for dynamic ROI
LM_EYEBROW_L  = 70
LM_EYEBROW_R  = 296
LM_EYE_L      = 33
LM_EYE_R      = 263
LM_CHEEK_L    = 234
LM_CHEEK_R    = 454


# ── Signal processing helpers ──────────────────────────────────────────────

def _bandpass(data: np.ndarray, low: float, high: float,
              fs: float, order: int = FILTER_ORDER) -> np.ndarray:
    nyq = 0.5 * fs
    lo  = max(low  / nyq, 1e-4)
    hi  = min(high / nyq, 1.0 - 1e-4)
    if lo >= hi or len(data) < order * 3:
        return data
    b, a = sp_signal.butter(order, [lo, hi], btype="band")
    return sp_signal.filtfilt(b, a, data)


def _compute_hr_from_peaks(filtered: np.ndarray, fs: float
                           ) -> Tuple[Optional[float], np.ndarray]:
    min_dist = int(fs * 60.0 / 200.0)
    peaks, _ = sp_signal.find_peaks(
        filtered,
        distance=max(min_dist, 1),
        prominence=np.std(filtered) * 0.3,
    )
    if len(peaks) < 2:
        return None, np.array([])
    rr = np.diff(peaks) / fs * 1000.0
    rr = rr[(rr > 300) & (rr < 2000)]
    if len(rr) == 0:
        return None, np.array([])
    return float(60_000.0 / np.mean(rr)), rr


def _compute_hrv(rr: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if len(rr) < 3:
        return None, None
    return float(np.std(rr, ddof=1)), float(np.sqrt(np.mean(np.diff(rr) ** 2)))


def _compute_rr_rate(filtered: np.ndarray, fs: float) -> Optional[float]:
    if len(filtered) < int(fs * 6):
        return None
    envelope = np.abs(sp_signal.hilbert(filtered))
    envelope = sp_signal.detrend(envelope)
    rr_band  = _bandpass(envelope, 0.1, 0.5, fs, order=2)
    freqs    = np.fft.rfftfreq(len(rr_band), d=1.0 / fs)
    power    = np.abs(np.fft.rfft(rr_band)) ** 2
    mask     = (freqs >= 0.1) & (freqs <= 0.5)
    if not np.any(mask):
        return None
    return float(freqs[mask][np.argmax(power[mask])] * 60.0)


def _snr(raw: np.ndarray, filtered: np.ndarray) -> float:
    total = np.var(raw) + 1e-9
    sig   = np.var(filtered) + 1e-9
    return float(np.clip(sig / total, 0.0, 1.0))


# ── Dynamic ROI helpers ────────────────────────────────────────────────────

def compute_forehead_roi(landmarks, h: int, w: int,
                         pad: float = 0.05) -> Optional[Tuple[int, int, int, int]]:
    """
    Dynamic forehead bounding box using eyebrow (70, 296) and eye (33, 263)
    landmarks. The ROI sits ABOVE the eyebrows — adapts to face shape/distance.
    """
    try:
        lm = landmarks.landmark
        eyebrow_y = min(lm[LM_EYEBROW_L].y, lm[LM_EYEBROW_R].y) * h
        pad_px    = int(abs(lm[LM_EYE_L].y * h - eyebrow_y) * pad)
        y2 = max(0, int(eyebrow_y) - pad_px)
        y1 = max(0, y2 - max(int((y2) * 0.15), 10))   # ~15% of face height above brow
        x1 = max(0, int(lm[LM_EYEBROW_L].x * w) - int(w * pad))
        x2 = min(w, int(lm[LM_EYEBROW_R].x * w) + int(w * pad))
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None
    except (IndexError, AttributeError):
        return None


def compute_cheek_rois(landmarks, h: int, w: int,
                       size: int = 30) -> Tuple[Optional[tuple], Optional[tuple]]:
    try:
        lm = landmarks.landmark
        def box(i):
            cx, cy = int(lm[i].x * w), int(lm[i].y * h)
            return (max(0, cx-size), max(0, cy-size),
                    min(w, cx+size), min(h, cy+size))
        return box(LM_CHEEK_L), box(LM_CHEEK_R)
    except (IndexError, AttributeError):
        return None, None


def compute_skin_texture(frame: np.ndarray, landmarks) -> Dict:
    """Experimental hydration proxy from cheek-ROI RGB statistics."""
    h, w = frame.shape[:2]
    lb, rb = compute_cheek_rois(landmarks, h, w)
    variances, speculars = [], []
    for box in [lb, rb]:
        if box is None:
            continue
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        variances.append(float(np.mean([np.var(roi[:, :, c]) for c in range(3)])))
        speculars.append(float(np.any(roi > 240, axis=-1).mean()))
    if not variances:
        return {"cheek_rgb_variance": None, "specular_ratio": None,
                "hydration_proxy_score": None, "alert_dehydration": False,
                "experimental_confidence_low": True}
    v = float(np.mean(variances))
    s = float(np.mean(speculars))
    score = float(np.clip(0.5 * np.clip((v - 50) / 450, 0, 1) + 0.5 * (1.0 - s), 0, 1))
    return {"cheek_rgb_variance": v, "specular_ratio": s,
            "hydration_proxy_score": score, "alert_dehydration": score < 0.2,
            "experimental_confidence_low": True}


# ── Main extractor class ───────────────────────────────────────────────────

class rPPGExtractor:
    """
    Stateful per-session rPPG processor.
    BUFFER_SIZE deques automatically drop oldest frames — no memory leak.
    """

    def __init__(self, buffer_size: int = BUFFER_SIZE, fps: float = FPS_DEFAULT):
        self.fps         = fps
        self._buf_r: collections.deque = collections.deque(maxlen=buffer_size)
        self._buf_g: collections.deque = collections.deque(maxlen=buffer_size)
        self._buf_b: collections.deque = collections.deque(maxlen=buffer_size)
        self._buf_ts: collections.deque = collections.deque(maxlen=buffer_size)
        self._frame_n   = 0
        self._update_every = 15         # recompute every N frames
        # Cached results
        self._hr: Optional[float]    = None
        self._sdnn: Optional[float]  = None
        self._rmssd: Optional[float] = None
        self._rr: Optional[float]    = None
        self._spo2: Optional[float]  = None
        self._quality: float         = 0.0
        self._filtered: List[float]  = []

    def process_frame(self, frame: np.ndarray, landmarks,
                      timestamp_ms: float = 0.0) -> Dict:
        h, w = frame.shape[:2]
        self._frame_n += 1

        roi = compute_forehead_roi(landmarks, h, w)
        if roi:
            x1, y1, x2, y2 = roi
            patch = frame[y1:y2, x1:x2]
            if patch.size > 0:
                self._buf_b.append(float(np.mean(patch[:, :, 0])))
                self._buf_g.append(float(np.mean(patch[:, :, 1])))
                self._buf_r.append(float(np.mean(patch[:, :, 2])))
                self._buf_ts.append(timestamp_ms)

        if self._frame_n % self._update_every == 0:
            self._recompute()

        return self._snapshot()

    def _recompute(self):
        n = len(self._buf_g)
        if n < MIN_FRAMES:
            return

        g  = np.array(self._buf_g)
        r  = np.array(self._buf_r)
        b  = np.array(self._buf_b)

        # Refine fps from timestamps
        if len(self._buf_ts) >= 2:
            ts  = np.array(self._buf_ts)
            dt  = np.median(np.diff(ts)) / 1000.0
            if dt > 0:
                self.fps = float(np.clip(1.0 / dt, 5.0, 60.0))

        # CHROM rPPG — more robust to illumination than raw green channel
        raw = sp_signal.detrend(3.0 * r - 2.0 * g)
        if len(raw) < 10:
            return
        filt = _bandpass(raw, BP_LOW, BP_HIGH, self.fps)

        hr, rr_intervals     = _compute_hr_from_peaks(filt, self.fps)
        sdnn, rmssd          = _compute_hrv(rr_intervals)
        rr_rate              = _compute_rr_rate(filt, self.fps)
        quality              = _snr(g, filt)

        # SpO2 surrogate (R/G ratio proxy for R/IR)
        r_ac, r_dc = np.std(r), np.mean(r)
        g_ac, g_dc = np.std(g), np.mean(g)
        spo2 = None
        if r_dc > 0 and g_dc > 0 and g_ac > 0:
            ratio = (r_ac / r_dc) / (g_ac / g_dc)
            spo2  = float(np.clip(SPO2_A - SPO2_B * ratio, 85.0, 100.0))

        self._hr      = hr
        self._sdnn    = sdnn
        self._rmssd   = rmssd
        self._rr      = rr_rate
        self._spo2    = spo2
        self._quality = quality
        self._filtered = filt[-150:].tolist()   # last 5 s for frontend chart

    def _snapshot(self) -> Dict:
        return {
            "heart_rate_bpm":       self._hr,
            "hrv_sdnn_ms":          self._sdnn,
            "hrv_rmssd_ms":         self._rmssd,
            "respiratory_rate_bpm": self._rr,
            "spo2_estimate_pct":    self._spo2,
            "pulse_wave_samples":   self._filtered,
            "rppg_quality_score":   self._quality,
        }

    def reset(self):
        self._buf_r.clear(); self._buf_g.clear()
        self._buf_b.clear(); self._buf_ts.clear()
        self._frame_n = 0; self._filtered = []
        self._hr = self._sdnn = self._rmssd = self._rr = self._spo2 = None
        self._quality = 0.0
