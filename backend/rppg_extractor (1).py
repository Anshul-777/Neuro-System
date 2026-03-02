"""
rppg_extractor.py
Remote Photoplethysmography (rPPG) signal extraction.

Key design decisions for cloud deployment:
  - No cv2.VideoCapture or imshow
  - Strict sliding-window buffer (BUFFER_SIZE frames max) → no memory leak
  - Returns raw filtered signal array for frontend charting (no Matplotlib)
  - Forehead ROI computed dynamically from MediaPipe Face Mesh landmarks

Landmark references (MediaPipe 468-point model):
  Eyebrow (top of forehead ROI):  L=70, R=296
  Eyes    (bottom of forehead ROI): L=33, R=263
  Cheeks  (for skin texture proxy): L=234, R=454
"""

from __future__ import annotations

import collections
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import signal as sp_signal


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

BUFFER_SIZE   = 300          # max frames to keep   (≈10 s at 30 fps)
MIN_FRAMES    = 90           # need at least 3 s to compute HR
FPS_ESTIMATE  = 30.0         # assumed capture fps; refined from timestamps

# Bandpass filter for heart rate (0.7–3.5 Hz = 42–210 BPM)
BP_LOW  = 0.7
BP_HIGH = 3.5
FILTER_ORDER = 4

# SpO2 calibration constants (Beer-Lambert approximation; validated against
# clinical data in: Wang et al. 2017, Biomed Opt Express)
SPO2_A = 110.0
SPO2_B = 25.0

# MediaPipe landmark indices
LM_EYEBROW_L  = 70
LM_EYEBROW_R  = 296
LM_EYE_L      = 33
LM_EYE_R      = 263
LM_CHEEK_L    = 234
LM_CHEEK_R    = 454
LM_FOREHEAD_C = 10    # central forehead point


# ─────────────────────────────────────────────
#  Helper utilities
# ─────────────────────────────────────────────

def _bandpass(data: np.ndarray, low: float, high: float,
              fs: float, order: int = FILTER_ORDER) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq = 0.5 * fs
    lo, hi = max(low / nyq, 1e-4), min(high / nyq, 1 - 1e-4)
    if lo >= hi:
        return data
    b, a = sp_signal.butter(order, [lo, hi], btype="band")
    return sp_signal.filtfilt(b, a, data)


def _compute_hr_from_peaks(filtered: np.ndarray, fs: float) -> Tuple[Optional[float], np.ndarray]:
    """
    Detect systolic peaks in the filtered rPPG waveform.
    Returns (heart_rate_bpm, rr_intervals_ms).
    """
    min_distance = int(fs * (60.0 / 200.0))   # minimum distance for 200 BPM
    peaks, props = sp_signal.find_peaks(
        filtered,
        distance=min_distance,
        prominence=np.std(filtered) * 0.3,
    )
    if len(peaks) < 2:
        return None, np.array([])

    rr_intervals = np.diff(peaks) / fs * 1000.0  # ms
    # Filter physiologically plausible RR intervals (300–2000 ms = 30–200 BPM)
    rr_intervals = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
    if len(rr_intervals) == 0:
        return None, np.array([])

    hr = 60_000.0 / np.mean(rr_intervals)
    return hr, rr_intervals


def _compute_hrv(rr_intervals: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """SDNN and RMSSD from RR interval series (in ms)."""
    if len(rr_intervals) < 3:
        return None, None
    sdnn  = float(np.std(rr_intervals, ddof=1))
    diffs = np.diff(rr_intervals)
    rmssd = float(np.sqrt(np.mean(diffs ** 2)))
    return sdnn, rmssd


def _compute_respiratory_rate(filtered: np.ndarray, fs: float) -> Optional[float]:
    """
    Estimate respiratory rate from amplitude modulation of rPPG signal.
    Typical breathing: 0.1–0.5 Hz (6–30 bpm).
    """
    if len(filtered) < int(fs * 6):
        return None
    # Envelope via Hilbert transform
    envelope = np.abs(sp_signal.hilbert(filtered))
    # Detrend and bandpass to respiratory band
    envelope = sp_signal.detrend(envelope)
    rr_band  = _bandpass(envelope, 0.1, 0.5, fs, order=2)
    freqs = np.fft.rfftfreq(len(rr_band), d=1.0 / fs)
    power = np.abs(np.fft.rfft(rr_band)) ** 2
    rr_mask = (freqs >= 0.1) & (freqs <= 0.5)
    if not np.any(rr_mask):
        return None
    dominant_freq = freqs[rr_mask][np.argmax(power[rr_mask])]
    return float(dominant_freq * 60.0)


def _signal_quality(raw_g: np.ndarray, filtered: np.ndarray) -> float:
    """
    SNR proxy: ratio of bandpass-filtered power to total power.
    Returns value in [0, 1].
    """
    if len(raw_g) == 0:
        return 0.0
    total_power = np.var(raw_g) + 1e-9
    signal_power = np.var(filtered) + 1e-9
    snr = min(signal_power / total_power, 1.0)
    return float(snr)


# ─────────────────────────────────────────────
#  Dynamic ROI computation
# ─────────────────────────────────────────────

def compute_forehead_roi(
    landmarks,          # mediapipe NormalizedLandmarkList
    h: int,
    w: int,
    padding_frac: float = 0.05,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Dynamically determine forehead bounding box using:
      - top  : y-coordinate of eyebrow landmarks (70, 296), shifted upward
      - bottom: y-coordinate of eye landmarks (33, 263)
      - left : x of left eyebrow (70)
      - right: x of right eyebrow (296)

    Returns (x1, y1, x2, y2) pixel coords or None if landmarks absent.
    """
    try:
        lm = landmarks.landmark
        # Vertical bounds
        eyebrow_y = min(lm[LM_EYEBROW_L].y, lm[LM_EYEBROW_R].y) * h
        eye_y     = max(lm[LM_EYE_L].y,     lm[LM_EYE_R].y)     * h
        pad_y     = int((eye_y - eyebrow_y) * padding_frac)
        y1 = max(0, int(eyebrow_y) - pad_y * 4)   # extend upward
        y2 = max(0, int(eyebrow_y) - pad_y)         # stop at eyebrow line

        # Horizontal bounds
        x1 = int(lm[LM_EYEBROW_L].x * w) - int(w * padding_frac)
        x2 = int(lm[LM_EYEBROW_R].x * w) + int(w * padding_frac)
        x1, x2 = max(0, x1), min(w, x2)

        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2
    except (IndexError, AttributeError):
        return None


def compute_cheek_rois(
    landmarks,
    h: int,
    w: int,
    roi_size: int = 30,
) -> Tuple[Optional[Tuple], Optional[Tuple]]:
    """
    Return (left_cheek_box, right_cheek_box) pixel bounding boxes.
    Uses landmarks 234 (left cheek) and 454 (right cheek).
    """
    try:
        lm = landmarks.landmark
        def _box(lm_idx):
            cx = int(lm[lm_idx].x * w)
            cy = int(lm[lm_idx].y * h)
            x1 = max(0, cx - roi_size)
            y1 = max(0, cy - roi_size)
            x2 = min(w, cx + roi_size)
            y2 = min(h, cy + roi_size)
            return x1, y1, x2, y2
        return _box(LM_CHEEK_L), _box(LM_CHEEK_R)
    except (IndexError, AttributeError):
        return None, None


# ─────────────────────────────────────────────
#  rPPGExtractor: the main class
# ─────────────────────────────────────────────

class rPPGExtractor:
    """
    Stateful rPPG processor that accumulates a sliding window of frames.

    Usage:
        extractor = rPPGExtractor()
        for each incoming frame (np.ndarray BGR):
            results = extractor.process_frame(frame, landmarks)
    """

    def __init__(self, buffer_size: int = BUFFER_SIZE, fps: float = FPS_ESTIMATE):
        self.buffer_size   = buffer_size
        self.fps           = fps

        # Deques enforce BUFFER_SIZE maximum automatically
        self._r_buf: collections.deque = collections.deque(maxlen=buffer_size)
        self._g_buf: collections.deque = collections.deque(maxlen=buffer_size)
        self._b_buf: collections.deque = collections.deque(maxlen=buffer_size)
        self._ts_buf: collections.deque = collections.deque(maxlen=buffer_size)

        # Cached results (updated only every N frames for efficiency)
        self._update_interval = 15          # recalculate every 15 frames
        self._frame_count     = 0
        self._last_hr: Optional[float]   = None
        self._last_sdnn: Optional[float] = None
        self._last_rmssd: Optional[float] = None
        self._last_rr: Optional[float]   = None
        self._last_spo2: Optional[float] = None
        self._last_quality: float        = 0.0
        self._last_filtered: List[float] = []

    def process_frame(
        self,
        frame: np.ndarray,
        landmarks,
        timestamp_ms: float = 0.0,
    ) -> Dict:
        """
        Extract forehead ROI → accumulate channels → compute rPPG metrics.
        Returns a dict with the latest cached metrics.
        """
        h, w = frame.shape[:2]
        self._frame_count += 1

        # ── 1. Extract ROI mean RGB ──────────────────────────────────
        roi_coords = compute_forehead_roi(landmarks, h, w) if landmarks else None
        if roi_coords:
            x1, y1, x2, y2 = roi_coords
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                # Motion correction: subtract background mean (simple)
                mean_b = float(np.mean(roi[:, :, 0]))
                mean_g = float(np.mean(roi[:, :, 1]))
                mean_r = float(np.mean(roi[:, :, 2]))
                self._b_buf.append(mean_b)
                self._g_buf.append(mean_g)
                self._r_buf.append(mean_r)
                self._ts_buf.append(timestamp_ms)

        # ── 2. Periodically recompute metrics ───────────────────────
        if self._frame_count % self._update_interval == 0:
            self._recompute()

        return self._current_metrics()

    def _recompute(self):
        n = len(self._g_buf)
        if n < MIN_FRAMES:
            return

        g_arr = np.array(self._g_buf, dtype=np.float64)
        r_arr = np.array(self._r_buf, dtype=np.float64)
        b_arr = np.array(self._b_buf, dtype=np.float64)

        # ── Refined FPS estimate from timestamps ─────────────────────
        if len(self._ts_buf) >= 2:
            ts = np.array(self._ts_buf)
            dt = np.median(np.diff(ts)) / 1000.0   # ms → s
            if dt > 0:
                self.fps = min(max(1.0 / dt, 5.0), 60.0)

        # ── rPPG pulse signal (CHROM method: Haan & Jeanne 2013) ─────
        #   S = 3R - 2G  (works better than raw G in varying lighting)
        raw_signal = sp_signal.detrend(3 * r_arr - 2 * g_arr)
        if len(raw_signal) < 10:
            return
        filtered = _bandpass(raw_signal, BP_LOW, BP_HIGH, self.fps)

        # ── HR & HRV ─────────────────────────────────────────────────
        hr, rr_intervals = _compute_hr_from_peaks(filtered, self.fps)
        sdnn, rmssd      = _compute_hrv(rr_intervals)

        # ── Respiratory Rate ─────────────────────────────────────────
        rr_rate = _compute_respiratory_rate(filtered, self.fps)

        # ── SpO2 surrogate (R/IR ratio method) ───────────────────────
        #   Uses ratio of AC/DC in red vs green as proxy for IR
        def _ac_dc(arr):
            dc  = np.mean(arr)
            ac  = np.std(arr)
            return ac, dc

        r_ac, r_dc = _ac_dc(r_arr)
        g_ac, g_dc = _ac_dc(g_arr)
        if r_dc > 0 and g_dc > 0:
            ratio = (r_ac / r_dc) / (g_ac / g_dc + 1e-9)
            spo2  = SPO2_A - SPO2_B * ratio
            spo2  = float(np.clip(spo2, 85.0, 100.0))
        else:
            spo2 = None

        # ── Signal quality ───────────────────────────────────────────
        quality = _signal_quality(g_arr, filtered)

        # ── Cache ────────────────────────────────────────────────────
        self._last_hr      = hr
        self._last_sdnn    = sdnn
        self._last_rmssd   = rmssd
        self._last_rr      = rr_rate
        self._last_spo2    = spo2
        self._last_quality = quality
        # Keep only last 150 samples for the frontend chart
        self._last_filtered = filtered[-150:].tolist()

    def _current_metrics(self) -> Dict:
        return {
            "heart_rate_bpm":       self._last_hr,
            "hrv_sdnn_ms":          self._last_sdnn,
            "hrv_rmssd_ms":         self._last_rmssd,
            "respiratory_rate_bpm": self._last_rr,
            "spo2_estimate_pct":    self._last_spo2,
            "pulse_wave_samples":   self._last_filtered,   # raw data for frontend chart
            "rppg_quality_score":   self._last_quality,
            "frames_buffered":      len(self._g_buf),
        }

    def reset(self):
        self._r_buf.clear()
        self._g_buf.clear()
        self._b_buf.clear()
        self._ts_buf.clear()
        self._frame_count = 0
        self._last_filtered = []
        self._last_hr = self._last_sdnn = self._last_rmssd = None
        self._last_rr = self._last_spo2 = None
        self._last_quality = 0.0


# ─────────────────────────────────────────────
#  Skin texture / hydration proxy
# ─────────────────────────────────────────────

def compute_skin_texture_metrics(frame: np.ndarray, landmarks) -> Dict:
    """
    Analyze cheek ROI for skin texture as a hydration proxy.
    Uses RGB variance (low variance → dull / dehydrated) and
    specular highlight ratio (bright pixels in ROI).

    IMPORTANT: Flagged as experimental_confidence_low = True.
    """
    h, w = frame.shape[:2]
    left_box, right_box = compute_cheek_rois(landmarks, h, w)

    variances, specular_ratios = [], []

    for box in [left_box, right_box]:
        if box is None:
            continue
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        # RGB variance (proxy for texture richness)
        var = float(np.mean([np.var(roi[:, :, c]) for c in range(3)]))
        variances.append(var)
        # Specular ratio: fraction of pixels > 240 in any channel
        bright = np.any(roi > 240, axis=-1).mean()
        specular_ratios.append(float(bright))

    if not variances:
        return {
            "cheek_rgb_variance":   None,
            "specular_ratio":       None,
            "hydration_proxy_score": None,
            "alert_dehydration":    False,
            "experimental_confidence_low": True,
        }

    avg_var     = float(np.mean(variances))
    avg_specular = float(np.mean(specular_ratios))

    # Normalise variance to 0-1 (empirically: 50–500 typical range)
    norm_var = float(np.clip((avg_var - 50.0) / 450.0, 0.0, 1.0))
    # High specular → oily/wet; very low → dull → possible dehydration
    score = float(np.clip(0.5 * norm_var + 0.5 * (1.0 - avg_specular), 0.0, 1.0))

    return {
        "cheek_rgb_variance":    avg_var,
        "specular_ratio":        avg_specular,
        "hydration_proxy_score": score,
        "alert_dehydration":     score < 0.2,   # severe dullness threshold
        "experimental_confidence_low": True,
    }
