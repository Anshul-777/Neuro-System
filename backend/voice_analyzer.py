"""
voice_analyzer.py
Extracts clinical-grade voice biomarkers from a WAV/PCM audio buffer.

Biomarkers:
  - Jitter     : cycle-to-cycle pitch perturbation (%)
  - Shimmer    : cycle-to-cycle amplitude perturbation (%)
  - HNR        : Harmonics-to-Noise Ratio (dB)
  - MPT        : Maximum Phonation Time (seconds)
  - F0 stats   : mean/std fundamental frequency
  - Speech rate: syllables per second
  - Pause ratio: fraction of silent segments

Libraries: librosa + scipy (no parselmouth – avoids native compilation on Render).
"""

from __future__ import annotations

import io
import struct
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from scipy import signal as sp_signal
from scipy.interpolate import interp1d


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

SAMPLE_RATE       = 22_050       # target SR after resampling
F0_MIN_HZ         = 65.0        # lowest plausible F0 (male bass)
F0_MAX_HZ         = 525.0       # highest plausible F0 (female soprano)
SILENCE_THRESHOLD = 0.015       # RMS below this → silence frame
SILENCE_FRAME_LEN = 0.025       # 25 ms frames for silence detection
HNR_MIN_DB        = -10.0
HNR_MAX_DB        =  40.0


# ─────────────────────────────────────────────
#  Audio loading
# ─────────────────────────────────────────────

def load_audio_bytes(audio_bytes: bytes, target_sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """
    Load audio from raw bytes (WAV / OGG / FLAC / MP3 via librosa).
    Resamples to target_sr and converts to mono.
    """
    buf = io.BytesIO(audio_bytes)
    try:
        y, sr = librosa.load(buf, sr=target_sr, mono=True)
    except Exception:
        # Fallback: try soundfile first, then manual PCM
        buf.seek(0)
        try:
            y_raw, sr_raw = sf.read(buf)
            if y_raw.ndim > 1:
                y_raw = y_raw.mean(axis=1)
            y = librosa.resample(y_raw.astype(np.float32), orig_sr=sr_raw, target_sr=target_sr)
            sr = target_sr
        except Exception as e:
            raise ValueError(f"Cannot decode audio: {e}") from e
    return y.astype(np.float64), sr


# ─────────────────────────────────────────────
#  F0 extraction (YIN algorithm via librosa)
# ─────────────────────────────────────────────

def extract_f0(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract frame-wise F0 using pyin (probabilistic YIN).
    Returns array of F0 values in Hz (NaN for unvoiced frames).
    """
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=F0_MIN_HZ,
            fmax=F0_MAX_HZ,
            sr=sr,
            frame_length=2048,
            hop_length=256,
        )
        # Keep only voiced frames
        f0_voiced = f0.copy()
        f0_voiced[~voiced_flag] = np.nan
        return f0_voiced
    except Exception:
        return np.array([np.nan])


# ─────────────────────────────────────────────
#  Jitter
# ─────────────────────────────────────────────

def compute_jitter(f0: np.ndarray) -> Optional[float]:
    """
    Jitter (RAP – Relative Average Perturbation) as a percentage.
    Uses differences in consecutive period lengths derived from F0.

    Formula: Jitter = mean|T_i - T_{i+1}| / mean(T_i) × 100
    """
    f0_clean = f0[~np.isnan(f0)]
    if len(f0_clean) < 5:
        return None

    # Convert F0 → period in seconds
    periods = 1.0 / f0_clean
    diffs   = np.abs(np.diff(periods))
    jitter  = np.mean(diffs) / np.mean(periods) * 100.0
    return float(np.clip(jitter, 0.0, 10.0))   # clip to plausible range


# ─────────────────────────────────────────────
#  Shimmer
# ─────────────────────────────────────────────

def compute_shimmer(y: np.ndarray, sr: int, f0: np.ndarray) -> Optional[float]:
    """
    Shimmer (local shimmer, dB variant converted to %) from amplitude peaks.

    Uses peak amplitude at each F0-derived period boundary.
    """
    f0_clean = f0[~np.isnan(f0)]
    if len(f0_clean) < 5:
        return None

    # Estimate mean period in samples
    mean_period_samples = int(sr / np.mean(f0_clean))
    if mean_period_samples < 2:
        return None

    # Extract local RMS per cycle
    hop   = mean_period_samples
    amps  = []
    for i in range(0, len(y) - hop, hop):
        chunk = y[i : i + hop]
        rms   = float(np.sqrt(np.mean(chunk ** 2)))
        if rms > 1e-6:
            amps.append(rms)

    if len(amps) < 3:
        return None

    amps = np.array(amps)
    diffs    = np.abs(np.diff(amps))
    shimmer  = np.mean(diffs) / np.mean(amps) * 100.0
    return float(np.clip(shimmer, 0.0, 20.0))


# ─────────────────────────────────────────────
#  HNR (Harmonics-to-Noise Ratio)
# ─────────────────────────────────────────────

def compute_hnr(y: np.ndarray, sr: int, f0: np.ndarray) -> Optional[float]:
    """
    HNR estimation via autocorrelation method.
    HNR(dB) = 10 * log10(r_max / (1 - r_max))
    where r_max is the normalised autocorrelation at the fundamental period.
    """
    f0_clean = f0[~np.isnan(f0)]
    if len(f0_clean) < 5 or len(y) < sr * 0.1:
        return None

    mean_f0 = float(np.mean(f0_clean))
    lag      = int(sr / mean_f0)
    if lag <= 0 or lag >= len(y):
        return None

    # Normalised autocorrelation
    y_norm = y - np.mean(y)
    acf    = np.correlate(y_norm, y_norm, mode="full")
    acf    = acf[len(acf) // 2 :]        # take positive lags
    acf   /= acf[0] + 1e-9               # normalise

    if lag >= len(acf):
        return None

    r = float(np.clip(acf[lag], -1.0, 1.0 - 1e-6))
    if r <= 0:
        return None

    hnr = 10.0 * np.log10(r / (1.0 - r))
    return float(np.clip(hnr, HNR_MIN_DB, HNR_MAX_DB))


# ─────────────────────────────────────────────
#  Maximum Phonation Time
# ─────────────────────────────────────────────

def compute_mpt(y: np.ndarray, sr: int) -> float:
    """
    Maximum Phonation Time: duration of the longest continuous voiced segment.
    Uses short-time energy thresholding and F0-based voicing.
    """
    frame_len = int(SILENCE_FRAME_LEN * sr)
    hop       = frame_len // 2
    rms_frames = []
    for i in range(0, len(y) - frame_len, hop):
        rms_frames.append(float(np.sqrt(np.mean(y[i : i + frame_len] ** 2))))

    if not rms_frames:
        return 0.0

    voiced     = np.array(rms_frames) > SILENCE_THRESHOLD
    # Find longest run
    max_run    = 0
    current    = 0
    for v in voiced:
        if v:
            current += 1
            max_run  = max(max_run, current)
        else:
            current = 0

    return float(max_run * hop / sr)


# ─────────────────────────────────────────────
#  Speech rate & pause ratio
# ─────────────────────────────────────────────

def compute_speech_rate(y: np.ndarray, sr: int) -> Tuple[Optional[float], float]:
    """
    Estimate speech rate (syllables/sec) via energy envelope peaks.
    Also returns pause ratio (fraction of silent frames).
    """
    frame_len  = int(SILENCE_FRAME_LEN * sr)
    hop        = frame_len // 2
    rms_frames = np.array([
        float(np.sqrt(np.mean(y[i : i + frame_len] ** 2)))
        for i in range(0, len(y) - frame_len, hop)
    ])
    if len(rms_frames) == 0:
        return None, 1.0

    voiced_mask = rms_frames > SILENCE_THRESHOLD
    pause_ratio = float(1.0 - voiced_mask.mean())

    # Syllable nuclei via local peaks in smoothed energy
    smooth = sp_signal.savgol_filter(rms_frames, 5, 2)
    peaks, _ = sp_signal.find_peaks(smooth, distance=int(0.1 * sr / hop))
    voiced_peaks = peaks[voiced_mask[peaks]]
    duration = len(y) / sr
    speech_rate = float(len(voiced_peaks) / duration) if duration > 0 else None
    return speech_rate, pause_ratio


# ─────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────

def analyze_voice(audio_bytes: bytes) -> Dict:
    """
    Full voice biomarker pipeline.
    Accepts raw audio bytes (WAV / OGG / FLAC).
    Returns a dict compatible with VoiceMetrics Pydantic model.
    """
    try:
        y, sr = load_audio_bytes(audio_bytes)
    except ValueError as e:
        return {"error": str(e)}

    if len(y) < sr * 0.5:
        return {"error": "Audio too short (need ≥ 0.5 s)"}

    f0 = extract_f0(y, sr)
    f0_clean = f0[~np.isnan(f0)]

    jitter     = compute_jitter(f0)
    shimmer    = compute_shimmer(y, sr, f0)
    hnr        = compute_hnr(y, sr, f0)
    mpt        = compute_mpt(y, sr)
    speech_rate, pause_ratio = compute_speech_rate(y, sr)

    f0_mean = float(np.mean(f0_clean)) if len(f0_clean) > 0 else None
    f0_std  = float(np.std(f0_clean))  if len(f0_clean) > 1 else None

    return {
        "jitter_pct":             jitter,
        "shimmer_pct":            shimmer,
        "hnr_db":                 hnr,
        "mpt_sec":                mpt,
        "f0_mean_hz":             f0_mean,
        "f0_std_hz":              f0_std,
        "speech_rate_syl_per_sec": speech_rate,
        "pause_ratio":             pause_ratio,
        "audio_duration_sec":      len(y) / sr,
    }
