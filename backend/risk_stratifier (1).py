"""
risk_stratifier.py
Multi-modal risk stratification engine.

Correlates patterns across ALL biomarker modules to produce probabilistic
risk signals with confidence scores and uncertainty flags.

Architecture: rule-based heuristics + normalised score fusion.
(A supervised ML model could replace this with enough labelled data.)

Risk domains:
  - cardiovascular
  - neurological
  - respiratory
  - neuromuscular
  - fatigue_cognitive
  - psychometric
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────
#  Reference ranges for normalisation
# ─────────────────────────────────────────────

REFS = {
    # (min_normal, max_normal, lower_is_worse, higher_is_worse)
    "heart_rate_bpm":         (60,  100, True, True),
    "hrv_rmssd_ms":           (20,  80,  True, False),    # low HRV → risk
    "hrv_sdnn_ms":            (30,  100, True, False),
    "respiratory_rate_bpm":   (12,  20,  True, True),
    "spo2_estimate_pct":      (95,  100, True, False),
    "jitter_pct":             (0,   1.0, False, True),    # high jitter → neurological risk
    "shimmer_pct":            (0,   3.0, False, True),
    "hnr_db":                 (15,  30,  True, False),
    "mpt_sec":                (15,  25,  True, False),
    "facial_asymmetry_score": (0,   0.2, False, True),
    "gait_symmetry_pct":      (90, 100,  True, False),
    "balance_score":          (0.8, 1.0, True, False),
    "ear_average":            (0.25, 0.40, True, False),  # very low = fatigue
    "fatigue_score":          (0,   0.3, False, True),
    "stress_structural_score": (0,  0.4, False, True),
    "tremor_amplitude":       (0,   1.0, False, True),
    "hydration_proxy_score":  (0.4, 1.0, True, False),
}


def _risk_score(value: Optional[float], key: str) -> Tuple[float, bool]:
    """
    Normalise a single metric to a risk contribution in [0, 1].
    Returns (risk_contribution, data_available).
    """
    if value is None or math.isnan(value):
        return 0.0, False

    if key not in REFS:
        return 0.0, False

    lo, hi, lower_bad, upper_bad = REFS[key]
    score = 0.0
    if lower_bad and value < lo:
        score = max(score, (lo - value) / max(lo - (lo - lo * 0.5), 1e-6))
    if upper_bad and value > hi:
        score = max(score, (value - hi) / max((hi * 1.5 - hi), 1e-6))

    return float(np.clip(score, 0.0, 1.0)), True


# ─────────────────────────────────────────────
#  Domain risk calculators
# ─────────────────────────────────────────────

def _cardiovascular_risk(biomarkers: Dict) -> Dict:
    keys     = ["heart_rate_bpm", "hrv_rmssd_ms", "hrv_sdnn_ms", "spo2_estimate_pct"]
    weights  = [0.3, 0.35, 0.25, 0.1]
    scores, available, flags = [], [], []

    for k, w in zip(keys, weights):
        v = biomarkers.get(k)
        s, ok = _risk_score(v, k)
        scores.append(s * w)
        available.append(ok)
        if not ok:
            flags.append(f"{k}_missing")

    if not any(available):
        return _unknown_signal("cardiovascular", "Cardiovascular Risk", flags)

    completeness = sum(available) / len(available)
    raw_risk     = sum(scores) / max(sum(w for w, ok in zip(weights, available) if ok), 1e-6)
    # SpO2 hard override
    spo2 = biomarkers.get("spo2_estimate_pct")
    if spo2 is not None and spo2 < 90:
        raw_risk = max(raw_risk, 0.9)
        flags.append("critically_low_spo2")

    return _build_signal("cardiovascular", "Cardiovascular Risk",
                         raw_risk, completeness, flags,
                         ["heart_rate_bpm", "hrv_rmssd_ms", "spo2_estimate_pct"])


def _neurological_risk(biomarkers: Dict) -> Dict:
    keys     = ["jitter_pct", "shimmer_pct", "hnr_db",
                "facial_asymmetry_score", "tremor_amplitude"]
    weights  = [0.25, 0.20, 0.20, 0.20, 0.15]
    scores, available, flags = [], [], []

    for k, w in zip(keys, weights):
        v = biomarkers.get(k)
        s, ok = _risk_score(v, k)
        scores.append(s * w)
        available.append(ok)
        if not ok:
            flags.append(f"{k}_missing")

    if not any(available):
        return _unknown_signal("neurological", "Neurological Risk Signal", flags)

    completeness = sum(available) / len(available)
    raw_risk     = sum(scores) / max(sum(w for w, ok in zip(weights, available) if ok), 1e-6)

    # Tremor severity hard boost
    ts = biomarkers.get("tremor_severity")
    if ts == "severe":
        raw_risk = max(raw_risk, 0.8)
        flags.append("high_tremor_severity")
    elif ts == "moderate":
        raw_risk = max(raw_risk, 0.55)

    return _build_signal("neurological", "Neurological Risk Signal",
                         raw_risk, completeness, flags,
                         ["jitter_pct", "facial_asymmetry_score", "tremor_amplitude"])


def _respiratory_risk(biomarkers: Dict) -> Dict:
    keys    = ["respiratory_rate_bpm", "spo2_estimate_pct", "mpt_sec", "hnr_db"]
    weights = [0.35, 0.40, 0.15, 0.10]
    scores, available, flags = [], [], []

    for k, w in zip(keys, weights):
        v = biomarkers.get(k)
        s, ok = _risk_score(v, k)
        scores.append(s * w)
        available.append(ok)
        if not ok:
            flags.append(f"{k}_missing")

    if not any(available):
        return _unknown_signal("respiratory", "Respiratory Risk Signal", flags)

    completeness = sum(available) / len(available)
    raw_risk     = sum(scores) / max(sum(w for w, ok in zip(weights, available) if ok), 1e-6)

    return _build_signal("respiratory", "Respiratory Risk Signal",
                         raw_risk, completeness, flags,
                         ["respiratory_rate_bpm", "spo2_estimate_pct"])


def _neuromuscular_risk(biomarkers: Dict) -> Dict:
    keys    = ["gait_symmetry_pct", "balance_score", "tremor_amplitude", "stride_length_cm"]
    weights = [0.35, 0.30, 0.25, 0.10]
    scores, available, flags = [], [], []

    for k, w in zip(keys, weights):
        v = biomarkers.get(k)
        s, ok = _risk_score(v, k)
        scores.append(s * w)
        available.append(ok)
        if not ok:
            flags.append(f"{k}_missing")

    if not any(available):
        return _unknown_signal("neuromuscular", "Neuro-Motor Risk Signal", flags)

    completeness = sum(available) / len(available)
    raw_risk     = sum(scores) / max(sum(w for w, ok in zip(weights, available) if ok), 1e-6)

    return _build_signal("neuromuscular", "Neuro-Motor Risk Signal",
                         raw_risk, completeness, flags,
                         ["gait_symmetry_pct", "balance_score", "tremor_amplitude"])


def _fatigue_cognitive_risk(biomarkers: Dict) -> Dict:
    keys    = ["fatigue_score", "ear_average", "hrv_rmssd_ms",
               "blink_rate_per_min", "pause_ratio"]
    weights = [0.30, 0.25, 0.20, 0.15, 0.10]
    scores, available, flags = [], [], []

    # Blink rate: normal ≈ 15-20/min; <5 or >30 flagged
    br = biomarkers.get("blink_rate_per_min")
    if br is not None:
        if br < 5 or br > 30:
            scores.append(0.7 * 0.15)
            available.append(True)
        else:
            scores.append(0.0)
            available.append(True)
        keys    = ["fatigue_score", "ear_average", "hrv_rmssd_ms", "pause_ratio"]
        weights = [0.35, 0.30, 0.25, 0.10]
    
    for k, w in zip(keys, weights):
        v = biomarkers.get(k)
        s, ok = _risk_score(v, k)
        scores.append(s * w)
        available.append(ok)
        if not ok:
            flags.append(f"{k}_missing")

    if not any(available):
        return _unknown_signal("fatigue_cognitive", "Cognitive Fatigue Signal", flags)

    completeness = sum(available) / len(available)
    raw_risk     = sum(scores) / max(sum(w for w, ok in zip(weights, available) if ok), 1e-6)

    return _build_signal("fatigue_cognitive", "Cognitive Fatigue Signal",
                         raw_risk, completeness, flags,
                         ["fatigue_score", "ear_average", "blink_rate_per_min"])


def _psychometric_risk(biomarkers: Dict) -> Dict:
    keys    = ["stress_structural_score", "emotional_load_baseline",
               "muscle_tone_imbalance_score", "facial_asymmetry_score"]
    weights = [0.35, 0.35, 0.15, 0.15]
    scores, available, flags = [], [], []

    for k, w in zip(keys, weights):
        v = biomarkers.get(k)
        s, ok = _risk_score(v, k) if k in REFS else (
            (float(np.clip(v, 0, 1)) * 0.5 if v is not None else 0.0),
            v is not None
        )
        scores.append(s * w)
        available.append(ok)
        if not ok:
            flags.append(f"{k}_missing")

    if not any(available):
        return _unknown_signal("psychometric", "Psychometric / Stress Signal", flags)

    completeness = sum(available) / len(available)
    raw_risk     = sum(scores) / max(sum(w for w, ok in zip(weights, available) if ok), 1e-6)

    return _build_signal("psychometric", "Psychometric / Stress Signal",
                         raw_risk, completeness, flags,
                         ["stress_structural_score", "emotional_load_baseline"])


# ─────────────────────────────────────────────
#  Signal builder helpers
# ─────────────────────────────────────────────

def _risk_level(prob: float) -> str:
    if prob < 0.30:
        return "low"
    elif prob < 0.60:
        return "moderate"
    else:
        return "high"


def _build_signal(
    domain: str,
    label: str,
    probability: float,
    completeness: float,
    flags: List[str],
    contributing: List[str],
) -> Dict:
    confidence = float(completeness * (1.0 - probability * 0.1))  # less certain for high risk
    return {
        "domain":                  domain,
        "label":                   label,
        "risk_level":              _risk_level(probability),
        "probability":             float(np.clip(probability, 0.0, 1.0)),
        "confidence_score":        float(np.clip(confidence, 0.0, 1.0)),
        "uncertainty_flags":       flags,
        "contributing_biomarkers": contributing,
    }


def _unknown_signal(domain: str, label: str, flags: List[str]) -> Dict:
    return {
        "domain":                  domain,
        "label":                   label,
        "risk_level":              "unknown",
        "probability":             0.0,
        "confidence_score":        0.0,
        "uncertainty_flags":       flags + ["insufficient_data"],
        "contributing_biomarkers": [],
    }


# ─────────────────────────────────────────────
#  Overall wellness score
# ─────────────────────────────────────────────

def _overall_wellness(signals: List[Dict], data_completeness: float) -> float:
    """
    Weighted inverse of risk signals → wellness score 0-100.
    """
    if not signals:
        return 50.0
    known  = [s for s in signals if s["risk_level"] != "unknown"]
    if not known:
        return 50.0
    avg_prob = np.mean([s["probability"] for s in known])
    base     = (1.0 - avg_prob) * 100.0
    # Penalise incomplete data
    return float(np.clip(base * (0.5 + 0.5 * data_completeness), 0.0, 100.0))


# ─────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────

def stratify_risk(biomarkers: Dict) -> Dict:
    """
    Accept a flat dict of all biomarker values and return:
      {
        "signals": [...],
        "overall_wellness_score": float,
        "data_completeness_pct": float,
      }
    """
    calculators = [
        _cardiovascular_risk,
        _neurological_risk,
        _respiratory_risk,
        _neuromuscular_risk,
        _fatigue_cognitive_risk,
        _psychometric_risk,
    ]

    signals = [calc(biomarkers) for calc in calculators]

    # Data completeness: fraction of critical biomarkers present
    critical_keys = [
        "heart_rate_bpm", "hrv_rmssd_ms", "spo2_estimate_pct",
        "respiratory_rate_bpm", "jitter_pct", "gait_symmetry_pct",
        "facial_asymmetry_score", "fatigue_score",
    ]
    present = sum(1 for k in critical_keys if biomarkers.get(k) is not None)
    completeness = present / len(critical_keys)

    wellness = _overall_wellness(signals, completeness)

    return {
        "signals":               signals,
        "overall_wellness_score": wellness,
        "data_completeness_pct": completeness * 100.0,
    }
