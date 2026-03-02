"""
analysis_results.py
Pydantic v2 models for all Neuro-Vitals biomarker outputs and WebSocket payloads.
Every field has a sensible default so partial results are always serialisable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import time


# ─────────────────────────────────────────────
#  Sub-models: individual biomarker groups
# ─────────────────────────────────────────────

class CardioRespiratoryMetrics(BaseModel):
    heart_rate_bpm: Optional[float] = None
    hrv_sdnn_ms: Optional[float] = None       # Standard deviation of NN intervals
    hrv_rmssd_ms: Optional[float] = None      # Root-mean-square successive differences
    respiratory_rate_bpm: Optional[float] = None
    spo2_estimate_pct: Optional[float] = None  # labelled as estimate
    bp_systolic_surrogate: Optional[float] = None
    bp_diastolic_surrogate: Optional[float] = None
    pulse_wave_samples: List[float] = Field(default_factory=list)   # raw filtered rPPG for frontend chart
    rppg_quality_score: float = 0.0           # 0-1, SNR proxy


class OcularMetrics(BaseModel):
    ear_left: Optional[float] = None
    ear_right: Optional[float] = None
    ear_average: Optional[float] = None
    blink_count: int = 0
    blink_rate_per_min: Optional[float] = None
    prolonged_blink_count: int = 0            # blink > 300 ms → fatigue marker
    fatigue_score: float = 0.0               # 0-1 composite


class SkinTextureMetrics(BaseModel):
    """
    Experimental proxy for skin hydration / dullness.
    Derived from cheek-ROI RGB variance + specular highlight ratio.
    """
    cheek_rgb_variance: Optional[float] = None
    specular_ratio: Optional[float] = None
    hydration_proxy_score: Optional[float] = None   # 0-1; lower = more dehydrated-looking
    alert_dehydration: bool = False
    experimental_confidence_low: bool = True        # always flagged per spec


class FaceMetrics(BaseModel):
    cardio_respiratory: CardioRespiratoryMetrics = Field(default_factory=CardioRespiratoryMetrics)
    ocular: OcularMetrics = Field(default_factory=OcularMetrics)
    skin_texture: SkinTextureMetrics = Field(default_factory=SkinTextureMetrics)
    frames_processed: int = 0


class PostureMetrics(BaseModel):
    head_tilt_deg: Optional[float] = None
    shoulder_asymmetry_deg: Optional[float] = None
    spinal_lateral_deviation: Optional[float] = None
    forward_head_posture_mm: Optional[float] = None
    pelvic_tilt_deg: Optional[float] = None
    cervical_score: float = 0.0
    thoracic_score: float = 0.0
    pelvic_score: float = 0.0


class GaitMetrics(BaseModel):
    stride_length_cm: Optional[float] = None
    cadence_steps_per_min: Optional[float] = None
    gait_symmetry_pct: Optional[float] = None
    balance_score: float = 0.0
    velocity_cm_per_sec: Optional[float] = None
    step_width_cm: Optional[float] = None


class TremorMetrics(BaseModel):
    dominant_tremor_hz: Optional[float] = None
    tremor_amplitude: Optional[float] = None
    tremor_severity: Optional[str] = None   # "none" | "mild" | "moderate" | "severe"


class BodyMetrics(BaseModel):
    posture: PostureMetrics = Field(default_factory=PostureMetrics)
    gait: GaitMetrics = Field(default_factory=GaitMetrics)
    tremor: TremorMetrics = Field(default_factory=TremorMetrics)
    frames_processed: int = 0


class VoiceMetrics(BaseModel):
    jitter_pct: Optional[float] = None          # Cycle-to-cycle pitch perturbation
    shimmer_pct: Optional[float] = None         # Cycle-to-cycle amplitude perturbation
    hnr_db: Optional[float] = None              # Harmonics-to-Noise Ratio
    mpt_sec: Optional[float] = None             # Maximum Phonation Time
    f0_mean_hz: Optional[float] = None          # Mean fundamental frequency
    f0_std_hz: Optional[float] = None
    speech_rate_syl_per_sec: Optional[float] = None
    pause_ratio: Optional[float] = None


class FaceStructureMetrics(BaseModel):
    facial_asymmetry_score: float = 0.0         # 0-1; higher = more asymmetric
    left_right_ratio: Optional[float] = None
    muscle_tone_imbalance_score: float = 0.0
    stress_structural_score: float = 0.0
    emotional_load_baseline: Optional[float] = None
    landmark_confidence: float = 0.0


# ─────────────────────────────────────────────
#  Risk Stratification
# ─────────────────────────────────────────────

class RiskSignal(BaseModel):
    domain: str                          # e.g. "cardiovascular", "neurological"
    label: str                           # human-readable label
    risk_level: str                      # "low" | "moderate" | "high" | "unknown"
    probability: float                   # 0-1
    confidence_score: float              # 0-1  how much data backed this
    uncertainty_flags: List[str] = Field(default_factory=list)
    contributing_biomarkers: List[str] = Field(default_factory=list)


class RiskReport(BaseModel):
    signals: List[RiskSignal] = Field(default_factory=list)
    overall_wellness_score: float = 0.0   # 0-100
    data_completeness_pct: float = 0.0
    generated_at: float = Field(default_factory=time.time)


# ─────────────────────────────────────────────
#  Session / Identity
# ─────────────────────────────────────────────

class UserProfile(BaseModel):
    face_id: str
    name: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    dominant_hand: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    last_seen: float = Field(default_factory=time.time)


class IntakeForm(BaseModel):
    name: str
    age: int
    sex: str
    height_cm: float
    weight_kg: float
    dominant_hand: str  # "left" | "right" | "ambidextrous"


# ─────────────────────────────────────────────
#  WebSocket Frame Payloads
# ─────────────────────────────────────────────

class FramePayload(BaseModel):
    """
    Incoming WebSocket message from the frontend.
    The frame field carries a base64-encoded JPEG/PNG.
    Audio is sent separately via a dedicated REST endpoint.
    """
    frame_b64: str
    frame_index: int = 0
    timestamp_ms: float = 0.0
    module: str = "face"   # "face" | "body" | "face_3d"
    session_id: str = ""


class LiveMetricsPayload(BaseModel):
    """
    Outgoing WebSocket message to the frontend.
    Contains live metrics + raw arrays for frontend charting.
    """
    session_id: str
    status: str = "processing"          # "processing" | "test_complete" | "error"
    elapsed_sec: float = 0.0
    frames_processed: int = 0

    # live metrics streamed per-frame
    heart_rate_bpm: Optional[float] = None
    hrv_rmssd_ms: Optional[float] = None
    spo2_estimate_pct: Optional[float] = None
    respiratory_rate_bpm: Optional[float] = None
    ear_average: Optional[float] = None
    blink_rate_per_min: Optional[float] = None
    fatigue_score: Optional[float] = None

    # raw arrays for frontend chart rendering (no server-side image)
    pulse_wave_samples: List[float] = Field(default_factory=list)
    hrv_samples: List[float] = Field(default_factory=list)

    # posture / body (when module == "body")
    posture_score: Optional[float] = None
    balance_score: Optional[float] = None
    tremor_hz: Optional[float] = None

    # 3d face
    facial_asymmetry_score: Optional[float] = None
    stress_structural_score: Optional[float] = None

    # skin hydration proxy
    hydration_proxy_score: Optional[float] = None
    alert_dehydration: bool = False

    # quality
    rppg_quality_score: float = 0.0
    warning: Optional[str] = None


# ─────────────────────────────────────────────
#  Full Session Results (stored, returned on completion)
# ─────────────────────────────────────────────

class SessionResults(BaseModel):
    session_id: str
    face_id: Optional[str] = None
    user_profile: Optional[UserProfile] = None
    face_metrics: FaceMetrics = Field(default_factory=FaceMetrics)
    body_metrics: BodyMetrics = Field(default_factory=BodyMetrics)
    voice_metrics: VoiceMetrics = Field(default_factory=VoiceMetrics)
    face_structure_metrics: FaceStructureMetrics = Field(default_factory=FaceStructureMetrics)
    risk_report: RiskReport = Field(default_factory=RiskReport)
    started_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None
    modules_run: List[str] = Field(default_factory=list)
