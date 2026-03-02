"""
main.py  –  Neuro-Vitals FastAPI Backend
=========================================

Endpoints
─────────
WebSocket
  WS  /api/v1/analyze-stream          Live multi-modal biomarker streaming
                                       (face / body / face_3d modules)
REST
  POST /api/v1/identity/match         Passive face-first identity check
  POST /api/v1/identity/{face_id}/profile  Save intake form data
  GET  /api/v1/identity/{face_id}/profile  Fetch stored user profile
  POST /api/v1/voice/analyze          Full voice biomarker extraction
  GET  /api/v1/session/{session_id}/results  Final session report
  POST /api/v1/session/{session_id}/risk     Trigger / refresh risk report
  GET  /health                        Liveness probe

Architecture (cloud-safe)
─────────────────────────
  • No cv2.VideoCapture, cv2.imshow, cv2.waitKey
  • No Matplotlib image generation
  • Sliding-window buffers prevent memory leaks
  • WebSocket auto-closes after 60 s and sends {"status": "test_complete"}
  • Raw signal arrays returned for frontend charting
  • Frontend handles: countdown UI, "Begin Next Test" button, results display
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from analysis_results import (
    IntakeForm,
    LiveMetricsPayload,
    RiskReport,
    RiskSignal,
    SessionResults,
    UserProfile,
)
from face_analyzer import FaceAnalyzer
from gait_analyzer import BodyAnalyzer
from identity_manager import get_identity_manager
from risk_stratifier import stratify_risk
from voice_analyzer import analyze_voice


# ─────────────────────────────────────────────
#  Session registry (in-memory; swap for Redis in multi-worker deploy)
# ─────────────────────────────────────────────

SESSION_STORE: Dict[str, Dict] = {}
MAX_SESSION_DURATION = 65          # seconds (hard limit, slightly > 60 s for cleanup)
MAX_SESSIONS         = 100         # evict oldest when full

def _evict_old_sessions():
    if len(SESSION_STORE) < MAX_SESSIONS:
        return
    oldest = sorted(SESSION_STORE.items(), key=lambda kv: kv[1].get("created_at", 0))
    for sid, _ in oldest[: len(SESSION_STORE) - MAX_SESSIONS + 1]:
        SESSION_STORE.pop(sid, None)


def _new_session(session_id: str, modules: list[str], face_id: Optional[str]) -> Dict:
    _evict_old_sessions()
    session = {
        "session_id":    session_id,
        "face_id":       face_id,
        "created_at":    time.time(),
        "modules":       modules,
        "face_analyzer": FaceAnalyzer()   if "face"    in modules else None,
        "body_analyzer": BodyAnalyzer()   if "body"    in modules else None,
        "biomarkers":    {},              # flat dict, updated incrementally
        "frame_count":   0,
        "completed":     False,
    }
    SESSION_STORE[session_id] = session
    return session


# ─────────────────────────────────────────────
#  FastAPI lifespan (warm up MediaPipe models)
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm identity manager (loads JSON store once)
    _ = get_identity_manager()
    print("✅  Neuro-Vitals backend ready")
    yield
    print("🛑  Shutting down")


app = FastAPI(
    title="Neuro-Vitals API",
    description="Multi-modal digital health screening platform",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # restrict to your Loveable domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
#  Frame decoding helper
# ─────────────────────────────────────────────

def decode_frame(frame_b64: str) -> Optional[np.ndarray]:
    """
    Decode a base64-encoded JPEG/PNG frame to a BGR numpy array.
    Returns None on any decoding failure.
    """
    try:
        # Accept "data:image/jpeg;base64,..." or bare base64
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(frame_b64)
        arr       = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


# ─────────────────────────────────────────────
#  WebSocket: /api/v1/analyze-stream
# ─────────────────────────────────────────────

"""
Frontend ↔ Backend WebSocket protocol
──────────────────────────────────────
Incoming JSON from frontend:
  {
    "frame_b64":    "<base64 JPEG>",
    "frame_index":  int,
    "timestamp_ms": float,
    "module":       "face" | "body" | "face_3d",
    "session_id":   str   (from /identity/match)
  }

Outgoing JSON from backend:
  LiveMetricsPayload  (see analysis_results.py)
  or  {"status": "test_complete", ...final_results}
  or  {"status": "error", "detail": str}

The frontend:
  1. Obtains session_id from POST /identity/match
  2. Shows 3-second countdown in its own UI
  3. Opens WebSocket with the session_id
  4. Streams frames for up to 60 seconds
  5. Receives "test_complete" and shows results + "Begin Next Test" button
"""

@app.websocket("/api/v1/analyze-stream")
async def analyze_stream(websocket: WebSocket):
    await websocket.accept()
    session_id    = None
    session       = None
    start_time    = time.time()

    try:
        # ── Receive first message to initialise session ─────────────
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        payload = json.loads(raw)

        session_id = payload.get("session_id")
        module     = payload.get("module", "face")
        modules    = payload.get("modules", [module])

        # Retrieve or create session
        session = SESSION_STORE.get(session_id)
        if session is None:
            session_id = session_id or str(uuid.uuid4())
            session    = _new_session(session_id, modules, face_id=None)

        # Process the first frame immediately (don't waste it)
        await _process_and_reply(websocket, payload, session, start_time)

        # ── Main streaming loop ──────────────────────────────────────
        while True:
            elapsed = time.time() - start_time

            # ── 60-second auto-close ────────────────────────────────
            if elapsed >= 60.0:
                final = _build_final_payload(session, elapsed)
                await websocket.send_json(final)
                break

            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=MAX_SESSION_DURATION - elapsed + 5.0,
                )
            except asyncio.TimeoutError:
                final = _build_final_payload(session, time.time() - start_time)
                await websocket.send_json(final)
                break

            payload = json.loads(raw)
            await _process_and_reply(websocket, payload, session, start_time)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        err = {"status": "error", "detail": str(e), "trace": traceback.format_exc()}
        try:
            await websocket.send_json(err)
        except Exception:
            pass
    finally:
        if session:
            session["completed"] = True
        try:
            await websocket.close()
        except Exception:
            pass


async def _process_and_reply(
    websocket: WebSocket,
    payload: Dict,
    session: Dict,
    start_time: float,
):
    """Decode frame, run analyser, send live metrics."""
    elapsed = time.time() - start_time
    frame   = decode_frame(payload.get("frame_b64", ""))
    ts_ms   = float(payload.get("timestamp_ms", elapsed * 1000))
    module  = payload.get("module", "face")

    if frame is None:
        await websocket.send_json({
            "session_id": session["session_id"],
            "status":     "processing",
            "elapsed_sec": elapsed,
            "warning":    "frame_decode_failed",
        })
        return

    session["frame_count"] += 1
    metrics: Dict = {}

    # ── Route to correct analyser ────────────────────────────────
    if module in ("face", "face_3d") and session.get("face_analyzer"):
        metrics = session["face_analyzer"].process_frame(frame, ts_ms)
    elif module == "body" and session.get("body_analyzer"):
        metrics = session["body_analyzer"].process_frame(frame, ts_ms)

    # ── Merge metrics into flat biomarker dict ────────────────────
    session["biomarkers"].update({k: v for k, v in metrics.items()
                                  if v is not None and not isinstance(v, (list, dict, bool))})

    # ── Build live payload ────────────────────────────────────────
    live: Dict = {
        "session_id":             session["session_id"],
        "status":                 "processing",
        "elapsed_sec":            round(elapsed, 2),
        "frames_processed":       session["frame_count"],

        # Cardio
        "heart_rate_bpm":         metrics.get("heart_rate_bpm"),
        "hrv_rmssd_ms":           metrics.get("hrv_rmssd_ms"),
        "spo2_estimate_pct":      metrics.get("spo2_estimate_pct"),
        "respiratory_rate_bpm":   metrics.get("respiratory_rate_bpm"),

        # EAR / Ocular
        "ear_average":            metrics.get("ear_average"),
        "blink_rate_per_min":     metrics.get("blink_rate_per_min"),
        "fatigue_score":          metrics.get("fatigue_score"),

        # Body
        "posture_score":          _posture_score(metrics),
        "balance_score":          metrics.get("balance_score"),
        "tremor_hz":              metrics.get("dominant_tremor_hz"),

        # 3D Face
        "facial_asymmetry_score": metrics.get("facial_asymmetry_score"),
        "stress_structural_score": metrics.get("stress_structural_score"),

        # Skin proxy
        "hydration_proxy_score":  metrics.get("hydration_proxy_score"),
        "alert_dehydration":      bool(metrics.get("alert_dehydration", False)),

        # Raw arrays for frontend charting (never server-side rendered)
        "pulse_wave_samples":     metrics.get("pulse_wave_samples", []),
        "hrv_samples":            [],

        # Quality
        "rppg_quality_score":     metrics.get("rppg_quality_score", 0.0),
        "warning":                _quality_warning(metrics),
    }
    await websocket.send_json(_sanitise(live))


def _posture_score(m: Dict) -> Optional[float]:
    cs = m.get("cervical_score")
    ts = m.get("thoracic_score")
    ps = m.get("pelvic_score")
    vals = [v for v in [cs, ts, ps] if v is not None]
    return float(np.mean(vals)) if vals else None


def _quality_warning(m: Dict) -> Optional[str]:
    q = m.get("rppg_quality_score", 1.0)
    if q < 0.15:
        return "low_signal_quality_check_lighting"
    if not m.get("landmarks_found", True):
        return "face_not_detected"
    return None


def _build_final_payload(session: Dict, elapsed: float) -> Dict:
    """Build the test_complete payload, run risk stratification."""
    biomarkers = session["biomarkers"].copy()

    # Final summaries from analysers
    if session.get("face_analyzer"):
        face_final = session["face_analyzer"].get_final_summary()
        biomarkers.update(face_final)
    if session.get("body_analyzer"):
        body_final = session["body_analyzer"].get_final_summary()
        biomarkers.update(body_final)

    # Risk stratification
    risk = stratify_risk(biomarkers)

    payload = {
        "session_id":    session["session_id"],
        "status":        "test_complete",
        "elapsed_sec":   round(elapsed, 2),
        "frames_processed": session["frame_count"],
        "biomarkers":    _sanitise(biomarkers),
        "risk_report":   risk,
        # Pulse wave for final chart
        "pulse_wave_samples": biomarkers.get("pulse_wave_samples", []),
    }
    # Persist final results in session store
    session["final_results"] = payload
    return _sanitise(payload)


def _sanitise(obj):
    """Recursively convert numpy types / NaN / Inf to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _sanitise(obj.tolist())
    if isinstance(obj, bool):
        return bool(obj)
    return obj


# ─────────────────────────────────────────────
#  REST: Identity
# ─────────────────────────────────────────────

class IdentityMatchRequest(BaseModel):
    frame_b64: str


@app.post("/api/v1/identity/match")
async def identity_match(req: IdentityMatchRequest):
    """
    Passive face-first identity resolution.
    Frontend sends a single frame on landing.
    Returns face_id + status ("matched" | "new_identity" | "no_face_detected").
    The session_id returned here is used as the WebSocket session_id.
    """
    frame = decode_frame(req.frame_b64)
    if frame is None:
        raise HTTPException(status_code=400, detail="Cannot decode frame")

    mgr    = get_identity_manager()
    result = mgr.match_or_create(frame)

    # Pre-create a session so the client can start immediately
    session_id = str(uuid.uuid4())
    _new_session(session_id, modules=["face", "body", "face_3d"],
                 face_id=result.get("face_id"))

    return {
        **result,
        "session_id":  session_id,
        "profile":     mgr.get_profile(result.get("face_id") or ""),
    }


@app.post("/api/v1/identity/{face_id}/profile")
async def save_profile(face_id: str, intake: IntakeForm):
    """Save medical intake form data to an identity profile."""
    mgr = get_identity_manager()
    ok  = mgr.update_profile(face_id, intake.model_dump())
    if not ok:
        raise HTTPException(status_code=404, detail="Face ID not found")
    return {"status": "saved", "face_id": face_id}


@app.get("/api/v1/identity/{face_id}/profile")
async def get_profile(face_id: str):
    """Retrieve stored user profile."""
    mgr     = get_identity_manager()
    profile = mgr.get_profile(face_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Face ID not found")
    return profile


# ─────────────────────────────────────────────
#  REST: Voice Analysis
# ─────────────────────────────────────────────

@app.post("/api/v1/voice/analyze")
async def voice_analyze(
    audio: UploadFile,
    session_id: Optional[str] = None,
):
    """
    Accept an audio file upload (WAV / OGG / FLAC / MP3).
    Returns full voice biomarker report.
    Optionally merges results into a running session for risk stratification.
    """
    try:
        audio_bytes = await audio.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read audio: {e}")

    metrics = analyze_voice(audio_bytes)

    if "error" in metrics:
        raise HTTPException(status_code=422, detail=metrics["error"])

    # Merge into session if provided
    if session_id and session_id in SESSION_STORE:
        SESSION_STORE[session_id]["biomarkers"].update(
            {k: v for k, v in metrics.items() if v is not None}
        )

    return _sanitise(metrics)


# ─────────────────────────────────────────────
#  REST: Session results
# ─────────────────────────────────────────────

@app.get("/api/v1/session/{session_id}/results")
async def get_session_results(session_id: str):
    """
    Retrieve the final session report (biomarkers + risk signals).
    Available after the WebSocket closes with "test_complete".
    """
    session = SESSION_STORE.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if "final_results" in session:
        return session["final_results"]

    # Session still running – return current snapshot
    biomarkers = session["biomarkers"].copy()
    risk       = stratify_risk(biomarkers)
    return _sanitise({
        "session_id":      session_id,
        "status":          "in_progress",
        "frames_processed": session["frame_count"],
        "biomarkers":      biomarkers,
        "risk_report":     risk,
    })


@app.post("/api/v1/session/{session_id}/risk")
async def refresh_risk(session_id: str):
    """
    Trigger a fresh risk stratification using current biomarkers.
    Useful after voice analysis is merged into a face session.
    """
    session = SESSION_STORE.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    biomarkers = session["biomarkers"].copy()

    if session.get("face_analyzer"):
        biomarkers.update(session["face_analyzer"].get_final_summary())
    if session.get("body_analyzer"):
        biomarkers.update(session["body_analyzer"].get_final_summary())

    risk = stratify_risk(biomarkers)
    return _sanitise({"session_id": session_id, "risk_report": risk})


# ─────────────────────────────────────────────
#  REST: AI Summary integration endpoint
# ─────────────────────────────────────────────

@app.get("/api/v1/session/{session_id}/ai-summary")
async def ai_summary(session_id: str):
    """
    Returns a structured payload for the frontend's AI summary panel.
    The frontend calls the Anthropic API directly (client-side) using this
    structured data as context.  The backend does NOT call the AI.
    """
    session = SESSION_STORE.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    biomarkers = session.get("biomarkers", {})
    risk       = stratify_risk(biomarkers)
    profile    = {}
    if session.get("face_id"):
        profile = get_identity_manager().get_profile(session["face_id"]) or {}

    return _sanitise({
        "session_id":  session_id,
        "profile":     profile,
        "biomarkers":  biomarkers,
        "risk_report": risk,
        "summary_schema_version": "2.0",
    })


# ─────────────────────────────────────────────
#  Health check
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "version":      "2.0.0",
        "active_sessions": len(SESSION_STORE),
        "timestamp":    time.time(),
    }


@app.get("/")
async def root():
    return {
        "name":    "Neuro-Vitals API",
        "version": "2.0.0",
        "docs":    "/docs",
        "health":  "/health",
    }


# ─────────────────────────────────────────────
#  Entrypoint (for local dev / Render start command)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
