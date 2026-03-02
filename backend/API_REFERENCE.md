# Neuro-Vitals API — Integration Reference v2.0

## Base URL
```
https://<your-render-url>.onrender.com
```

---

## 1. Identity Resolution (Landing Page)

**Frontend action:** Capture one passive frame on page load.

```http
POST /api/v1/identity/match
Content-Type: application/json

{
  "frame_b64": "<base64 JPEG>"
}
```

**Response:**
```json
{
  "face_id":    "uuid-string",
  "session_id": "uuid-string",
  "status":     "matched" | "new_identity" | "no_face_detected",
  "confidence": 0.92,
  "profile":    { ... }
}
```

> Store `face_id` and `session_id`.  
> `session_id` is the key for the WebSocket and all subsequent calls.

---

## 2. Intake Form (Full System Analysis only)

```http
POST /api/v1/identity/{face_id}/profile
Content-Type: application/json

{
  "name":         "Jane Doe",
  "age":          34,
  "sex":          "female",
  "height_cm":    165.0,
  "weight_kg":    62.0,
  "dominant_hand": "right"
}
```

---

## 3. Live Analysis — WebSocket

```
ws://<host>/api/v1/analyze-stream
```

### Connection flow
```
Frontend                     Backend
   |                            |
   |─── open WebSocket ────────>|
   |─── send frame JSON ───────>|  (triggers session init)
   |<── LiveMetricsPayload ─────|
   |─── send frame JSON ───────>|  (repeat at camera fps)
   |<── LiveMetricsPayload ─────|  (every frame)
   |    ...                     |
   |    [60 seconds elapsed]    |
   |<── { "status": "test_complete", biomarkers, risk_report } ──|
   |─── close ─────────────────>|
```

### Outgoing frame JSON
```json
{
  "frame_b64":    "<base64 JPEG / PNG>",
  "frame_index":  42,
  "timestamp_ms": 1700000000000,
  "module":       "face",
  "session_id":   "<session_id from /identity/match>",
  "modules":      ["face", "body"]
}
```

| `module` value | Analyser invoked |
|---|---|
| `face` | Face Mesh + rPPG + EAR + Skin proxy |
| `body` | Pose + Gait + Tremor + Posture |
| `face_3d` | Face Mesh + 3D Asymmetry + Stress |

### Incoming LiveMetricsPayload
```json
{
  "session_id":             "...",
  "status":                 "processing",
  "elapsed_sec":            12.3,
  "frames_processed":       369,

  "heart_rate_bpm":         72.1,
  "hrv_rmssd_ms":           38.4,
  "spo2_estimate_pct":      97.2,
  "respiratory_rate_bpm":   16.0,

  "ear_average":            0.31,
  "blink_rate_per_min":     18.2,
  "fatigue_score":          0.12,

  "posture_score":          0.88,
  "balance_score":          0.91,
  "tremor_hz":              null,

  "facial_asymmetry_score": 0.07,
  "stress_structural_score": 0.14,

  "hydration_proxy_score":  0.72,
  "alert_dehydration":      false,

  "pulse_wave_samples":     [0.01, -0.02, 0.05, ...],
  "rppg_quality_score":     0.74,
  "warning":                null
}
```

**`pulse_wave_samples`** is the raw filtered rPPG array.  
Render this directly with Chart.js / Recharts — no server-side image.

### test_complete payload
```json
{
  "session_id":      "...",
  "status":          "test_complete",
  "elapsed_sec":     60.0,
  "frames_processed": 1800,
  "biomarkers":      { ... full flat dict ... },
  "risk_report": {
    "signals": [
      {
        "domain":       "cardiovascular",
        "label":        "Cardiovascular Risk",
        "risk_level":   "low",
        "probability":  0.12,
        "confidence_score": 0.87,
        "uncertainty_flags": [],
        "contributing_biomarkers": ["heart_rate_bpm", "hrv_rmssd_ms"]
      }
    ],
    "overall_wellness_score": 82.4,
    "data_completeness_pct":  75.0
  },
  "pulse_wave_samples": [...]
}
```

---

## 4. Voice Analysis

Send after the voice test recording finishes.

```http
POST /api/v1/voice/analyze?session_id=<session_id>
Content-Type: multipart/form-data

audio: <WAV/OGG/FLAC/MP3 file>
```

**Response:**
```json
{
  "jitter_pct":               0.48,
  "shimmer_pct":              1.92,
  "hnr_db":                   22.1,
  "mpt_sec":                  19.3,
  "f0_mean_hz":               205.4,
  "f0_std_hz":                18.2,
  "speech_rate_syl_per_sec":  4.1,
  "pause_ratio":              0.22,
  "audio_duration_sec":       23.0
}
```

---

## 5. Get Session Results

```http
GET /api/v1/session/{session_id}/results
```

Call after `test_complete` to fetch the persisted full report.

---

## 6. Refresh Risk Report

Useful after voice metrics are added to a face session:

```http
POST /api/v1/session/{session_id}/risk
```

---

## 7. AI Summary Context

Fetch structured context for the embedded AI panel:

```http
GET /api/v1/session/{session_id}/ai-summary
```

Feed the response JSON as context to the Anthropic API  
(claude-sonnet-4-20250514 recommended) for natural language explanations.

---

## Frontend Responsibilities (not handled by backend)

| Feature | Owner |
|---|---|
| 3-second countdown before test | Frontend |
| "Begin Next Test" button | Frontend |
| Graph rendering of `pulse_wave_samples` | Frontend (Chart.js / Recharts) |
| Camera capture (getUserMedia) | Frontend |
| Audio recording | Frontend |
| Results page UI | Frontend |

---

## Biomarker Reference

| Field | Normal Range | Unit | Source Module |
|---|---|---|---|
| heart_rate_bpm | 60–100 | BPM | face (rPPG) |
| hrv_rmssd_ms | 20–80 | ms | face (rPPG) |
| spo2_estimate_pct | 95–100 | % | face (rPPG) ⚠️ estimate |
| respiratory_rate_bpm | 12–20 | BPM | face (rPPG) |
| ear_average | 0.25–0.40 | — | face (FaceMesh) |
| fatigue_score | 0–0.3 | 0-1 | face (FaceMesh) |
| jitter_pct | 0–1.0 | % | voice |
| shimmer_pct | 0–3.0 | % | voice |
| hnr_db | 15–30 | dB | voice |
| mpt_sec | 15–25 | s | voice |
| facial_asymmetry_score | 0–0.2 | 0-1 | face_3d |
| gait_symmetry_pct | 90–100 | % | body |
| balance_score | 0.8–1.0 | 0-1 | body |
| tremor_amplitude | 0–1.0 | px | body |
| hydration_proxy_score | 0.4–1.0 | 0-1 | face ⚠️ experimental |

---

## Risk Domains

| Domain | Key Biomarkers |
|---|---|
| cardiovascular | HR, HRV, SpO₂ |
| neurological | Jitter, Shimmer, Asymmetry, Tremor |
| respiratory | RR, SpO₂, MPT |
| neuromuscular | Gait symmetry, Balance, Tremor |
| fatigue_cognitive | EAR, Fatigue score, Blink rate |
| psychometric | Stress score, Emotional load, Muscle tone |

Each signal includes:
- `risk_level`: `low` / `moderate` / `high` / `unknown`
- `probability`: 0.0–1.0
- `confidence_score`: 0.0–1.0 (how much data backed it)
- `uncertainty_flags`: list of missing or low-quality inputs
