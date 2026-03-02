"""
identity_manager.py
Lightweight face identity system using MediaPipe Face Mesh landmark embeddings.

Design rationale:
  - Zero external model downloads (no TensorFlow / InsightFace on startup)
  - Uses a 68-key-point subset of the 468 MediaPipe landmarks, normalised
    to a canonical pose (nose tip at origin, inter-eye distance = 1.0)
  - Cosine similarity matching; threshold tunable
  - In-memory store with optional JSON persistence

For production upgrade: swap _embed_from_landmarks() with an InsightFace
ArcFace embedding (onnxruntime, ~150 MB model) for much higher accuracy.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

MATCH_THRESHOLD  = 0.88        # cosine similarity threshold for match
PERSIST_PATH     = Path(os.getenv("IDENTITY_STORE_PATH", "/tmp/neuro_vitals_identities.json"))

# 68-point subset of MediaPipe 468 landmarks (mirrors dlib's 68-point model)
KEY_LANDMARKS = [
    # Jaw line
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
    # Eyebrows
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    # Nose
    168, 197, 195, 5, 4, 45, 220, 115, 49, 131, 134, 51, 5,
    # Eyes
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
    # Lips
    61, 185, 40, 39, 37, 0, 267, 270, 409, 291, 84, 17, 314, 405,
]
# deduplicate while preserving order
KEY_LANDMARKS = list(dict.fromkeys(KEY_LANDMARKS))

# ─────────────────────────────────────────────
#  MediaPipe Face Detection (lightweight, faster than Face Mesh for identity)
# ─────────────────────────────────────────────

_mp_fm = mp.solutions.face_mesh

_FACE_MESH_ID = _mp_fm.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
)


# ─────────────────────────────────────────────
#  Embedding extraction
# ─────────────────────────────────────────────

def _embed_from_landmarks(landmarks, h: int, w: int) -> np.ndarray:
    """
    Extract a normalised 2D landmark embedding from a Face Mesh result.

    Normalisation:
      1. Select KEY_LANDMARKS subset
      2. Centre on nose tip (landmark 1)
      3. Scale by inter-eye distance (landmarks 33–263)
      4. Flatten to 1-D vector, L2-normalise
    """
    lm  = landmarks.landmark
    pts = np.array([[lm[i].x * w, lm[i].y * h] for i in KEY_LANDMARKS], dtype=np.float64)

    # Centre on nose tip
    nose = np.array([lm[1].x * w, lm[1].y * h])
    pts -= nose

    # Scale to unit inter-eye distance
    l_eye = np.array([lm[33].x * w,  lm[33].y * h])
    r_eye = np.array([lm[263].x * w, lm[263].y * h])
    ied   = np.linalg.norm(r_eye - l_eye)
    if ied > 1e-3:
        pts /= ied

    vec  = pts.flatten()
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-9)


def extract_embedding(image_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    """
    Run face mesh on a BGR image and return (embedding, detection_confidence).
    Returns None if no face is detected.
    """
    h, w = image_bgr.shape[:2]
    rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = _FACE_MESH_ID.process(rgb)

    if not result.multi_face_landmarks:
        return None

    lm = result.multi_face_landmarks[0]
    # Proxy confidence: mean of landmark visibility (MediaPipe doesn't expose
    # a single detection confidence for FaceMesh)
    conf = 0.75   # fixed placeholder

    emb = _embed_from_landmarks(lm, h, w)
    return emb, conf


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ─────────────────────────────────────────────
#  Identity store
# ─────────────────────────────────────────────

class IdentityManager:
    """
    In-memory identity store with JSON persistence.

    Each identity:
      face_id  : UUID string
      embedding: mean of multiple embedding snapshots (improves robustness)
      name     : optional metadata
      ...      : other user profile fields
    """

    def __init__(self, persist_path: Path = PERSIST_PATH):
        self._path = persist_path
        # face_id → {embedding: list, profile: dict, seen_count: int}
        self._store: Dict[str, Dict] = {}
        self._load()

    # ─────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────

    def match_or_create(
        self,
        image_bgr: np.ndarray,
        threshold: float = MATCH_THRESHOLD,
    ) -> Dict:
        """
        Core identity resolution method.

        1. Extract embedding from image.
        2. Search for a matching identity (cosine sim > threshold).
        3. If found → return existing face_id.
        4. If not  → create a new identity.

        Returns:
          {
            "face_id": str,
            "status":  "matched" | "new_identity",
            "confidence": float,
          }
        """
        result = extract_embedding(image_bgr)
        if result is None:
            return {"face_id": None, "status": "no_face_detected", "confidence": 0.0}

        emb, det_conf = result

        # Search existing identities
        best_id, best_sim = self._find_best_match(emb, threshold)

        if best_id is not None:
            # Update running mean embedding for stability
            entry = self._store[best_id]
            old_emb = np.array(entry["embedding"])
            n       = entry["seen_count"]
            new_emb = (old_emb * n + emb) / (n + 1)
            new_emb /= np.linalg.norm(new_emb) + 1e-9
            entry["embedding"]  = new_emb.tolist()
            entry["seen_count"] = n + 1
            entry["last_seen"]  = time.time()
            self._save()
            return {
                "face_id":    best_id,
                "status":     "matched",
                "confidence": best_sim,
            }

        # New identity
        face_id = str(uuid.uuid4())
        self._store[face_id] = {
            "embedding":  emb.tolist(),
            "seen_count": 1,
            "created_at": time.time(),
            "last_seen":  time.time(),
            "profile":    {},
        }
        self._save()
        return {
            "face_id":    face_id,
            "status":     "new_identity",
            "confidence": float(det_conf),
        }

    def update_profile(self, face_id: str, profile_data: Dict) -> bool:
        if face_id not in self._store:
            return False
        self._store[face_id]["profile"].update(profile_data)
        self._save()
        return True

    def get_profile(self, face_id: str) -> Optional[Dict]:
        entry = self._store.get(face_id)
        if entry is None:
            return None
        return {
            "face_id":    face_id,
            "created_at": entry["created_at"],
            "last_seen":  entry["last_seen"],
            "seen_count": entry["seen_count"],
            **entry.get("profile", {}),
        }

    def list_identities(self) -> List[Dict]:
        return [
            {"face_id": fid, **entry.get("profile", {})}
            for fid, entry in self._store.items()
        ]

    def delete_identity(self, face_id: str) -> bool:
        if face_id not in self._store:
            return False
        del self._store[face_id]
        self._save()
        return True

    # ─────────────────────────────────────────
    #  Private helpers
    # ─────────────────────────────────────────

    def _find_best_match(
        self, emb: np.ndarray, threshold: float
    ) -> Tuple[Optional[str], float]:
        best_id, best_sim = None, -1.0
        for fid, entry in self._store.items():
            stored_emb = np.array(entry["embedding"])
            sim = cosine_similarity(emb, stored_emb)
            if sim > best_sim:
                best_sim = sim
                best_id  = fid
        if best_sim >= threshold:
            return best_id, best_sim
        return None, best_sim

    def _load(self):
        try:
            if self._path.exists():
                with open(self._path, "r") as f:
                    self._store = json.load(f)
        except Exception:
            self._store = {}

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as f:
                json.dump(self._store, f)
        except Exception:
            pass   # non-fatal in stateless cloud containers


# Module-level singleton
_identity_manager: Optional[IdentityManager] = None


def get_identity_manager() -> IdentityManager:
    global _identity_manager
    if _identity_manager is None:
        _identity_manager = IdentityManager()
    return _identity_manager
