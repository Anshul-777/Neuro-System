"""
identity_manager.py
Face identity via normalised MediaPipe landmark embedding.

CRITICAL FIX: FaceMesh created inside extract_embedding() on first call,
NOT at module level.
"""

from __future__ import annotations

import json, math, os, time, uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

MATCH_THRESHOLD = 0.88
PERSIST_PATH = Path(os.getenv("IDENTITY_STORE_PATH", "/tmp/neuro_vitals_ids.json"))

# 68-point subset of MediaPipe 468 landmarks
KEY_LM = list(dict.fromkeys([
    10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,
    400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,
    103,67,109,70,63,105,66,107,336,296,334,293,300,168,197,195,5,4,
    45,220,115,49,131,134,51,33,160,158,133,153,144,362,385,387,263,
    373,380,61,185,40,39,37,0,267,270,409,291,84,17,314,405,
]))

# Module-level lazy singleton for the FaceMesh used in identity
_fm_identity = None

def _get_fm():
    global _fm_identity
    if _fm_identity is None:
        import mediapipe as mp
        _fm_identity = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        )
    return _fm_identity


def _embed(landmarks, h: int, w: int) -> np.ndarray:
    lm  = landmarks.landmark
    pts = np.array([[lm[i].x*w, lm[i].y*h] for i in KEY_LM], dtype=np.float64)
    nose = np.array([lm[1].x*w, lm[1].y*h])
    pts -= nose
    ied = math.sqrt(((lm[33].x-lm[263].x)*w)**2 + ((lm[33].y-lm[263].y)*h)**2)
    if ied > 1e-3:
        pts /= ied
    v = pts.flatten()
    n = np.linalg.norm(v)
    return v / (n + 1e-9)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))


def extract_embedding(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    h, w = frame_bgr.shape[:2]
    rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res  = _get_fm().process(rgb)
    if not res.multi_face_landmarks:
        return None
    return _embed(res.multi_face_landmarks[0], h, w)


class IdentityManager:
    def __init__(self, path: Path = PERSIST_PATH):
        self._path  = path
        self._store: Dict[str, Dict] = {}
        self._load()

    def match_or_create(self, frame_bgr: np.ndarray,
                        threshold: float = MATCH_THRESHOLD) -> Dict:
        emb = extract_embedding(frame_bgr)
        if emb is None:
            return {"face_id": None, "status": "no_face_detected", "confidence": 0.0}

        best_id, best_sim = None, -1.0
        for fid, entry in self._store.items():
            s = cosine_sim(emb, np.array(entry["embedding"]))
            if s > best_sim:
                best_sim = s; best_id = fid

        if best_sim >= threshold and best_id:
            e = self._store[best_id]
            n = e["seen_count"]
            me = (np.array(e["embedding"])*n + emb)/(n+1)
            me /= np.linalg.norm(me)+1e-9
            e["embedding"]  = me.tolist()
            e["seen_count"] = n+1
            e["last_seen"]  = time.time()
            self._save()
            return {"face_id": best_id, "status": "matched", "confidence": best_sim}

        fid = str(uuid.uuid4())
        self._store[fid] = {"embedding": emb.tolist(), "seen_count": 1,
                            "created_at": time.time(), "last_seen": time.time(), "profile": {}}
        self._save()
        return {"face_id": fid, "status": "new_identity", "confidence": 0.75}

    def update_profile(self, face_id: str, data: Dict) -> bool:
        if face_id not in self._store: return False
        self._store[face_id]["profile"].update(data)
        self._save(); return True

    def get_profile(self, face_id: str) -> Optional[Dict]:
        e = self._store.get(face_id)
        if e is None: return None
        return {"face_id": face_id, "created_at": e["created_at"],
                "last_seen": e["last_seen"], **e.get("profile", {})}

    def _load(self):
        try:
            if self._path.exists():
                self._store = json.loads(self._path.read_text())
        except Exception:
            self._store = {}

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._store))
        except Exception:
            pass


_manager: Optional[IdentityManager] = None

def get_identity_manager() -> IdentityManager:
    global _manager
    if _manager is None:
        _manager = IdentityManager()
    return _manager
