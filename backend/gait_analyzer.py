"""
gait_analyzer.py
Neuro-motor biomarkers via MediaPipe Pose.

CRITICAL FIX: Pose model created inside BodyAnalyzer.__init__(),
NOT at module level. Module-level instantiation crashed Render.
"""

from __future__ import annotations

import collections
import math
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import signal as sp_signal

# Pose landmark indices
LM_L_EYE=2; LM_R_EYE=5
LM_L_SH=11; LM_R_SH=12
LM_L_WRIST=15; LM_R_WRIST=16
LM_L_HIP=23; LM_R_HIP=24
LM_L_KNEE=25; LM_R_KNEE=26
LM_L_ANKLE=27; LM_R_ANKLE=28


def _d2d(a,b,h,w): return math.sqrt(((a.x-b.x)*w)**2+((a.y-b.y)*h)**2)


# ── Posture ───────────────────────────────────────────────────────────────

def analyze_posture(lm, h: int, w: int) -> Dict:
    pts = lm.landmark
    head_tilt = math.degrees(math.atan2(
        (pts[LM_L_EYE].y-pts[LM_R_EYE].y)*h,
        (pts[LM_L_EYE].x-pts[LM_R_EYE].x)*w))
    sh_asym = abs(pts[LM_L_SH].y - pts[LM_R_SH].y)*h
    sh_mid_x = (pts[LM_L_SH].x + pts[LM_R_SH].x)/2
    hip_mid_x= (pts[LM_L_HIP].x+ pts[LM_R_HIP].x)/2
    spinal   = abs(sh_mid_x - hip_mid_x)*w
    ear_mid_x= (pts[LM_L_EYE].x+pts[LM_R_EYE].x)/2
    fhp      = (ear_mid_x - sh_mid_x)*w
    pelvic   = math.degrees(math.atan2(
        (pts[LM_L_HIP].y-pts[LM_R_HIP].y)*h,
        (pts[LM_L_HIP].x-pts[LM_R_HIP].x)*w))
    return {
        "head_tilt_deg":            float(head_tilt),
        "shoulder_asymmetry_deg":   float(sh_asym),
        "spinal_lateral_deviation": float(spinal),
        "forward_head_posture_mm":  float(fhp),
        "pelvic_tilt_deg":          float(pelvic),
        "cervical_score":  float(np.clip(1-abs(head_tilt)/20,  0,1)),
        "thoracic_score":  float(np.clip(1-spinal/(w*0.05),    0,1)),
        "pelvic_score":    float(np.clip(1-abs(pelvic)/15,     0,1)),
    }


# ── Tremor ────────────────────────────────────────────────────────────────

class TremorDetector:
    BUF = 300
    def __init__(self, fps=30.0):
        self.fps = fps
        self._lx: collections.deque = collections.deque(maxlen=self.BUF)
        self._ly: collections.deque = collections.deque(maxlen=self.BUF)
        self._rx: collections.deque = collections.deque(maxlen=self.BUF)
        self._ry: collections.deque = collections.deque(maxlen=self.BUF)
        self._n=0; self._cached={}

    def update(self, lm, h, w):
        pts=lm.landmark
        self._lx.append(pts[LM_L_WRIST].x*w); self._ly.append(pts[LM_L_WRIST].y*h)
        self._rx.append(pts[LM_R_WRIST].x*w); self._ry.append(pts[LM_R_WRIST].y*h)
        self._n+=1
        if self._n%30==0: self._recompute()

    def _recompute(self):
        if len(self._lx)<60: return
        freqs,amps=[],[]
        for buf in [self._lx,self._ly,self._rx,self._ry]:
            arr=sp_signal.detrend(np.array(buf))
            b,a=sp_signal.butter(2,2.0/(self.fps/2),btype="high")
            f=sp_signal.filtfilt(b,a,arr)
            fx=np.fft.rfftfreq(len(f),1/self.fps)
            pw=np.abs(np.fft.rfft(f))**2
            m=(fx>=2)&(fx<=12)
            if np.any(m):
                freqs.append(float(fx[m][np.argmax(pw[m])]))
                amps.append(float(np.sqrt(np.mean(f**2))))
        if not freqs: return
        amp=float(np.mean(amps))
        self._cached={"dominant_tremor_hz":float(np.mean(freqs)),
                      "tremor_amplitude":amp,
                      "tremor_severity":"none" if amp<1 else "mild" if amp<3 else "moderate" if amp<7 else "severe"}

    def get(self):
        return self._cached or {"dominant_tremor_hz":None,"tremor_amplitude":None,"tremor_severity":None}


# ── Gait ──────────────────────────────────────────────────────────────────

class GaitAnalyzer:
    BUF=300
    def __init__(self,fps=30.0):
        self.fps=fps
        self._lay: collections.deque = collections.deque(maxlen=self.BUF)
        self._ray: collections.deque = collections.deque(maxlen=self.BUF)
        self._lax: collections.deque = collections.deque(maxlen=self.BUF)
        self._rax: collections.deque = collections.deque(maxlen=self.BUF)
        self._hy:  collections.deque = collections.deque(maxlen=self.BUF)
        self._n=0; self._cached={}

    def update(self,lm,h,w):
        pts=lm.landmark
        self._lay.append(pts[LM_L_ANKLE].y*h); self._ray.append(pts[LM_R_ANKLE].y*h)
        self._lax.append(pts[LM_L_ANKLE].x*w); self._rax.append(pts[LM_R_ANKLE].x*w)
        self._hy.append((pts[LM_L_HIP].y+pts[LM_R_HIP].y)/2*h)
        self._n+=1
        if self._n%30==0: self._recompute(h,w)

    def _recompute(self,h,w):
        if len(self._lay)<60: return
        la=np.array(self._lay); ra=np.array(self._ray)
        lax=np.array(self._lax); rax=np.array(self._rax)
        def steps(y):
            f=sp_signal.savgol_filter(y,11,3)
            p,_=sp_signal.find_peaks(-f,distance=int(self.fps*0.3))
            return p
        ls=steps(la); rs=steps(ra)
        total=len(ls)+len(rs)
        if total<2: return
        dur=len(la)/self.fps
        cadence=total/dur*60
        stride=(np.std(lax)+np.std(rax))
        sw=float(np.mean(np.abs(lax-rax)))
        sym=None
        if len(ls)>=2 and len(rs)>=2:
            li=np.diff(ls)/self.fps; ri=np.diff(rs)/self.fps
            sym=float(min(np.mean(li),np.mean(ri))/max(np.mean(li),np.mean(ri))*100)
        hy=np.array(self._hy)
        hvar=float(np.var(sp_signal.detrend(hy))) if len(hy)>10 else 0
        bal=float(np.clip(1-hvar/(h*0.02)**2,0,1))
        vel=float(stride*cadence/60) if cadence>0 else None
        self._cached={"stride_length_cm":float(stride),"cadence_steps_per_min":float(cadence),
                      "gait_symmetry_pct":sym,"balance_score":bal,
                      "velocity_cm_per_sec":vel,"step_width_cm":sw}

    def get(self):
        return self._cached or {"stride_length_cm":None,"cadence_steps_per_min":None,
                                "gait_symmetry_pct":None,"balance_score":0.0,
                                "velocity_cm_per_sec":None,"step_width_cm":None}


# ── BodyAnalyzer ──────────────────────────────────────────────────────────

class BodyAnalyzer:
    """
    Lazy MediaPipe Pose init inside __init__ — safe for cloud import.
    """
    def __init__(self, fps: float = 30.0):
        import mediapipe as mp
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._tremor  = TremorDetector(fps=fps)
        self._gait    = GaitAnalyzer(fps=fps)
        self._posture_hist: List[Dict] = []
        self._n = 0

    def process_frame(self, frame: np.ndarray,
                      timestamp_ms: float = 0.0) -> Dict:
        h, w = frame.shape[:2]
        self._n += 1
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return {"landmarks_found": False, "frames_processed": self._n}
        lm = result.pose_landmarks
        posture = analyze_posture(lm, h, w)
        self._posture_hist.append(posture)
        if len(self._posture_hist) > 300:
            self._posture_hist = self._posture_hist[-300:]
        self._tremor.update(lm, h, w)
        self._gait.update(lm, h, w)
        tm = self._tremor.get(); gm = self._gait.get()
        cs=np.mean([p["cervical_score"]  for p in self._posture_hist])
        ts=np.mean([p["thoracic_score"]  for p in self._posture_hist])
        ps=np.mean([p["pelvic_score"]    for p in self._posture_hist])
        return {
            "landmarks_found":True,"frames_processed":self._n,
            **posture,
            "cervical_score":float(cs),"thoracic_score":float(ts),"pelvic_score":float(ps),
            **tm, **gm,
        }

    def get_final_summary(self) -> Dict:
        if not self._posture_hist: return {}
        return {
            "head_tilt_deg":         float(np.mean([p["head_tilt_deg"]         for p in self._posture_hist])),
            "shoulder_asymmetry_deg":float(np.mean([p["shoulder_asymmetry_deg"]for p in self._posture_hist])),
            "cervical_score":        float(np.mean([p["cervical_score"]         for p in self._posture_hist])),
            "thoracic_score":        float(np.mean([p["thoracic_score"]         for p in self._posture_hist])),
            "pelvic_score":          float(np.mean([p["pelvic_score"]           for p in self._posture_hist])),
            **self._tremor.get(), **self._gait.get(),
        }

    def reset(self):
        self._tremor=TremorDetector(); self._gait=GaitAnalyzer()
        self._posture_hist.clear(); self._n=0

    def __del__(self):
        try: self._pose.close()
        except Exception: pass
