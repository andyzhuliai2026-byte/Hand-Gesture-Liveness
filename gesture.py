"""Hand gesture classification from MediaPipe landmarks (PROMPT.md 4, 5.1, dev-1, dev-5).

Portable core: numpy + stdlib + liveness.frame + gesture_templates only.
dev-5: finger extension needs both the 2D (lm_px x, y) and the 3D (lm_world) PIP angle; thumb
extension uses |P4-P6| / palm_width; the OK tip gap is measured in 2D (world tips sit ~0.4 palm
widths apart on a real OK ring, image tips touch).
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

import gesture_templates as gt
from liveness.frame import NONE, UNKNOWN

_EPS = 1e-9  # palm width below this is degenerate

TARGET_E = {  # PROMPT.md 4, thumb..pinky; None = finger not judged (*)
    "OK": (None, None, 1.0, 1.0, 1.0), "OPEN_PALM": (1.0, 1.0, 1.0, 1.0, 1.0), "FIST": (0.0, 0.0, 0.0, 0.0, 0.0),
    "THUMBS_UP": (1.0, 0.0, 0.0, 0.0, 0.0), "V_SIGN": (None, 1.0, 1.0, 0.0, 0.0), "POINT_UP": (None, 1.0, 0.0, 0.0, 0.0)}


@dataclass(frozen=True)
class HandPose:
    label: str
    extended: tuple[bool, bool, bool, bool, bool]  # thumb..pinky
    e: np.ndarray                                  # (5,) in [0,1], NaN when no hand
    pip_deg: np.ndarray                            # (4,) index..pinky, 3D (lm_world), degrees
    pip2d_deg: np.ndarray                          # (4,) index..pinky, 2D (lm_px x, y), degrees
    thumb_tip_ratio: float                         # |P4-P6| / palm_width
    thumb_wrist_ratio: float                       # |P4-P0| / palm_width
    thumb_cross_cos: float                         # cos(P4-P2, P17-P5)
    tip_gap_ratio: float                           # |P4-P8| / palm width, 2D (lm_px x, y)
    palm_width: float                              # |P5-P17|, metres


def _empty(label: str, palm_width: float) -> HandPose:
    return HandPose(label, (False,) * 5, np.full(5, np.nan), np.full(4, np.nan), np.full(4, np.nan),
                    np.nan, np.nan, np.nan, np.nan, palm_width)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _pip_deg(p: np.ndarray) -> np.ndarray:
    return np.array([_angle_deg(p[f[0]] - p[f[1]], p[f[2]] - p[f[1]]) for f in gt.FINGERS[1:]])


def _ramp(x, lohi) -> np.ndarray:
    lo, hi = lohi
    return np.clip((np.asarray(x) - lo) / (hi - lo), 0.0, 1.0)


def analyze(lm_world: np.ndarray, lm_px: np.ndarray, gcfg: dict) -> HandPose:
    p = np.asarray(lm_world, dtype=np.float64)
    q = np.asarray(lm_px, dtype=np.float64)[:, :2]
    if np.isnan(p).any() or np.isnan(q).any():
        return _empty(NONE, np.nan)
    palm_width = float(np.linalg.norm(p[5] - p[17]))
    palm_px = float(np.linalg.norm(q[5] - q[17]))
    if palm_width <= _EPS or palm_px <= _EPS:
        return _empty(UNKNOWN, np.nan)

    pip_deg, pip2d = _pip_deg(p), _pip_deg(q)
    tip_ratio = float(np.linalg.norm(p[4] - p[6]) / palm_width)
    thumb_ratio = float(np.linalg.norm(p[4] - p[0]) / palm_width)
    a, b = p[4] - p[2], p[17] - p[5]
    cross_cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    tip_gap = float(np.linalg.norm(q[4] - q[8]) / palm_px)

    # 2D alone reads a finger curled toward the camera as straight (foreshortening); 3D alone reads
    # real open hands ~25 deg short of 2D, so both must pass.
    ext = (tip_ratio > gcfg["thumb_tip_ratio"],
           *(bool(a > gcfg["pip_extended_deg"] and b > gcfg["pip3d_min_deg"]) for a, b in zip(pip2d, pip_deg)))
    # min(e2d, e3d): with the palm facing the camera a fist's curl is invisible in 2D, only 3D sees it.
    e_fing = np.minimum(_ramp(pip2d, gcfg["ext_pip_deg"]), _ramp(pip_deg, gcfg["ext_pip3d_deg"]))
    e = np.concatenate(([_ramp(tip_ratio, gcfg["ext_thumb_ratio"])], e_fing))

    th, ix, mi, ri, pk = ext
    if mi and ri and pk and tip_gap < gcfg["ok_tip_ratio"]:
        label = "OK"
    elif all(ext):
        label = "OPEN_PALM"
    elif not any(ext):
        label = "FIST"
    elif th and not (ix or mi or ri or pk):
        label = "THUMBS_UP"
    elif ix and mi and not (ri or pk):
        label = "V_SIGN"
    elif ix and not (mi or ri or pk):
        label = "POINT_UP"
    else:
        label = UNKNOWN
    return HandPose(label, tuple(bool(x) for x in ext), e, pip_deg, pip2d, tip_ratio, thumb_ratio, cross_cos,
                    tip_gap, palm_width)


def classify(lm_world: np.ndarray, lm_px: np.ndarray, gcfg: dict) -> str:
    return analyze(lm_world, lm_px, gcfg).label


class LabelSmoother:
    """Majority vote over the last window_s seconds; strict >50% else UNKNOWN."""

    def __init__(self, window_s: float):
        self._window = window_s
        self._buf: deque[tuple[float, str]] = deque()

    def reset(self) -> None:
        self._buf.clear()

    def update(self, t: float, label: str) -> str:
        if self._buf and t < self._buf[-1][0]:
            raise ValueError(f"time went backwards: {t} < {self._buf[-1][0]}")
        self._buf.append((t, label))
        while self._buf[0][0] < t - self._window:
            self._buf.popleft()
        top, n = Counter(l for _, l in self._buf).most_common(1)[0]
        return top if n * 2 > len(self._buf) else UNKNOWN
