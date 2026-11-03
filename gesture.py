"""Hand gesture classification from MediaPipe world landmarks (PROMPT.md 4, 5.1, dev-1).

Portable core: numpy + stdlib + liveness.frame + gesture_templates only.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

import gesture_templates as gt
from liveness.frame import NONE, UNKNOWN

_EPS = 1e-9  # palm width below this is degenerate


@dataclass(frozen=True)
class HandPose:
    label: str
    extended: tuple[bool, bool, bool, bool, bool]  # thumb..pinky
    e: np.ndarray                                  # (5,) in [0,1], NaN when no hand
    pip_deg: np.ndarray                            # (4,) index..pinky, degrees
    thumb_wrist_ratio: float                       # |P4-P0| / palm_width
    thumb_cross_cos: float                         # cos(P4-P2, P17-P5)
    tip_gap_ratio: float                           # |P4-P8| / palm_width
    palm_width: float                              # |P5-P17|, metres


def _empty(label: str, palm_width: float) -> HandPose:
    return HandPose(label, (False,) * 5, np.full(5, np.nan), np.full(4, np.nan),
                    np.nan, np.nan, np.nan, palm_width)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def analyze(lm_world: np.ndarray, gcfg: dict) -> HandPose:
    p = np.asarray(lm_world, dtype=np.float64)
    if np.isnan(p).any():
        return _empty(NONE, np.nan)
    palm_width = float(np.linalg.norm(p[5] - p[17]))
    if palm_width <= _EPS:
        return _empty(UNKNOWN, np.nan)

    pip_deg = np.array([_angle_deg(p[f[0]] - p[f[1]], p[f[2]] - p[f[1]]) for f in gt.FINGERS[1:]])
    thumb_ratio = float(np.linalg.norm(p[4] - p[0]) / palm_width)
    a, b = p[4] - p[2], p[17] - p[5]
    cross_cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    tip_gap = float(np.linalg.norm(p[4] - p[8]) / palm_width)

    ext = (thumb_ratio > gcfg["thumb_ratio"] and cross_cos < gcfg["thumb_cross_cos"],
           *(bool(d > gcfg["pip_extended_deg"]) for d in pip_deg))
    lo, hi = gcfg["ext_pip_deg"]
    tlo, thi = gcfg["ext_thumb_ratio"]
    e = np.clip(np.concatenate(([(thumb_ratio - tlo) / (thi - tlo)], (pip_deg - lo) / (hi - lo))), 0.0, 1.0)

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
    return HandPose(label, tuple(bool(x) for x in ext), e, pip_deg, thumb_ratio, cross_cos, tip_gap, palm_width)


def classify(lm_world: np.ndarray, gcfg: dict) -> str:
    return analyze(lm_world, gcfg).label


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
