"""Synthetic world-landmark templates for the six gestures (PROMPT.md 4, dev-1).

Portable core: numpy + stdlib only. Hand frame: palm in z~0, fingers point -y, thumb on the -x
side, palm faces -z, flexion rotates distal segments toward -z. Units: metres, centred on the
21-point centroid like MediaPipe world landmarks.
"""
from __future__ import annotations

import numpy as np

WRIST = 0
FINGERS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
HAND_CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
                    (17, 18), (18, 19), (19, 20))

_MCP = ((-0.039, -0.085, 0.0), (-0.013, -0.090, 0.0), (0.013, -0.087, 0.0), (0.039, -0.078, 0.0))
_SEG = ((0.040, 0.024, 0.020), (0.045, 0.028, 0.022), (0.042, 0.027, 0.021), (0.033, 0.019, 0.018))
_SPREAD = (-8.0, 0.0, 6.0, 12.0)
_V_SPREAD = {0: -12.0, 1: 4.0}  # finger idx (index, middle) -> abduction in V_SIGN
_EXT, _CURL, _OK_INDEX = (5, 4, 3), (75, 100, 60), (45, 60, 30)
_T1 = np.array([-0.024, -0.020, -0.010])
_T_SEG = (0.040, 0.032, 0.025)
_T_FOLDED = ((-0.045, -0.050, -0.022), (-0.030, -0.072, -0.040), (-0.010, -0.092, -0.052))
_Z = np.array([0.0, 0.0, -1.0])

# gesture -> (index, middle, ring, pinky curled?, thumb mode)
_SPEC = {
    "OPEN_PALM": ((0, 0, 0, 0), "abducted"),
    "FIST": ((1, 1, 1, 1), "folded"),
    "THUMBS_UP": ((1, 1, 1, 1), "radial"),
    "V_SIGN": ((0, 0, 1, 1), "folded"),
    "POINT_UP": ((0, 1, 1, 1), "folded"),
    "OK": ((1, 0, 0, 0), "ok"),
}


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _finger(k: int, flex_deg, spread_deg: float) -> list[np.ndarray]:
    s = np.radians(spread_deg)
    d0 = np.array([np.sin(s), -np.cos(s), 0.0])
    p = np.array(_MCP[k], dtype=np.float64)
    out, phi = [p], 0.0
    for ang, length in zip(flex_deg, _SEG[k]):
        phi += np.radians(ang)
        p = p + length * (np.cos(phi) * d0 + np.sin(phi) * _Z)
        out.append(p)
    return out  # MCP, PIP, DIP, TIP


def _thumb_straight(direction) -> list[np.ndarray]:
    u = _unit(direction)
    p2 = _T1 + _T_SEG[0] * u
    p3 = p2 + _T_SEG[1] * u
    a = np.radians(10.0)  # IP flexes 10 deg toward -z
    p4 = p3 + _T_SEG[2] * (np.cos(a) * u + np.sin(a) * _Z)
    return [_T1, p2, p3, p4]


def _build(gesture: str) -> np.ndarray:
    curled, thumb_mode = _SPEC[gesture]
    pts = np.zeros((21, 3))
    for k in range(4):
        if gesture == "OK" and k == 0:
            flex = _OK_INDEX
        else:
            flex = _CURL if curled[k] else _EXT
        spread = _V_SPREAD.get(k, _SPREAD[k]) if gesture == "V_SIGN" else _SPREAD[k]
        pts[FINGERS[k + 1], :] = _finger(k, flex, spread)
    if thumb_mode == "abducted":
        th = _thumb_straight((-0.65, -0.75, -0.10))
    elif thumb_mode == "radial":
        th = _thumb_straight((-0.95, -0.25, -0.15))
    elif thumb_mode == "folded":
        th = [_T1, *map(np.array, _T_FOLDED)]
    else:  # ok: tip meets the index tip, slight -z bend in between
        p4 = pts[8] + np.array([0.004, 0.004, 0.0])
        d = p4 - _T1
        th = [_T1, _T1 + 0.4 * d + [0, 0, -0.008], _T1 + 0.75 * d + [0, 0, -0.012], p4]
    pts[FINGERS[0], :] = th
    pts -= pts.mean(axis=0)
    return pts


TEMPLATES: dict[str, np.ndarray] = {g: _build(g) for g in _SPEC}
for _a in TEMPLATES.values():
    _a.flags.writeable = False


def template(gesture_id: str) -> np.ndarray:
    return TEMPLATES[gesture_id].copy()
