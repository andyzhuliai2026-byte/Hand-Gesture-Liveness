"""On-screen overlay (PROMPT.md 3.2). ASCII text only (Hershey fonts)."""
from __future__ import annotations

import cv2
import numpy as np

from gesture_templates import HAND_CONNECTIONS
from liveness.frame import Frame, palm_width_px

WIN = "HandLiveness"
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE, _DARK, _GREEN, _CYAN = (255, 255, 255), (20, 20, 20), (80, 255, 80), (255, 220, 60)


def _text(img, s, org, scale, color=_WHITE, thick=1):
    (tw, th), base = cv2.getTextSize(s, _FONT, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + base + 2), _DARK, -1)
    cv2.putText(img, s, org, _FONT, scale, color, thick, cv2.LINE_AA)


def _f(v, fmt):
    return "-" if v is None or not np.isfinite(v) else format(v, fmt)


def draw(img_bgr: np.ndarray, fr: Frame, pose, smooth: str, stats: dict, debug: bool, gcfg: dict) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    if fr.present:
        pts = [(int(round(x)), int(round(y))) for x, y, _ in fr.lm_px]
        for a, b in HAND_CONNECTIONS:
            cv2.line(out, pts[a], pts[b], _GREEN, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(out, p, 4, _CYAN, -1, cv2.LINE_AA)
    _text(out, f"raw {fr.raw_label}   smooth {smooth}", (16, 40), 1.0, thick=2)
    _text(out, f"{stats['fps']:.0f} fps   proc {stats['proc_ms']:.1f} ms", (16, h - 16), 0.7)
    if not debug:
        return out
    ok = fr.present
    lines = [f"hands {fr.n_hands}  score {_f(fr.hand_score, '.2f')}",
             f"palm {_f(palm_width_px(fr.lm_px) / w * 100 if ok else None, '.1f')}% of width",
             f"roi mean {_f(fr.roi_mean, '.0f')}  sat {_f(fr.roi_sat_ratio * 100, '.1f')}%"]
    if pose is not None:
        e, p, q = pose.e, pose.pip_deg, pose.pip2d_deg
        lines += [f"e  T{e[0]:.2f} I{e[1]:.2f} M{e[2]:.2f} R{e[3]:.2f} P{e[4]:.2f}",
                  f"pip2d  I{q[0]:.0f} M{q[1]:.0f} R{q[2]:.0f} P{q[3]:.0f}  (>{gcfg['pip_extended_deg']})",
                  f"pip3d  I{p[0]:.0f} M{p[1]:.0f} R{p[2]:.0f} P{p[3]:.0f}  (>{gcfg['pip3d_min_deg']})",
                  f"thumb tip {pose.thumb_tip_ratio:.2f} (>{gcfg['thumb_tip_ratio']})  "
                  f"wrist {pose.thumb_wrist_ratio:.2f}  cross {pose.thumb_cross_cos:+.2f}  gap {pose.tip_gap_ratio:.2f}"]
    else:
        lines += ["e  -", "pip2d  -", "pip3d  -", "thumb  -"]
    for i, s in enumerate(lines):
        _text(out, s, (16, 76 + 26 * i), 0.6)
    e = pose.e if pose is not None else np.full(5, np.nan)
    for i, name in enumerate(("T", "I", "M", "R", "P")):
        x = w - 160 + i * 28
        cv2.rectangle(out, (x, h - 140), (x + 18, h - 40), _WHITE, 1)
        if np.isfinite(e[i]):
            cv2.rectangle(out, (x, h - 40 - int(round(100 * float(e[i])))), (x + 18, h - 40), _GREEN, -1)
        _text(out, name, (x + 2, h - 18), 0.5)
    return out
