"""On-screen overlay (PROMPT.md 3.1, 3.2). ASCII text only (Hershey fonts).

ponytail: plain M1-style overlay; the designed HUD replaces this module later, keyed on SessionOutput.
"""
from __future__ import annotations

import cv2
import numpy as np

from gesture_templates import HAND_CONNECTIONS, template
from liveness.frame import Frame, palm_width_px

WIN = "HandLiveness"
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE, _DARK, _GREEN, _CYAN = (255, 255, 255), (20, 20, 20), (80, 255, 80), (255, 220, 60)
_RED, _YELLOW = (60, 60, 255), (0, 220, 255)
_VERDICT_COLOR = {"PASS": _GREEN, "FAIL": _RED, "INSUFFICIENT": _YELLOW}
_ICON = 160


def _text(img, s, org, scale, color=_WHITE, thick=1):
    (tw, th), base = cv2.getTextSize(s, _FONT, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + base + 2), _DARK, -1)
    cv2.putText(img, s, org, _FONT, scale, color, thick, cv2.LINE_AA)


def _center_text(img, s, y, scale, color=_WHITE, thick=1):
    tw = cv2.getTextSize(s, _FONT, scale, thick)[0][0]
    _text(img, s, ((img.shape[1] - tw) // 2, y), scale, color, thick)


def _f(v, fmt):
    return "-" if v is None or not np.isfinite(v) else format(v, fmt)


def _icon(img, target: str, x0: int, y0: int) -> None:
    """Template wireframe (x, y of the world template; fingers point -y = up) fit into a 160x160 box."""
    xy = template(target)[:, :2]
    xy = xy - (xy.min(0) + xy.max(0)) / 2
    xy = xy * ((_ICON - 16) / max(np.ptp(xy, axis=0).max(), 1e-9)) + _ICON / 2
    pts = [(int(round(x0 + x)), int(round(y0 + y))) for x, y in xy]
    cv2.rectangle(img, (x0, y0), (x0 + _ICON, y0 + _ICON), _DARK, -1)
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], _WHITE, 2, cv2.LINE_AA)


def draw(img_bgr: np.ndarray, fr: Frame, pose, out, stats: dict, debug: bool, cfg: dict) -> np.ndarray:
    gcfg, scfg = cfg["gesture"], cfg["session"]
    img = img_bgr.copy()
    h, w = img.shape[:2]
    if fr.present:
        pts = [(int(round(x)), int(round(y))) for x, y, _ in fr.lm_px]
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], _GREEN, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, p, 4, _CYAN, -1, cv2.LINE_AA)
    if out.phase != "RESULT":  # RESULT shows verdict + reason in the centre
        _text(img, out.prompt, (16, 40), 1.0, thick=2)
    if out.phase in ("CHALLENGE", "HOLD") and out.target is not None:
        total = scfg["challenge_timeout_s"] if out.phase == "CHALLENGE" else scfg["hold_s"]
        left = f"{max(total - out.t_in_phase, 0.0):.1f}s"
        tw = cv2.getTextSize(left, _FONT, 1.2, 2)[0][0]
        _text(img, left, (w - tw - 20, 44), 1.2, _YELLOW, 2)
        _center_text(img, out.target, 110, 1.6, _YELLOW, 3)
        x0, y0 = (w - _ICON) // 2, 130
        _icon(img, out.target, x0, y0)
        bar_y = y0 + _ICON + 10
        cv2.rectangle(img, (x0, bar_y), (x0 + _ICON, bar_y + 12), _WHITE, 1)
        cv2.rectangle(img, (x0, bar_y), (x0 + int(round(_ICON * out.stable_progress)), bar_y + 12), _GREEN, -1)
    if out.phase == "RESULT" and out.verdict is not None:
        color = _VERDICT_COLOR[out.verdict]
        _center_text(img, out.verdict, h // 2, 2.5, color, 5)
        _center_text(img, out.reason_text or "", h // 2 + 60, 1.0, color, 2)
    if out.hint:
        _center_text(img, out.hint, h - 60, 1.0, _YELLOW, 2)
    _text(img, f"{stats['fps']:.0f} fps   proc {stats['proc_ms']:.1f} ms", (16, h - 16), 0.7)
    if not debug:
        return img
    ok = fr.present
    q = "-" if out.quality_ok is None else ("ok" if out.quality_ok else "bad")
    lines = [f"raw {fr.raw_label}   smooth {out.smooth_label}",
             f"hands {fr.n_hands}  score {_f(fr.hand_score, '.2f')}",
             f"palm {_f(palm_width_px(fr.lm_px) / w * 100 if ok else None, '.1f')}% of width",
             f"roi mean {_f(fr.roi_mean, '.0f')}  sat {_f(fr.roi_sat_ratio * 100, '.1f')}%",
             f"quality {q}  hint {out.hint or '-'}"]
    if pose is not None:
        e, p, qd = pose.e, pose.pip_deg, pose.pip2d_deg
        lines += [f"e  T{e[0]:.2f} I{e[1]:.2f} M{e[2]:.2f} R{e[3]:.2f} P{e[4]:.2f}",
                  f"pip2d  I{qd[0]:.0f} M{qd[1]:.0f} R{qd[2]:.0f} P{qd[3]:.0f}  (>{gcfg['pip_extended_deg']})",
                  f"pip3d  I{p[0]:.0f} M{p[1]:.0f} R{p[2]:.0f} P{p[3]:.0f}  (>{gcfg['pip3d_min_deg']})",
                  f"thumb tip {pose.thumb_tip_ratio:.2f} (>{gcfg['thumb_tip_ratio']})  "
                  f"wrist {pose.thumb_wrist_ratio:.2f}  cross {pose.thumb_cross_cos:+.2f}  gap {pose.tip_gap_ratio:.2f}"]
    else:
        lines += ["e  -", "pip2d  -", "pip3d  -", "thumb  -"]
    for i, s in enumerate(lines):
        _text(img, s, (16, 76 + 26 * i), 0.6)
    e = pose.e if pose is not None else np.full(5, np.nan)
    for i, name in enumerate(("T", "I", "M", "R", "P")):
        x = w - 160 + i * 28
        cv2.rectangle(img, (x, h - 140), (x + 18, h - 40), _WHITE, 1)
        if np.isfinite(e[i]):
            cv2.rectangle(img, (x, h - 40 - int(round(100 * float(e[i])))), (x + 18, h - 40), _GREEN, -1)
        _text(img, name, (x + 2, h - 18), 0.5)
    return img
