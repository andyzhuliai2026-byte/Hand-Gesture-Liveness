"""Liveness core (PROMPT.md section 5). Portable core: numpy + stdlib only.

score() always computes every check it can (5.1-5.5) and writes the intermediates to debug; fusion.fuse() decides.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from liveness import fusion, nonrigid, planarity, quality, temporal, texture
from liveness.frame import NONE, Sequence
from liveness.fusion import INSUFFICIENT_ORDER, VETO_ORDER  # noqa: F401  (re-exported contract)


@dataclass
class LivenessResult:
    verdict: str                      # "PASS" | "FAIL" | "INSUFFICIENT"
    total: float | None               # None for INSUFFICIENT and hard vetoes
    scores: dict[str, float | None]   # temporal, nonrigid, planarity, texture
    reasons: list[str]
    debug: dict


REASON_TEXT = {  # PROMPT.md 5.6, verbatim
    "TIMEOUT": "Timeout", "WRONG_GESTURE": "Wrong gesture", "PRERECORDED": "Too fast - pre-recorded?",
    "INSTANT_SWITCH": "Suspected photo switch", "PLANAR": "Suspected flat photo/screen",
    "SCREEN": "Suspected screen replay", "STATIC": "Suspected static image", "LOW_QUALITY": "Retry: low quality",
    "TWO_HANDS": "Show only one hand", "HOLD_BROKEN": "Hold the gesture while tilting",
    "HAND_LOST": "Keep your hand in view", "NO_TRANSITION": "Show your hand before the gesture appears",
    "NEED_ROTATION": "Tilt your hand more and retry"}


def score(seq: Sequence, cfg: dict) -> LivenessResult:
    import gesture  # ponytail: imported here, gesture imports liveness.frame (a top-level import would be circular)
    qc, gc = cfg["quality"], cfg["gesture"]
    q = quality.assess(seq, qc)
    hold = seq.hold_match_ratio
    insufficient = ((["LOW_QUALITY"] if q.qualified_ratio < qc["qualified_ratio_min"]
                     or q.n_qualified < qc["qualified_min_frames"] else [])
                    + (["TWO_HANDS"] if q.two_hands_ratio >= qc["two_hands_ratio_max"] else [])
                    + (["HOLD_BROKEN"] if hold is None or hold < cfg["session"]["hold_match_min"] else []))

    # 5.0: labels and extensions are recomputed here from the landmarks, never taken as input
    fr = seq.frames
    n = len(fr)
    t = np.array([f.t for f in fr], np.float64)
    present = np.array([f.present for f in fr], bool)
    lm_px = np.array([f.lm_px for f in fr], np.float64).reshape(n, 21, 3)
    lm_world = np.array([f.lm_world for f in fr], np.float64).reshape(n, 21, 3)
    poses = [gesture.analyze(f.lm_world, f.lm_px, gc) if f.present else None for f in fr]
    labels = [p.label if p is not None else NONE for p in poses]
    e = np.array([p.e if p is not None else np.full(5, np.nan) for p in poses], np.float64).reshape(n, 5)
    ok = np.array([quality.frame_ok(f, qc) for f in fr], bool)
    in_hold = ((t >= seq.t_hold_start) & (t <= seq.t_hold_end) if seq.t_hold_start is not None
               else np.zeros(n, bool))
    target = seq.targets[0]

    s_t, r_t, d_t = temporal.evaluate(t, present, labels, e, target, seq.t0,
                                      temporal.track_switches(t, present, lm_px, cfg), cfg)
    s_n, _, d_n = nonrigid.evaluate(lm_px[in_hold & ok], cfg)
    s_p, r_p, d_p = planarity.evaluate(t, lm_px, lm_world, labels, in_hold & ok, target, cfg)
    roi_ok = np.isin(seq.roi_t, t[ok]) if len(seq.roi_t) else np.zeros(0, bool)  # blocks of quality-passed frames
    s_x, r_x, d_x = texture.evaluate(seq.roi[roi_ok], cfg)

    scores = {"temporal": s_t, "nonrigid": s_n, "planarity": s_p, "texture": s_x}
    codes = insufficient + r_t + r_p + r_x + (["LOW_QUALITY"] if s_n is None else [])  # 5.2 has no code of its own
    vetoes = [c for c in codes if c in VETO_ORDER]
    verdict, total, reasons = fusion.fuse(scores, [c for c in codes if c in INSUFFICIENT_ORDER], vetoes, cfg["fusion"])
    debug = {"qualified_ratio": q.qualified_ratio, "n_qualified": q.n_qualified, "n_window": q.n_window,
             "two_hands_ratio": q.two_hands_ratio, "hold_match_ratio": hold,
             "t_match": d_t["t_match"], "t_r": d_t["t_r"], "F": d_t["F"], "max_ext_rate_observed": d_t["max_ext_rate_observed"],
             "gap_max_s": d_t["gap_max_s"], "t_switch": d_t["t_switch"], "mid_band_hit": d_t["mid_band_hit"],
             "r": d_n["r"], "n_nonrigid": d_n["n_frames"],
             "R": d_p["R"], "rho": d_p["rho"], "n_pairs": d_p["n_pairs"], "normal_angles": d_p["normal_angles"],
             "max_normal_deg": d_p["max_normal_deg"], "z_spread": d_p["z_spread"],
             "M": d_x["M"], "n_roi": d_x["n_roi"],
             # 7.4 veto hit rates: whether each veto check fired, even when an INSUFFICIENT code decides the
             # verdict or another 5.1 code was returned first (5.1 lists every check that fired)
             "veto_prerecorded": "PRERECORDED" in d_t["fired"], "veto_instant_switch": "INSTANT_SWITCH" in d_t["fired"],
             "veto_planar": "PLANAR" in r_p}
    return LivenessResult(verdict, total, scores, reasons, debug)
