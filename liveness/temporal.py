"""Challenge-response timing gate against photo switching (PROMPT.md 5.0 track switch, 5.1).

All rules run on raw per-frame labels/extensions (never smoothed) and on seconds, not frame
counts, so verdicts hold when the camera drops from 30 to 15 fps. Pure: numpy + stdlib only.
"""
from __future__ import annotations

import numpy as np

from liveness.frame import PALM_CENTER_IDX


def _num(x) -> float | None:
    """JSON-safe float: NaN/inf/None -> None."""
    return float(x) if x is not None and np.isfinite(x) else None


def track_switches(t: np.ndarray, present: np.ndarray, lm_px: np.ndarray, cfg: dict) -> list[float]:
    """Times of track switches (5.0): palm-centre displacement between consecutive detected frames
    > switch_jump x pixel palm width. Consecutive detected frames with absent frames between them is
    the re-entry case (new centre vs last known position), so one rule covers both.
    The reference palm width is the earlier (last known) frame's.
    """
    t = np.asarray(t, np.float64)
    xy = np.asarray(lm_px, np.float64).reshape(-1, 21, 3)[:, :, :2]
    if len(t) < 2:
        return []
    centre = xy[:, list(PALM_CENTER_IDX)].mean(axis=1)                                 # (N, 2)
    width = np.maximum(np.linalg.norm(xy[:, 5] - xy[:, 17], axis=1),
                       np.linalg.norm(xy[:, 0] - xy[:, 9], axis=1))                     # palm_width_px, vectorised
    det = np.flatnonzero(np.asarray(present, bool) & np.isfinite(centre).all(1)
                         & np.isfinite(width) & (width > 0))
    if len(det) < 2:
        return []
    a, b = det[:-1], det[1:]
    jump = np.linalg.norm(centre[b] - centre[a], axis=1) > cfg["temporal"]["switch_jump"] * width[a]
    return [float(x) for x in t[b[jump]]]


def first_match(t: np.ndarray, present: np.ndarray, labels, target: str, t0: float, cfg: dict) -> float | None:
    """t_match (5.1): time of the first frame of a run of >= match_frames consecutive detected frames
    labelled target. A frame without a hand breaks the run, so the confirmation stays tight in time.
    Runs that end before t0 are ignored here (shown during the countdown, then dropped; PLAN dev-9).
    session.py keeps its own first-match counter from the reveal frame on (its t precedes the t0 set
    after waitKey), so a run spanning t0 counts from the reveal frame there but from its start here
    (-> PRERECORDED).
    """
    t = np.asarray(t, np.float64)
    hit = np.asarray(present, bool) & (np.asarray(labels, dtype=object) == target)
    if not hit.any():
        return None
    edges = np.diff(np.concatenate(([0], hit.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1  # inclusive run bounds
    ok = (ends - starts + 1 >= cfg["temporal"]["match_frames"]) & (t[ends] >= t0)
    return float(t[starts[ok][0]]) if ok.any() else None


def evaluate(t: np.ndarray, present: np.ndarray, labels, e: np.ndarray, target: str, t0: float,
             t_switch: list[float], cfg: dict) -> tuple[float | None, list[str], dict]:
    """5.1 gate -> (1.0 PASS | 0.0 FAIL | None INSUFFICIENT, [deciding reason], debug).

    Every check is computed and listed in debug["fired"]; the deciding reason is the first that
    fires in this order: no t_match -> NO_TRANSITION; PRERECORDED (needs only t_match, so it
    stands even when the window evidence is thin); HAND_LOST; NO_TRANSITION; INSTANT_SWITCH.
    Known limits (spec-level, kept as written in 5.1): (a) counts any detected frame in W, so a start
    pose or the target pose at t_match with an F finger in mid_band satisfies it (a photo cut can pass);
    (b) is a per-frame rate and e is in [0, 1], so it is <= 1/dt: max_ext_rate 41 cannot fire below
    ~41 fps and at 45-60 fps it rejects smooth real transitions (PLAN dev-8).
    """
    c = cfg["temporal"]
    t = np.asarray(t, np.float64)
    e = np.asarray(e, np.float64).reshape(-1, 5)
    det = np.asarray(present, bool) & np.isfinite(e).all(1)
    debug = {"t_match": None, "t_r": None, "F": [], "max_ext_rate_observed": None, "gap_max_s": None,
             "t_switch": None, "W": None, "mid_band_hit": None, "fired": []}

    idx = np.flatnonzero(det)
    t_match = first_match(t, det, labels, target, t0, cfg)
    if t_match is None:
        debug["fired"] = ["NO_TRANSITION"]
        return None, ["NO_TRANSITION"], debug

    fired = []
    t_r = t_match - t0
    if t_r < c["prerecorded_s"]:
        fired.append("PRERECORDED")

    # Window W; detected frames before the bounding switch are dropped (switch frame itself kept).
    sw = [s for s in t_switch if s <= t_match]
    t_sw = max(sw) if sw else None
    w0 = max(t0, t_match - c["window_s"], t_sw if t_sw is not None else -np.inf)
    w = idx[(t[idx] >= w0) & (t[idx] <= t_match)]
    debug.update(t_match=t_match, t_r=t_r, t_switch=_num(t_sw), W=[float(w0), t_match])

    if len(w) >= 2:
        dt = np.diff(t[w])
        gap = float(dt.max())
        de = np.abs(np.diff(e[w], axis=0)).max(axis=1)
        ok = dt > 0  # duplicate timestamps carry no rate information
        rate = float((de[ok] / dt[ok]).max()) if ok.any() else 0.0
        F = np.flatnonzero(np.abs(e[w[0]] - e[w[-1]]) >= c["f_delta"])
        lo, hi = c["mid_band"]
        mid = bool(((e[w][:, F] >= lo) & (e[w][:, F] <= hi)).any()) if len(F) else None
        debug.update(gap_max_s=gap, max_ext_rate_observed=rate, F=[int(i) for i in F], mid_band_hit=mid)
        if gap > c["max_gap_s"]:
            fired.append("HAND_LOST")
        if not len(F):
            fired.append("NO_TRANSITION")
        if mid is False or rate > c["max_ext_rate"]:
            fired.append("INSTANT_SWITCH")
    else:  # only the t_match frame (or nothing) in W: no transition observed
        fired.append("NO_TRANSITION")

    debug["fired"] = fired
    for code, score in (("PRERECORDED", 0.0), ("HAND_LOST", None), ("NO_TRANSITION", None),
                        ("INSTANT_SWITCH", 0.0)):
        if code in fired:
            return score, [code], debug
    return 1.0, [], debug
