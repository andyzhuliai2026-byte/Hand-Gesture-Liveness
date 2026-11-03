"""Quality gate (PROMPT.md 5.5): per-frame checks, live hints and the [t0, HOLD end] summary.

Portable core: numpy + stdlib + liveness.frame only. Uses the per-frame ROI fields (5.0), never the 10 Hz blocks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from liveness.frame import Frame, Sequence, palm_width_px

HINT_TWO_HANDS, HINT_CLOSER, HINT_LIGHT, HINT_BRIGHT = "Show only one hand", "Move closer", "More light", "Too bright"


def _palm_small(fr: Frame, qcfg: dict) -> bool:
    return not palm_width_px(fr.lm_px) >= qcfg["min_palm_frac"] * fr.frame_wh[0]  # NaN -> small


def frame_ok(fr: Frame, qcfg: dict) -> bool:
    # comparisons with NaN are False, so a NaN field fails its own check
    return bool(fr.present and not _palm_small(fr, qcfg)
                and qcfg["roi_mean_min"] <= fr.roi_mean <= qcfg["roi_mean_max"]
                and fr.roi_sat_ratio <= qcfg["sat_ratio_max"] and fr.hand_score >= qcfg["hand_score_min"])


def frame_hint(fr: Frame, qcfg: dict) -> str | None:
    if fr.n_hands >= 2:
        return HINT_TWO_HANDS
    if not fr.present:
        return None
    if _palm_small(fr, qcfg):
        return HINT_CLOSER
    if fr.roi_mean < qcfg["roi_mean_min"]:
        return HINT_LIGHT
    if fr.roi_mean > qcfg["roi_mean_max"] or fr.roi_sat_ratio > qcfg["sat_ratio_max"]:
        return HINT_BRIGHT
    return None


@dataclass
class QualitySummary:
    n_window: int
    n_qualified: int
    qualified_ratio: float
    two_hands_ratio: float
    qualified: np.ndarray  # bool per window frame


def assess(seq: Sequence, qcfg: dict) -> QualitySummary:
    t_end = seq.t_hold_end if seq.t_hold_end is not None else (seq.frames[-1].t if seq.frames else seq.t0)
    win = [fr for fr in seq.frames if seq.t0 <= fr.t <= t_end]
    n = len(win)
    qualified = np.array([frame_ok(fr, qcfg) for fr in win], bool)
    n_q = int(qualified.sum())
    two = sum(fr.n_hands >= 2 for fr in win)
    return QualitySummary(n, n_q, n_q / n if n else 0.0, two / n if n else 0.0, qualified)
