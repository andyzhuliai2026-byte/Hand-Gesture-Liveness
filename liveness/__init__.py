"""Liveness core (PROMPT.md section 5). Portable core: numpy + stdlib only.

M2: score() runs the 5.5 quality gate and HOLD_BROKEN for real; 5.1-5.4 are placeholders (1.0) until M3.
"""
from __future__ import annotations

from dataclasses import dataclass

from liveness import fusion, quality
from liveness.frame import Sequence
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
    qc = cfg["quality"]
    q = quality.assess(seq, qc)
    hold = seq.hold_match_ratio
    insufficient = ((["LOW_QUALITY"] if q.qualified_ratio < qc["qualified_ratio_min"]
                     or q.n_qualified < qc["qualified_min_frames"] else [])
                    + (["TWO_HANDS"] if q.two_hands_ratio >= qc["two_hands_ratio_max"] else [])
                    + (["HOLD_BROKEN"] if hold is None or hold < cfg["session"]["hold_match_min"] else []))
    scores = {"temporal": 1.0, "nonrigid": 1.0, "planarity": 1.0, "texture": 1.0}  # ponytail: M2 placeholders, M3 replaces
    vetoes: list[str] = []
    verdict, total, reasons = fusion.fuse(scores, insufficient, vetoes, cfg["fusion"])
    debug = {"qualified_ratio": q.qualified_ratio, "n_qualified": q.n_qualified, "n_window": q.n_window,
             "two_hands_ratio": q.two_hands_ratio, "hold_match_ratio": hold, "placeholder_scores": True}
    return LivenessResult(verdict, total, scores, reasons, debug)
