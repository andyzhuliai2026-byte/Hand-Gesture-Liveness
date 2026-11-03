"""Fusion and verdict (PROMPT.md 5.6). Portable core: stdlib only."""
from __future__ import annotations

INSUFFICIENT_ORDER = ("LOW_QUALITY", "TWO_HANDS", "HOLD_BROKEN", "HAND_LOST", "NO_TRANSITION", "NEED_ROTATION")
VETO_ORDER = ("PRERECORDED", "INSTANT_SWITCH", "PLANAR")
_LOSS_REASON = {"planarity": "PLANAR", "texture": "SCREEN", "nonrigid": "STATIC"}


def _ordered(codes: list[str], order: tuple[str, ...]) -> list[str]:
    return [c for c in order if c in codes]


def fuse(scores: dict, insufficient: list[str], vetoes: list[str], fcfg: dict) -> tuple[str, float | None, list[str]]:
    """-> (verdict, total, reasons). total is None for INSUFFICIENT and hard vetoes only."""
    if insufficient:
        return "INSUFFICIENT", None, _ordered(insufficient, INSUFFICIENT_ORDER)
    if vetoes:
        return "FAIL", None, _ordered(vetoes, VETO_ORDER)
    if any(scores[k] is None for k in _LOSS_REASON):
        raise ValueError(f"fuse: None score in {scores}")  # ponytail: M3 decides how a missing score maps to a verdict
    w = {k: fcfg[f"w_{k}"] for k in _LOSS_REASON}
    total = sum(w[k] * scores[k] for k in _LOSS_REASON)
    floors = (["PLANAR"] if scores["planarity"] < fcfg["planarity_floor"] else []) + \
             (["SCREEN"] if scores["texture"] < fcfg["texture_floor"] else [])
    if floors:
        return "FAIL", total, floors
    if total < fcfg["pass_total"]:
        worst = max(_LOSS_REASON, key=lambda k: w[k] * (1.0 - scores[k]))
        return "FAIL", total, [_LOSS_REASON[worst]]
    return "PASS", total, []
