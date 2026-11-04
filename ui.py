"""On-screen overlay (PROMPT.md 3.1, 3.2): the designed HUD (hud.py) keyed on SessionOutput. ASCII text only."""
from __future__ import annotations

import numpy as np

import hud
from liveness.frame import Frame

WIN = "HandLiveness"
_ST = hud.HudState()  # ponytail: one window per process, so one display state; HudState per window if that changes


def draw(img_bgr: np.ndarray, fr: Frame, pose, out, stats: dict, debug: bool, cfg: dict) -> np.ndarray:
    return hud.draw(img_bgr, fr, pose, out, stats, debug, cfg, _ST)
