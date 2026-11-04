"""Screen / re-capture texture score from palm ROI blocks (PROMPT.md 5.4).

A screen or print seen by the camera adds a narrow-band periodic pattern (pixel grid, moire);
skin does not. Per block we measure how peaked the mid/high-frequency power spectrum is.
Known limit: score is a sigmoid linear in M, so strong straight edges (finger gaps, a palm silhouette
well inside the block) reach grating-level M and score ~0 whatever c is; a log-M sigmoid would be a spec change.
"""
from __future__ import annotations

import numpy as np


def block_peak_ratio(block: np.ndarray, band) -> float:
    """m = max / mean of |FFT|^2 inside the radial band (cycles/px); NaN for a zero-power block."""
    x = np.asarray(block, dtype=np.float64)
    x = x - x.mean()
    h, w = x.shape
    p = np.abs(np.fft.fft2(x * np.outer(np.hanning(h), np.hanning(w)))) ** 2
    r = np.hypot(np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :])
    a = p[(r >= band[0]) & (r <= band[1])]
    mean = a.mean() if a.size else 0.0
    return float(a.max() / mean) if mean > 0 else float("nan")


def evaluate(roi_ok: np.ndarray, cfg: dict) -> tuple[float | None, list[str], dict]:
    """(score, reasons, debug) over quality-gated ROI blocks (M, H, W) uint8; score None = INSUFFICIENT."""
    c = cfg["texture"]
    ms = [block_peak_ratio(b, c["band"]) for b in np.asarray(roi_ok)]
    good = [m for m in ms if np.isfinite(m)]   # constant blocks carry no texture evidence
    debug = {"M": None, "n_roi": len(good), "m": [m if np.isfinite(m) else None for m in ms]}
    if len(good) < c["min_blocks"]:
        return None, ["LOW_QUALITY"], debug
    big_m = float(np.median(good))
    debug["M"] = big_m
    # 1 - sigmoid(z) = (1 - tanh(z/2)) / 2, overflow-free for any z
    score = float(0.5 * (1.0 - np.tanh((big_m - c["c"]) / c["s"] / 2.0)))
    return score, [], debug
