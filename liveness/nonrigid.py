"""Non-rigid residual after 2D similarity alignment (PROMPT.md 5.2). Weak auxiliary score, never a veto.

Portable core: numpy + stdlib only.
"""
from __future__ import annotations

import numpy as np

from liveness.frame import palm_width_px

_MAX_ITER = 100
_TOL = 1e-10


def _align(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Similarity-align centred shapes x (N,P,2) onto centred y (P,2): s * x @ R, R a proper rotation."""
    u, sv, vt = np.linalg.svd(np.einsum("npi,pj->nij", x, y))
    d = np.sign(np.linalg.det(u @ vt))
    d[d == 0] = 1.0
    u[..., :, 1] *= d[:, None]                 # flip the weakest axis instead of reflecting
    rot = u @ vt
    s = (sv[:, 0] + d * sv[:, 1]) / np.maximum(np.einsum("npi,npi->n", x, x), 1e-300)
    return s[:, None, None] * (x @ rot)


def _gpa(x: np.ndarray) -> np.ndarray:
    """Generalised Procrustes on centred shapes; aligned to the converged mean at the median frame size (px)."""
    size = np.sqrt(np.einsum("npi,npi->n", x, x))
    mean = x[0] / size[0]
    for _ in range(_MAX_ITER):
        new = _align(x, mean).mean(axis=0)
        new /= np.linalg.norm(new)
        done = np.sum((new - mean) ** 2) < _TOL
        mean = new
        if done:
            break
    return _align(x, mean * np.median(size))  # median, like the palm width: one outlier frame cannot rescale r


def evaluate(lm_px_hold_ok: np.ndarray, cfg: dict) -> tuple[float | None, list[str], dict]:
    """(score, [], debug) from HOLD-window quality-passed pixel landmarks (N, 21, 3); x, y only.

    r = sqrt(mean_i Var_t(aligned P_i)) / median pixel palm width, Var_t of a 2D point = trace of its
    temporal covariance. score = sigmoid((r - c) / s). NaN rows are dropped; < cfg min_frames usable
    frames or a degenerate palm width -> score None. min_frames is only a statistical floor: the API has
    no timestamps, so the caller must gate on the HOLD time span (r is biased low on short spans).
    Static noise floor: r = 1.345 * sigma / pw * sqrt((N-1)/N) (2P-4 dof per frame, P = 21).
    """
    lm = np.asarray(lm_px_hold_ok, np.float64)
    if lm.ndim != 3 or lm.shape[0] == 0:
        lm = np.zeros((0, 21, 3))
    lm = lm[np.isfinite(lm[..., :2]).all(axis=(1, 2))]
    debug = {"r": None, "n_frames": int(len(lm))}
    p = cfg["nonrigid"]
    if len(lm) < p["min_frames"]:
        return None, [], debug
    pw = float(np.median([palm_width_px(f) for f in lm]))
    xy = lm[..., :2] - lm[..., :2].mean(axis=1, keepdims=True)
    if not np.isfinite(pw) or pw <= 0 or np.any(np.einsum("npi,npi->n", xy, xy) <= 0):
        return None, [], debug
    aligned = _gpa(xy)
    r = float(np.sqrt(aligned.var(axis=0).sum(axis=-1).mean()) / pw)
    score = float(0.5 * (1.0 + np.tanh((r - p["c"]) / p["s"] / 2.0)))  # sigmoid without exp overflow
    debug["r"] = r
    return score, [], debug
