"""Planarity check (PROMPT.md 5.3): a flat photo maps between frames by one homography, a 3D hand
does not once it turns out of plane. Core item, weight 0.65, hard veto PLANAR.

Portable core: numpy + stdlib + liveness.frame only.
"""
from __future__ import annotations

import numpy as np

from liveness.frame import palm_width_px

RIGID = [0, 1, 5, 9, 13, 17]  # wrist + finger bases: the near-rigid part of the hand


def palm_normal(lm_world: np.ndarray) -> np.ndarray:
    """(3,) unit normal of the P0, P5, P17 plane (also batched (..., 21, 3)); NaN if undefined."""
    w = np.asarray(lm_world, np.float64)
    n = np.cross(w[..., 5, :] - w[..., 0, :], w[..., 17, :] - w[..., 0, :])
    with np.errstate(all="ignore"):
        n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    return np.where(np.isfinite(n).all(-1, keepdims=True), n, np.nan)


def swing_deg(lm_world: np.ndarray) -> np.ndarray:
    """(N, N) out-of-image-plane angle of the relative rotation a -> b (Kabsch on RIGID world points).

    swing = arccos(R_rel[2, 2]): the change of viewing direction. Roll about the optical axis is a 2D
    similarity (no parallax, PROMPT 5.3) and gives 0; turning about any image-plane axis gives the full angle.
    lm_world axes follow the camera (smoke: in-plane orientation tracks lm_px roll, r = 0.98).
    """
    A = np.asarray(lm_world, np.float64)[:, RIGID, :]
    A = A - A.mean(1, keepdims=True)
    U, _, Vt = np.linalg.svd(np.einsum("aki,bkj->abij", A, A))
    V, Ut = np.swapaxes(Vt, -1, -2), np.swapaxes(U, -1, -2)
    D = np.ones(U.shape[:-1])
    D[..., 2] = np.sign(np.linalg.det(V @ Ut))
    R = (V * D[..., None, :]) @ Ut
    return np.degrees(np.arccos(np.clip(R[..., 2, 2], -1.0, 1.0)))


def _hartley(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shift to the centroid, scale to mean distance sqrt(2). -> (normalised points, T)."""
    c = p.mean(-2, keepdims=True)
    s = np.sqrt(2) / np.linalg.norm(p - c, axis=-1).mean(-1)
    T = np.zeros(p.shape[:-2] + (3, 3))
    T[..., 0, 0] = T[..., 1, 1] = s
    T[..., :2, 2] = -s[..., None] * c[..., 0, :]
    T[..., 2, 2] = 1.0
    return (p - c) * s[..., None, None], T


def homography_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """(3,3) H with dst ~ H src: Hartley normalisation + DLT least squares (SVD). Batched over leading dims."""
    (s, Ts), (d, Td) = _hartley(np.asarray(src, np.float64)), _hartley(np.asarray(dst, np.float64))
    x, y, u, v = s[..., 0], s[..., 1], d[..., 0], d[..., 1]
    z, o = np.zeros_like(x), np.ones_like(x)
    A = np.concatenate([np.stack([-x, -y, -o, z, z, z, u * x, u * y, u], -1),
                        np.stack([z, z, z, -x, -y, -o, v * x, v * y, v], -1)], -2)
    Hn = np.linalg.svd(A)[2][..., -1, :].reshape(A.shape[:-2] + (3, 3))
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / np.linalg.norm(H, axis=(-2, -1), keepdims=True)


def _apply(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    h = np.concatenate([xy, np.ones(xy.shape[:-1] + (1,))], -1) @ np.swapaxes(H, -1, -2)
    return h[..., :2] / h[..., 2:]


def pair_residual(xy_a: np.ndarray, xy_b: np.ndarray, pw: float,
                  thresh: float, iters: int, seed: int) -> tuple[float, float]:
    """-> (e, rho). e = RMS reprojection error of ALL points / pw (the score feature, never on inliers);
    rho = best RANSAC inlier ratio over `iters` 4-point sets, threshold thresh * pw (recorded only)."""
    n = len(xy_a)
    with np.errstate(all="ignore"):
        r = _apply(homography_dlt(xy_a, xy_b), xy_a) - xy_b
        e = float(np.sqrt(np.mean(np.sum(r * r, -1))) / pw)
        idx = np.random.default_rng(seed).permuted(np.tile(np.arange(n), (iters, 1)), axis=1)[:, :4]
        err = np.linalg.norm(_apply(homography_dlt(xy_a[idx], xy_b[idx]), xy_a) - xy_b, axis=-1)
        best = err[np.argmax((err < thresh * pw).sum(1))] < thresh * pw
        if best.sum() >= 4:  # one least-squares refit on the best consensus set, then recount
            best = np.linalg.norm(_apply(homography_dlt(xy_a[best], xy_b[best]), xy_a) - xy_b, axis=-1) < thresh * pw
        rho = float(best.sum() / n)
    return e, rho


def _sigmoid(x: float) -> float:
    return float(0.5 * (1.0 + np.tanh(0.5 * x)))


def _f(x) -> float | None:
    return float(x) if x is not None and np.isfinite(x) else None


def evaluate(t: np.ndarray, lm_px: np.ndarray, lm_world: np.ndarray, labels: list[str],
             ok: np.ndarray, target: str, cfg: dict) -> tuple[float | None, list[str], dict]:
    """Planarity over the HOLD window. ok = frames in HOLD that passed quality gating (caller).

    Pair angle = swing_deg (out-of-image-plane rotation), compared with min_normal_deg (dev entry: PROMPT
    says palm-normal angle, which also admits parallax-free image-plane roll). Pairs are taken greedily by
    angle, descending, each frame at most once, up to max_pairs (one deviant frame cannot decide R).
    -> (score, reasons, debug). (None, ["NEED_ROTATION"], debug) when < min_pairs qualified pairs;
    reasons ["PLANAR"] when R < tau_hard (score still reported).
    """
    p = cfg["planarity"]
    t = np.asarray(t, np.float64).reshape(-1)
    px = np.asarray(lm_px, np.float64).reshape(len(t), 21, 3)
    world = np.asarray(lm_world, np.float64).reshape(len(t), 21, 3)
    debug = {"R": None, "rho": None, "n_pairs": 0, "normal_angles": [], "palm_normal_deg": [], "pair_e": [],
             "pair_rho": [], "pairs": [], "max_normal_deg": None, "z_spread": None}

    normals = palm_normal(world)  # debug / z report only
    with np.errstate(invalid="ignore"):  # inf landmarks; those frames are dropped by cand
        pw = np.array([palm_width_px(f) for f in px])
    cand = (np.asarray(ok, bool).reshape(-1) & (np.asarray(labels, dtype=object).reshape(-1) == target)
            & np.isfinite(px[..., :2]).all((1, 2)) & np.isfinite(world[:, RIGID]).all((1, 2)) & (pw > 0))
    c = np.flatnonzero(cand)
    ang = swing_deg(world[c])
    dt_ok = (t[c][None, :] - t[c][:, None]) >= p["min_pair_dt_s"]
    if dt_ok.any():
        debug["max_normal_deg"] = float(ang[dt_ok].max())
    ia, ib = np.nonzero(dt_ok & (ang >= p["min_normal_deg"]))
    pairs, used = [], set()
    for k in np.argsort(-ang[ia, ib], kind="stable"):
        a, b = int(c[ia[k]]), int(c[ib[k]])
        if a in used or b in used:
            continue
        used |= {a, b}
        pairs.append((a, b, float(ang[ia[k], ib[k]])))
        if len(pairs) == p["max_pairs"]:
            break

    stats = [pair_residual(px[a, :, :2], px[b, :, :2], pw[a], p["ransac_thresh"], p["ransac_iters"],
                           p["ransac_seed"]) for a, b, _ in pairs]
    good = [(pr, s) for pr, s in zip(pairs, stats) if np.isfinite(s[0])]
    e = np.array([s[0] for _, s in good])
    rho = np.array([s[1] for _, s in good])
    with np.errstate(invalid="ignore"):
        nd = [np.degrees(np.arccos(np.clip(normals[a] @ normals[b], -1.0, 1.0))) for (a, b, _), _ in good]
    debug.update(n_pairs=len(good), normal_angles=[ang_ for (_, _, ang_), _ in good],
                 palm_normal_deg=[_f(x) for x in nd], pair_e=e.tolist(), pair_rho=rho.tolist(),
                 pairs=[[a, b] for (a, b, _), _ in good])
    if len(good) < p["min_pairs"]:
        return None, ["NEED_ROTATION"], debug

    R = float(np.median(e))
    frames = sorted({f for (a, b, _), _ in good for f in (a, b)})
    zs = [np.std((world[f] - world[f, 0]) @ normals[f]) for f in frames]  # MediaPipe z, report only
    debug.update(R=R, rho=float(np.median(rho)), z_spread=_f(np.median(zs)))
    score = _sigmoid((R - p["c"]) / p["s"])
    return score, (["PLANAR"] if R < p["tau_hard"] else []), debug
