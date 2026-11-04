"""PROMPT 5.2 non-rigid residual: synthetic in-plane motion vs per-finger wiggle."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gesture_templates import FINGERS, template  # noqa: E402
from liveness.frame import GESTURES, palm_width_px  # noqa: E402
from liveness.nonrigid import evaluate  # noqa: E402

# synth placeholders: c = mid(neg p95 0.0069, pos p50 0.0135), s = (c - neg p95) / 2. c is ~1.5x the synthetic
# noise floor and says nothing about real data (jitter 2-5 px, HOLD tilt); M4 recalibrates from attack/real hands.
CFG = {"nonrigid": {"c": 0.010, "s": 0.0015, "min_frames": 5}}
MOTIONS = ("static", "translate", "rotate", "scale")
HOLD_S = 1.5


def base_px(gesture="OPEN_PALM"):
    """Pinhole projection f = 900 px, 40 cm away, centred in a 1280x720 frame; (21, 3)."""
    w = template(gesture)
    zc = 0.40 + w[:, 2]
    return np.c_[900 * w[:, 0] / zc + 640, 900 * w[:, 1] / zc + 360, 900 * w[:, 2]]


def sequence(rng, motion, wiggle, fps=30.0, gesture="OPEN_PALM", amp=None):
    """(N, 21, 3) px landmarks over HOLD_S: in-plane motion + noise 0.5% pw (+ per-finger wiggle 1-3% pw)."""
    b = base_px(gesture)
    pw = palm_width_px(b)
    t = np.arange(0.0, HOLD_S, 1.0 / fps)
    u = t / HOLD_S
    ang = {"rotate": np.deg2rad(20) * np.sin(2 * np.pi * u)}.get(motion, 0 * u)
    sc = {"scale": 1 + 0.15 * np.sin(2 * np.pi * u)}.get(motion, 1 + 0 * u)
    sh = {"translate": np.c_[60 * u, -40 * u]}.get(motion, np.zeros((len(t), 2)))
    c = b[:, :2].mean(0)
    xy0 = np.broadcast_to(b[:, :2] - c, (len(t), 21, 2)).copy()
    if wiggle:
        for f in FINGERS:
            a = (amp if amp is not None else rng.uniform(0.01, 0.03)) * pw
            freq, ph = rng.uniform(0.5, 2.0), rng.uniform(0, 2 * np.pi)
            d = rng.normal(size=2)
            xy0[:, list(f)] += (a * np.sin(2 * np.pi * freq * t + ph))[:, None, None] * (d / np.linalg.norm(d))
    cs, sn = np.cos(ang), np.sin(ang)
    rot = np.stack([np.stack([cs, -sn], -1), np.stack([sn, cs], -1)], -2)  # (N, 2, 2)
    xy = np.einsum("nij,npj->npi", rot, xy0) * sc[:, None, None] + c + sh[:, None, :]
    xy += rng.normal(0, 0.005 * pw, xy.shape)
    return np.concatenate([xy, np.broadcast_to(b[:, 2:], (len(t), 21, 1))], -1)


def r_of(lm):
    s, reasons, dbg = evaluate(lm, CFG)
    assert reasons == []
    return dbg["r"]


def test_positive_median_above_negative_p95():
    rng = np.random.default_rng(0)
    neg = [r_of(sequence(rng, m, False)) for m in MOTIONS for _ in range(25)]
    pos = [r_of(sequence(rng, m, True)) for m in MOTIONS for _ in range(25)]
    assert np.median(pos) > np.percentile(neg, 95)
    assert np.median(pos) >= 1.5 * np.percentile(neg, 95)  # fixed margin (PROMPT 7.1)
    # synth defaults: c sits between negative p95 and positive median
    assert np.percentile(neg, 95) < CFG["nonrigid"]["c"] < np.median(pos)


def test_scores_are_directional_and_bounded():
    rng = np.random.default_rng(0)
    s_neg, _, _ = evaluate(sequence(rng, "rotate", False), CFG)
    s_pos, _, _ = evaluate(sequence(rng, "rotate", True), CFG)
    assert 0.0 <= s_neg < 0.5 < s_pos <= 1.0
    assert s_pos - s_neg >= 0.5  # fixed margin (PROMPT 7.1)
    # r grows with wiggle amplitude, same noise/phase stream for every amplitude
    seed = rng.spawn(1)[0]
    rs = [r_of(sequence(np.random.default_rng(seed.bit_generator.seed_seq), "translate", True, amp=a))
          for a in (0.0, 0.01, 0.02, 0.04)]
    assert all(b >= 1.05 * a for a, b in zip(rs, rs[1:])), rs


def test_similarity_motion_is_absorbed_without_noise():
    b = base_px()
    ang = np.deg2rad(np.linspace(-25, 25, 30))
    R = np.stack([np.stack([np.cos(ang), -np.sin(ang)], -1), np.stack([np.sin(ang), np.cos(ang)], -1)], -2)
    xy = np.einsum("nij,pj->npi", R, b[:, :2] - 640) * np.linspace(0.8, 1.2, 30)[:, None, None] + 300
    lm = np.concatenate([xy, np.zeros((30, 21, 1))], -1)
    assert r_of(lm) < 1e-6


def test_reflection_is_not_used_for_alignment():
    for g in GESTURES:
        b = base_px(g)
        mirrored = b.copy()
        mirrored[:, 0] = 1280 - mirrored[:, 0]
        lm = np.stack([b, mirrored] * 5)
        # a reflection would align the mirror exactly (r = 0); proper rotations cannot
        assert r_of(lm) > 0.05, g
        assert r_of(np.stack([b] * 10)) < 1e-9


def test_frame_rate_invariance():
    rs = {}
    for fps in (30.0, 15.0):
        rs[fps] = np.median([r_of(sequence(g, "rotate", True, fps=fps)) for g in np.random.default_rng(0).spawn(20)])
    assert abs(rs[30.0] - rs[15.0]) / rs[30.0] < 0.15


def test_nan_rows_dropped_and_input_not_mutated():
    lm = sequence(np.random.default_rng(0), "translate", True)
    ref = r_of(lm)
    lm2 = lm.copy()
    lm2[::4] = np.nan
    before = lm2.copy()
    s, reasons, dbg = evaluate(lm2, CFG)
    np.testing.assert_array_equal(np.isnan(lm2), np.isnan(before))
    np.testing.assert_array_equal(lm2[~np.isnan(lm2)], before[~np.isnan(before)])
    assert dbg["n_frames"] == len(lm) - len(lm[::4])
    assert abs(dbg["r"] - ref) / ref < 0.25
    json.dumps(dbg)


def test_degenerate_inputs():
    b = base_px()
    cases = [
        np.zeros((0, 21, 3)),
        np.zeros((0,)),
        np.full((20, 21, 3), np.nan),
        np.stack([b] * 4),                        # too few frames
        np.ones((10, 21, 3)),                     # zero palm width
        np.concatenate([np.stack([b] * 3), np.full((10, 21, 3), np.nan)]),
    ]
    for lm in cases:
        s, reasons, dbg = evaluate(lm, CFG)
        assert s is None and reasons == [], lm.shape
        json.dumps(dbg, allow_nan=False)
        assert dbg["r"] is None


def test_static_noise_floor_known_answer():
    """Generator-free: static shape + iid noise -> r = sqrt((2P-4)/P) * sigma/pw (4 similarity dof per frame).

    Affine alignment would give 1.309, translation-only 1.380 instead of 1.345."""
    rng = np.random.default_rng(0)
    b, n, sigma = base_px(), 2000, 2.0
    lm = np.broadcast_to(b, (n, 21, 3)).copy()
    lm[..., :2] += rng.normal(0, sigma, (n, 21, 2))
    pw = np.median([palm_width_px(f) for f in lm])
    k = r_of(lm) * pw / sigma / np.sqrt((n - 1) / n)
    assert abs(k - np.sqrt(38 / 21)) < 0.01, k


def test_z_is_ignored():
    rng = np.random.default_rng(0)
    lm = sequence(rng, "rotate", True)
    noisy = lm.copy()
    noisy[..., 2] = rng.normal(0, 1e3, noisy.shape[:2])
    assert r_of(noisy) == r_of(lm)


def test_min_frames_boundary():
    lm = sequence(np.random.default_rng(0), "rotate", True)
    n = CFG["nonrigid"]["min_frames"]
    assert evaluate(lm[:n], CFG)[0] is not None
    assert evaluate(lm[:n - 1], CFG)[0] is None


def test_median_palm_width_and_size_resist_outlier_frames():
    lm = sequence(np.random.default_rng(0), "translate", True)
    big = lm.copy()
    c = big[::9, :, :2].mean(axis=1, keepdims=True)
    big[::9, :, :2] = c + 3.0 * (big[::9, :, :2] - c)  # 5 of 45 frames 3x larger: a mean would move by ~22%
    assert abs(r_of(big) - r_of(lm)) / r_of(lm) < 0.01  # median rank shifts slightly; a mean moves ~20%
