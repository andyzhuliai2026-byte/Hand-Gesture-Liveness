"""liveness/planarity.py (PROMPT.md 5.3): synthetic pinhole sequences from the 3D gesture templates."""
import json

import numpy as np
import pytest

from gesture_templates import template
from liveness import planarity
from liveness.frame import GESTURES, palm_width_px

F, DIST, CX, CY = 900.0, 0.40, 640.0, 360.0
CFG = {"planarity": {"min_pair_dt_s": 0.3, "min_normal_deg": 12.0, "max_pairs": 10, "min_pairs": 3,
                     "tau_hard": 0.02, "c": 0.04, "s": 0.01, "ransac_thresh": 0.02,
                     "ransac_iters": 200, "ransac_seed": 0}}
# OPEN_PALM is nearly flat (fingers flex 3-5 deg): synthetic R ~ tau_hard, so the strict test is expected to
# fail for it; PROMPT 5.3 M4 rule (drop from the challenge set or per-gesture threshold) handles it.
STRICT = tuple(pytest.param(g, marks=pytest.mark.xfail(strict=True, reason="near-planar; PROMPT 5.3 M4 rule"))
               if g == "OPEN_PALM" else g for g in GESTURES)


def rot(axis, deg):
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    a = np.radians(deg)
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def flatten(pts):
    out = pts.copy()
    out[:, 2] = out[0, 2]  # palm plane (template palm lies at constant z)
    return out


def project(cam):
    px = np.empty_like(cam)
    px[:, 0] = F * cam[:, 0] / cam[:, 2] + CX
    px[:, 1] = F * cam[:, 1] / cam[:, 2] + CY
    px[:, 2] = F * (cam[:, 2] - DIST) / DIST
    return px


def make_seq(pts, seed, theta=None, fps=30.0, dur=1.5, noise=1.0, world_noise=0.0):
    """Hand turns 0 -> theta (15-25 deg) about a random in-palm axis over the HOLD window."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(15, 25) if theta is None else theta
    phi = rng.uniform(0, np.pi)
    axis = (np.cos(phi), np.sin(phi), 0.0)
    base = rot(rng.normal(size=3), rng.uniform(0, 10))
    offset = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.03, 0.03), DIST])
    t = np.arange(int(round(dur * fps)) + 1) / fps
    world = np.stack([pts @ (base @ rot(axis, theta * tk / dur)).T for tk in t])
    px = np.stack([project(w + offset) for w in world])
    px[..., :2] += rng.normal(0, noise, px[..., :2].shape)
    return t, px, world + rng.normal(0, world_noise, world.shape)


def run(pts, seed, label="FIST", **kw):
    t, px, world = make_seq(pts, seed, **kw)
    return planarity.evaluate(t, px, world, [label] * len(t), np.ones(len(t), bool), label, CFG)


def test_homography_dlt_exact():
    H = np.array([[1.1, 0.05, 30.0], [-0.08, 0.95, -12.0], [2e-4, -1e-4, 1.0]])
    src = np.random.default_rng(0).uniform(0, 600, (21, 2))
    h = np.c_[src, np.ones(21)] @ H.T
    dst = h[:, :2] / h[:, 2:]
    est = planarity.homography_dlt(src, dst)
    np.testing.assert_allclose(est / est[2, 2], H, rtol=1e-8, atol=1e-10)


def test_palm_normal_unit_and_nan():
    n = planarity.palm_normal(template("FIST"))
    assert np.isclose(np.linalg.norm(n), 1.0)
    assert np.all(np.isnan(planarity.palm_normal(np.full((21, 3), np.nan))))


@pytest.mark.parametrize("gesture", STRICT)
def test_real_hand_vs_flat_photo(gesture):
    """Directional: 20 trials each; median positive R > neg 95th pct; FAR = FRR = 0 at tau_hard."""
    pts = template(gesture)
    pos = [run(pts, s, gesture) for s in range(20)]
    neg = [run(flatten(pts), s, gesture) for s in range(20)]
    r_pos = np.array([d["R"] for _, _, d in pos])
    r_neg = np.array([d["R"] for _, _, d in neg])
    assert np.median(r_pos) > np.percentile(r_neg, 95)
    assert all("PLANAR" not in r for _, r, _ in pos)          # FRR = 0
    assert all(r == ["PLANAR"] for _, r, _ in neg)            # FAR = 0
    assert np.median([s for s, _, _ in pos]) > np.median([s for s, _, _ in neg])
    for _, _, d in pos + neg:
        json.dumps(d, allow_nan=False)
        assert CFG["planarity"]["min_pairs"] <= d["n_pairs"] <= CFG["planarity"]["max_pairs"]
        assert len({f for pr in d["pairs"] for f in pr}) == 2 * d["n_pairs"]  # frame-disjoint


def test_open_palm_reported():
    """OPEN_PALM is the near-planar gesture the PROMPT warns about: classes separate, but below tau_hard
    (strict test xfails). Synthetic: live R min 0.012 / median 0.016, photo max 0.009."""
    pts = template("OPEN_PALM")
    r_pos = [run(pts, s, "OPEN_PALM")[2]["R"] for s in range(20)]
    r_neg = [run(flatten(pts), s, "OPEN_PALM")[2]["R"] for s in range(20)]
    assert min(r_pos) > max(r_neg) and np.median(r_pos) > 1.5 * np.median(r_neg)


def test_fps_independent():
    pts = template("FIST")
    r30 = run(pts, 3, noise=0.0, fps=30.0)[2]["R"]
    r15 = run(pts, 3, noise=0.0, fps=15.0)[2]["R"]
    assert abs(r30 - r15) < 0.15 * r30


def test_no_motion_is_insufficient():
    score, reasons, d = run(template("FIST"), 0, theta=0.0)
    assert score is None and reasons == ["NEED_ROTATION"] and d["R"] is None
    json.dumps(d, allow_nan=False)


def _rolled(pts, base, axis, deg, seed, label):
    """Hand at orientation `base`, turning `deg` about camera `axis` over 1.5 s."""
    rng = np.random.default_rng(seed)
    t = np.arange(46) / 30.0
    world = np.stack([pts @ (rot(axis, deg * tk / 1.5) @ base).T for tk in t])
    px = np.stack([project(w + [0.02, -0.01, DIST]) for w in world])
    px[..., :2] += rng.normal(0, 1.0, px[..., :2].shape)
    return planarity.evaluate(t, px, world, [label] * len(t), np.ones(len(t), bool), label, CFG)


def test_image_plane_roll_is_insufficient():
    """PL-2: palm sideways, rolled 25 deg about the optical axis = 2D similarity, no parallax -> NEED_ROTATION
    (the palm normal turns 25 deg, so a normal-angle selector would issue a hard PLANAR veto on a live hand)."""
    for s in range(5):
        assert _rolled(template("THUMBS_UP"), rot((0, 1, 0), 90), (0, 0, 1), 25, s, "THUMBS_UP")[:2] \
            == (None, ["NEED_ROTATION"])
    # converse: sideways palm turned about the image-horizontal axis has parallax although its normal stays put
    for s in range(5):
        score, reasons, _ = _rolled(template("THUMBS_UP"), rot((0, 1, 0), 90), (1, 0, 0), 20, s, "THUMBS_UP")
        assert score is not None and reasons == []


def test_still_hand_with_world_jitter_is_insufficient():
    """PL-1/PL-2: held still, 3 mm world-landmark jitter (smoke-like normal noise) -> NEED_ROTATION, never PLANAR."""
    for g in ("FIST", "THUMBS_UP", "OPEN_PALM"):
        for s in range(10):
            assert run(template(g), s, g, theta=0.0, world_noise=0.003)[:2] == (None, ["NEED_ROTATION"]), (g, s)


def test_one_glitched_frame_does_not_decide():
    """PL-1: tilting flat photo with one bad frame (world +35 deg, 2D sigma 4% palm width) stays PLANAR;
    a still live hand with one 15-deg world glitch gives NEED_ROTATION, not PLANAR."""
    for s in range(10):
        t, px, w = make_seq(flatten(template("FIST")), s)
        w[0] = w[0] @ rot((0, 1, 0), 35).T
        px[0, :, :2] += np.random.default_rng(100 + s).normal(0, 0.04 * palm_width_px(px[0]), (21, 2))
        assert planarity.evaluate(t, px, w, ["FIST"] * len(t), np.ones(len(t), bool), "FIST", CFG)[1] == ["PLANAR"]
        t, px, w = make_seq(template("FIST"), s, theta=0.0)
        w[20] = w[20] @ rot((1, 0, 0), 15).T
        assert planarity.evaluate(t, px, w, ["FIST"] * len(t), np.ones(len(t), bool), "FIST", CFG)[:2] \
            == (None, ["NEED_ROTATION"])


def test_insufficient_keeps_pair_debug():
    """PL-5: pairs that qualified are reported even when fewer than min_pairs."""
    cfg = {"planarity": dict(CFG["planarity"], max_pairs=2)}
    t, px, w = make_seq(template("FIST"), 1)
    score, reasons, d = planarity.evaluate(t, px, w, ["FIST"] * len(t), np.ones(len(t), bool), "FIST", cfg)
    assert (score, reasons, d["R"], d["rho"]) == (None, ["NEED_ROTATION"], None, None)
    assert d["n_pairs"] == 2 and len(d["pairs"]) == len(d["pair_e"]) == len(d["normal_angles"]) == 2
    json.dumps(d, allow_nan=False)


def test_pairs_filter_label_ok_dt():
    t, px, world = make_seq(template("FIST"), 1)
    labels = ["FIST"] * len(t)
    s0, _, _ = planarity.evaluate(t, px, world, labels, np.ones(len(t), bool), "FIST", CFG)
    assert s0 is not None
    for lab, ok in ((["V_SIGN"] * len(t), np.ones(len(t), bool)), (labels, np.zeros(len(t), bool))):
        assert planarity.evaluate(t, px, world, lab, ok, "FIST", CFG)[:2] == (None, ["NEED_ROTATION"])
    # rotation fully inside 0.25 s -> no pair satisfies dt >= 0.3 s with enough angle
    t2, px2, w2 = make_seq(template("FIST"), 1, dur=0.25)
    assert planarity.evaluate(t2, px2, w2, ["FIST"] * len(t2), np.ones(len(t2), bool), "FIST", CFG)[1] \
        == ["NEED_ROTATION"]


@pytest.mark.parametrize("n", [0, 1, 5])
def test_degenerate_inputs(n):
    t = np.arange(n) / 30.0
    nan = np.full((n, 21, 3), np.nan)
    score, reasons, d = planarity.evaluate(t, nan, nan, ["FIST"] * n, np.ones(n, bool), "FIST", CFG)
    assert (score, reasons) == (None, ["NEED_ROTATION"])
    json.dumps(d, allow_nan=False)


def test_nan_rows_skipped_and_inputs_untouched():
    t, px, world = make_seq(template("FIST"), 2)
    px[::4] = np.nan
    world[1::4] = np.nan
    px0, w0 = px.copy(), world.copy()
    score, reasons, d = planarity.evaluate(t, px, world, ["FIST"] * len(t), np.ones(len(t), bool), "FIST", CFG)
    assert score is not None and reasons == []
    np.testing.assert_array_equal(px, px0)
    np.testing.assert_array_equal(world, w0)
    json.dumps(d, allow_nan=False)


def test_bent_photo_cylinder_known_limitation():
    """Photo wrapped on a r = 5 cm cylinder (arc ~1.9 cm): 5.3 is expected to miss it (PROMPT threat model)."""
    flat, r = flatten(template("FIST")), 0.05
    bent = flat.copy()
    x = flat[:, 0] - flat[:, 0].mean()
    bent[:, 0] = flat[:, 0].mean() + r * np.sin(x / r)
    bent[:, 2] = flat[:, 2] + r * (1 - np.cos(x / r))
    caught = sum("PLANAR" in run(bent, s)[1] for s in range(20))
    if caught < 20:
        pytest.xfail(f"known limitation: cylinder-bent photo r=5cm caught {caught}/20 (DESIGN.md blind spot)")


def test_gross_outlier_raises_e_and_rho_flags_it():
    """(c) flat sample + one 20%-palm-width outlier: full-point e rises >= 3x, rho = 20/21 (+-1/21)."""
    t, px, _ = make_seq(flatten(template("FIST")), 4)
    a, b = px[0, :, :2], px[-1, :, :2]
    pw = palm_width_px(px[0])
    p = CFG["planarity"]
    e0, rho0 = planarity.pair_residual(a, b, pw, p["ransac_thresh"], p["ransac_iters"], p["ransac_seed"])
    b2 = b.copy()
    b2[8] += 0.2 * pw * np.array([0.6, 0.8])
    e1, rho1 = planarity.pair_residual(a, b2, pw, p["ransac_thresh"], p["ransac_iters"], p["ransac_seed"])
    assert e1 >= 3 * e0
    assert abs(rho1 - 20 / 21) <= 1 / 21 + 1e-9
    assert rho0 >= 19 / 21  # planar ~ 1.0; 1 px noise vs a 3.5 px threshold can drop one point
