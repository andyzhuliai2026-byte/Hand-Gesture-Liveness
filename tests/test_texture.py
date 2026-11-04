"""Synthetic tests for liveness/texture.py (PROMPT.md 5.4).

c / s are synth placeholders pending M4 calibration, derived from the structured-skin set below:
white skin M ~8-9, 1/f^2.5 texture + 0.7 px blur M ~29-31, 2 creases M ~44-68, 12 px palm edge M ~57-66
(all score >= 0.78); 4 px grating amp 0.5 / 1 / 10 -> M ~84-94 / ~290-312 / ~1560-1710 (score ~0.6 / 0 / 0).
Known limit (sigmoid linear in M): straight strong edges are indistinguishable from gratings - finger gaps
reach M 740-1270 and a palm edge 20 px inside the block M ~240, both score ~0 whatever c is.
"""
import json

import numpy as np

from liveness.texture import block_peak_ratio, evaluate

N = 128
# PLAN.md section 2 M3 defaults; c / s are synth placeholders pending calibration.
CFG = {"texture": {"band": [0.15, 0.45], "c": 100.0, "s": 25.0, "min_blocks": 5}}


def _skin(rng: np.random.Generator) -> np.ndarray:
    """Skin-like float ROI: smooth low-pass field + illumination gradient + mild sensor noise."""
    f = np.fft.fftfreq(N)
    r = np.hypot(f[:, None], f[None, :])
    lp = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(N, N))) * np.exp(-(r / 0.04) ** 2)))
    yy, xx = np.mgrid[0:N, 0:N]
    return 118 + 15 * lp / lp.std() + 0.15 * xx + 0.05 * yy + rng.normal(0, 2.0, (N, N))


def _blur(x: np.ndarray, sig: float) -> np.ndarray:
    f = np.fft.fftfreq(N)
    r = np.hypot(f[:, None], f[None, :])
    return np.real(np.fft.ifft2(np.fft.fft2(x) * np.exp(-2 * (np.pi * sig * r) ** 2)))


def _fractal_skin(rng):
    """118 + isotropic 1/f^2.5 texture (std 10), 0.7 px optics blur, noise 1.5."""
    f = np.fft.fftfreq(N)
    r = np.hypot(f[:, None], f[None, :])
    g = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(N, N))) * np.where(r > 0, r, np.inf) ** -1.25))
    return 118 + _blur(10 * g / g.std(), 0.7) + rng.normal(0, 1.5, (N, N))


def _crease_skin(rng, k=2, contrast=25.0, width=2.5):
    """Skin with k curved dark palm creases."""
    x = _skin(rng)
    yy, xx = np.mgrid[0:N, 0:N]
    for _ in range(k):
        th, c0, curv = rng.uniform(0, np.pi), rng.uniform(30, 98, 2), rng.uniform(-0.01, 0.01)
        u = (xx - c0[0]) * np.cos(th) + (yy - c0[1]) * np.sin(th)
        v = -(xx - c0[0]) * np.sin(th) + (yy - c0[1]) * np.cos(th)
        x = x - contrast * np.exp(-0.5 * ((v - curv * u ** 2) / width) ** 2) * (np.abs(u) < 50)
    return _blur(x, 0.8)


def _edge_skin(rng, d=12, contrast=80.0):
    """Skin with a dark background strip d px wide at the block border (palm silhouette edge)."""
    xx = np.mgrid[0:N, 0:N][1]
    return _blur(_skin(rng) - contrast * (xx < d), 1.0) + rng.normal(0, 0.5, (N, N))


def _grating(period: float, amp: float, theta: float) -> np.ndarray:
    yy, xx = np.mgrid[0:N, 0:N]
    return amp * np.sin(2 * np.pi * (xx * np.cos(theta) + yy * np.sin(theta)) / period)


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def _rois(rng, n=12):
    return [_skin(rng) for _ in range(n)]


def test_grating_4px_amp10_lowers_score():
    rng = np.random.default_rng(0)
    base = _rois(rng)
    clean = np.stack([_u8(b) for b in base])
    screen = np.stack([_u8(b + _grating(4.0, 10.0, rng.uniform(0, np.pi))) for b in base])
    s_clean, r_clean, _ = evaluate(clean, CFG)
    s_screen, r_screen, d = evaluate(screen, CFG)
    assert r_clean == [] and r_screen == []
    assert s_clean > 0.8 and s_screen < 0.5
    assert s_clean - s_screen > 0.3
    assert d["n_roi"] == 12 and len(d["m"]) == 12


def test_in_band_gratings_raise_peak_ratio():
    rng = np.random.default_rng(1)
    band = CFG["texture"]["band"]
    for period in (2.5, 3.0, 4.0, 5.0, 6.0):          # 0.40 .. 0.17 cycles/px, inside the band
        for amp in (2.0, 5.0, 10.0):
            b = _skin(rng)
            m0 = block_peak_ratio(_u8(b), band)
            m1 = block_peak_ratio(_u8(b + _grating(period, amp, rng.uniform(0, np.pi))), band)
            assert m1 > 5 * m0, (period, amp, m0, m1)


def test_out_of_band_grating_ignored():
    rng = np.random.default_rng(2)
    band = CFG["texture"]["band"]
    b = _skin(rng)
    m0 = block_peak_ratio(_u8(b), band)
    m1 = block_peak_ratio(_u8(b + _grating(16.0, 10.0, 0.4)), band)   # 0.0625 cycles/px
    assert m1 < 2 * m0


def test_degenerate_inputs():
    rng = np.random.default_rng(3)
    good = np.stack([_u8(b) for b in _rois(rng, 4)])
    flat = np.full((6, N, N), 128, np.uint8)
    for roi in (np.zeros((0, N, N), np.uint8), good, flat):   # empty, < min_blocks, all constant
        score, reasons, debug = evaluate(roi, CFG)
        assert score is None and reasons == ["LOW_QUALITY"]
        json.dumps(debug, allow_nan=False)
    assert evaluate(flat, CFG)[2]["m"] == [None] * 6
    assert np.isnan(block_peak_ratio(flat[0], CFG["texture"]["band"]))


def test_constant_blocks_dropped_and_input_untouched():
    rng = np.random.default_rng(4)
    roi = np.stack([_u8(b) for b in _rois(rng, 5)] + [np.zeros((N, N), np.uint8)])
    before = roi.copy()
    score, reasons, debug = evaluate(roi, CFG)
    assert score is not None and reasons == []
    assert debug["n_roi"] == 5 and debug["m"][-1] is None
    json.dumps(debug, allow_nan=False)
    assert np.array_equal(roi, before)


def test_structured_skin_scores_high_and_grating_low():
    """Defaults must not fail real-looking palms (texture_floor 0.3) yet still catch a 4 px / amp 10 grating."""
    for gen in (_skin, _fractal_skin, _crease_skin, _edge_skin):
        for seed in range(3):
            rng = np.random.default_rng(seed)
            score, reasons, d = evaluate(np.stack([_u8(gen(rng)) for _ in range(12)]), CFG)
            assert reasons == [] and score > 0.75, (gen.__name__, seed, d["M"], score)
    rng = np.random.default_rng(5)
    screen = np.stack([_u8(_crease_skin(rng) + _grating(4.0, 10.0, rng.uniform(0, np.pi))) for _ in range(12)])
    assert evaluate(screen, CFG)[0] < 0.05


# --- formula pins: c / s are calibrated against this exact m, so pin it ---
BAND = CFG["texture"]["band"]


def test_known_answers():
    imp = np.zeros((N, N))
    imp[64, 64] = 255.0
    assert abs(block_peak_ratio(imp, BAND) - 1.0) < 1e-3              # flat spectrum -> m = 1
    # bin-centred cosine (k0 = 32 -> 0.25 c/px): exact value from the separable Hann spectrum
    W = np.fft.fft(np.hanning(N))
    pred = (np.abs(W) ** 2)[:, None] * (np.abs(0.5 * (np.roll(W, 32) + np.roll(W, -32))) ** 2)[None, :]
    f = np.fft.fftfreq(N)
    r = np.hypot(f[:, None], f[None, :])
    msk = (r >= BAND[0]) & (r <= BAND[1])
    xx = np.mgrid[0:N, 0:N][1]
    m = block_peak_ratio(np.cos(2 * np.pi * 32 * xx / N), BAND)
    assert abs(m / (pred[msk].max() / pred[msk].mean()) - 1) < 1e-9
    rng = np.random.default_rng(0)
    wn = np.median([block_peak_ratio(rng.normal(0, 5, (N, N)), BAND) for _ in range(50)])
    assert 7.5 < wn < 10.0                                             # max/mean of ~4.6k Exp(1) bins


def test_above_band_grating_ignored():
    rng = np.random.default_rng(0)
    xx = np.mgrid[0:N, 0:N][1]
    b = 118 + rng.normal(0, 2, (N, N))
    m0 = block_peak_ratio(_u8(b), BAND)
    m1 = block_peak_ratio(_u8(b + 10 * np.sin(2 * np.pi * xx / 2.05)), BAND)   # 0.488 c/px > 0.45
    assert m1 < 2 * m0, (m0, m1)


def test_median_over_blocks_and_exact_score():
    rng = np.random.default_rng(0)
    xx = np.mgrid[0:N, 0:N][1]
    base = [118 + rng.normal(0, 2, (N, N)) for _ in range(7)]
    g = 10 * np.sin(2 * np.pi * xx / 4.0)
    few = np.stack([_u8(b + g) if i < 3 else _u8(b) for i, b in enumerate(base)])    # 3/7 screen blocks
    many = np.stack([_u8(b + g) if i < 4 else _u8(b) for i, b in enumerate(base)])   # 4/7 screen blocks
    s_few, _, d = evaluate(few, CFG)
    s_many, _, _ = evaluate(many, CFG)
    assert s_few > 0.8 and s_many < 0.1
    c, s = CFG["texture"]["c"], CFG["texture"]["s"]
    assert abs(s_few - 1 / (1 + np.exp((d["M"] - c) / s))) < 1e-12
