import cv2
import numpy as np
import pytest

from liveness.frame import (NONE, UNKNOWN, Frame, crop_roi_gray, empty_frame, load_sample, palm_width_px,
                            save_sample)


def test_crop_clamps_to_border():
    img = np.random.default_rng(0).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    for center, (x0, y0) in (((5, 5), (0, 0)), ((1275, 715), (1280 - 128, 720 - 128))):
        roi = crop_roi_gray(img, center, 128)
        want = cv2.cvtColor(img[y0:y0 + 128, x0:x0 + 128], cv2.COLOR_BGR2GRAY)
        assert roi.shape == (128, 128) and roi.dtype == np.uint8
        assert np.abs(roi.astype(int) - want.astype(int)).max() <= 1


def test_crop_frame_too_small():
    with pytest.raises(ValueError):
        crop_roi_gray(np.zeros((100, 200, 3), np.uint8), (50, 50), 128)


def test_palm_width_falls_back_to_p0_p9():
    lm = np.zeros((21, 3))
    lm[5], lm[17] = [100, 100, 0], [101, 100, 0]  # squashed
    lm[0], lm[9] = [0, 0, 0], [0, 50, 0]
    assert palm_width_px(lm) == pytest.approx(50.0)


def _frame(t, rng, present):
    if not present:
        return empty_frame(t, (320, 240))
    return Frame(t, 1, True, rng.random((21, 3), dtype=np.float32), rng.random((21, 3), dtype=np.float32),
                 0.875, "FIST", 120.5, 0.0625, (320, 240))


def test_sample_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    frames = [_frame(0.0, rng, True), _frame(0.04, rng, False), _frame(0.09, rng, True)]
    meta = {"frame_wh": [320, 240], "note": "x"}
    save_sample(tmp_path, frames, meta)
    got, gmeta, _, _ = load_sample(tmp_path)
    assert gmeta == meta and len(got) == 3
    for a, b in zip(frames, got):
        for f in ("t", "n_hands", "present", "lm_px", "lm_world", "hand_score", "raw_label",
                  "roi_mean", "roi_sat_ratio", "frame_wh"):
            np.testing.assert_array_equal(getattr(a, f), getattr(b, f))
    assert got[1].raw_label == NONE and got[0].raw_label != UNKNOWN
