import dataclasses

import numpy as np
import pytest

from config import load_config
from liveness.frame import Frame, Sequence, empty_frame
from liveness.quality import HINT_BRIGHT, HINT_CLOSER, HINT_LIGHT, HINT_TWO_HANDS, assess, frame_hint, frame_ok

WH = (1280, 720)


@pytest.fixture
def q():
    return load_config("config.yaml")["quality"]


def good(t=0.0, palm_px=200.0, **kw):
    lm = np.zeros((21, 3), np.float32)
    lm[5], lm[17] = [500, 300, 0], [500 + palm_px, 300, 0]
    fr = Frame(t, 1, True, lm, np.zeros((21, 3), np.float32), 0.95, "OPEN_PALM", 120.0, 0.0, WH)
    return dataclasses.replace(fr, **kw)


def seq(frames, t0, t_hold_end):
    return Sequence(frames, np.zeros((0, 0, 0), np.uint8), np.zeros(0), ["OPEN_PALM"], t0, None, t_hold_end, None, WH)


def test_good_frame_ok(q):
    assert frame_ok(good(), q)


@pytest.mark.parametrize("bad", [
    {"present": False}, {"palm_px": 50.0}, {"roi_mean": 30.0}, {"roi_mean": 230.0},
    {"roi_sat_ratio": 0.2}, {"hand_score": 0.5}])
def test_each_condition_rejects(q, bad):
    palm = bad.pop("palm_px", 200.0)
    assert not frame_ok(good(palm_px=palm, **bad), q)


@pytest.mark.parametrize("field", ["roi_mean", "roi_sat_ratio", "hand_score"])
def test_nan_rejects(q, field):
    assert not frame_ok(good(**{field: float("nan")}), q)


def test_nan_landmarks_reject(q):
    fr = good()
    fr.lm_px[5] = np.nan
    assert not frame_ok(fr, q)


def test_hint_priority(q):
    assert frame_hint(good(palm_px=10.0, n_hands=2), q) == HINT_TWO_HANDS
    assert frame_hint(empty_frame(0.0, WH), q) is None
    assert frame_hint(good(palm_px=10.0, roi_mean=10.0), q) == HINT_CLOSER
    assert frame_hint(good(roi_mean=10.0), q) == HINT_LIGHT
    assert frame_hint(good(roi_sat_ratio=0.5), q) == HINT_BRIGHT
    assert frame_hint(good(), q) is None


def test_assess_window(q):
    frames = [good(t=0.1 * i, n_hands=2 if i == 5 else 1) for i in range(20)]  # t 0.0 .. 1.9
    s = assess(seq(frames, 0.5, 1.0), q)  # t 0.5..1.0 -> i = 5..10 (6 frames)
    assert (s.n_window, s.n_qualified) == (6, 6)
    assert s.two_hands_ratio == pytest.approx(1 / 6)
    assert s.qualified.dtype == bool and s.qualified.shape == (6,)


def test_assess_no_hold_end_uses_last_frame(q):
    frames = [good(t=0.1 * i) for i in range(10)]
    assert assess(seq(frames, 0.5, None), q).n_window == 5


def test_assess_empty_window(q):
    s = assess(seq([good(t=0.0)], 5.0, 6.0), q)
    assert (s.n_window, s.qualified_ratio, s.two_hands_ratio) == (0, 0.0, 0.0)
