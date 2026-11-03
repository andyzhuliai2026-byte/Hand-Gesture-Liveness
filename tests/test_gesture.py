import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402
from gesture import LabelSmoother, analyze, classify  # noqa: E402
from gesture_templates import TEMPLATES, template  # noqa: E402
from liveness.frame import GESTURES, NONE, UNKNOWN  # noqa: E402

G = load_config(str(ROOT / "config.yaml"))["gesture"]

# extended (1) / curled (0) for index..pinky
FINGER_STATE = {
    "OPEN_PALM": (1, 1, 1, 1), "FIST": (0, 0, 0, 0), "THUMBS_UP": (0, 0, 0, 0),
    "V_SIGN": (1, 1, 0, 0), "POINT_UP": (1, 0, 0, 0), "OK": (0, 1, 1, 1),
}


def _rot(rng):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    a = np.radians(rng.uniform(-30, 30))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


def test_templates_classify():
    rng = np.random.default_rng(0)
    for g in GESTURES:
        for mirror in (False, True):
            for _ in range(10):
                p = template(g)
                if mirror:
                    p[:, 0] *= -1
                pw = np.linalg.norm(p[5] - p[17])
                p = p @ _rot(rng).T * rng.uniform(0.7, 1.3) + rng.uniform(-0.05, 0.05, 3)
                p += rng.normal(0, 0.01 * np.linalg.norm(p[5] - p[17]), p.shape)
                assert classify(p, G) == g, (g, mirror)


def test_template_margins():
    for g in GESTURES:
        a = analyze(TEMPLATES[g], G)
        for d, ext in zip(a.pip_deg, FINGER_STATE[g]):
            assert d >= 170 if ext else d <= 130, (g, a.pip_deg)
        if g in ("OPEN_PALM", "THUMBS_UP"):
            assert a.thumb_wrist_ratio >= G["thumb_ratio"] + 0.15
            assert a.thumb_cross_cos <= G["thumb_cross_cos"] - 0.3
        if g == "FIST":
            assert a.thumb_cross_cos >= G["thumb_cross_cos"] + 0.15
        if g == "OK":
            assert a.tip_gap_ratio <= G["ok_tip_ratio"] - 0.1


def test_folded_thumb_not_extended():
    fist = template("FIST")
    a = analyze(fist, G)
    assert a.thumb_wrist_ratio > G["thumb_ratio"] and a.label == "FIST"
    g2 = dict(G, thumb_cross_cos=1.01)  # direction test always true -> wrist-distance rule only
    assert classify(fist, g2) == "THUMBS_UP"


def test_extension_degree():
    for g in GESTURES:
        e = analyze(TEMPLATES[g], G).e
        assert np.all((e >= 0) & (e <= 1))
    assert np.all(analyze(TEMPLATES["OPEN_PALM"], G).e >= 0.95)
    assert np.all(analyze(TEMPLATES["FIST"], G).e[1:] <= 0.05)
    p = template("FIST")
    p[3, 1] = np.nan
    a = analyze(p, G)
    assert a.label == NONE and np.all(np.isnan(a.e))


def _feed(s, items):
    return [s.update(t, l) for t, l in items]


def test_smoother_majority():
    s = LabelSmoother(0.2)
    out = _feed(s, [(i / 30, l) for i, l in enumerate("AAAABB")])
    assert out[-1] == "A"


def test_smoother_tie_unknown():
    s = LabelSmoother(0.2)
    assert _feed(s, [(0.0, "A"), (0.05, "B")])[-1] == UNKNOWN


def test_smoother_none_votes():
    s = LabelSmoother(0.2)
    assert _feed(s, [(0.0, "A"), (0.03, NONE), (0.06, NONE)])[-1] == NONE


def test_smoother_old_labels_expire():
    s = LabelSmoother(0.2)
    _feed(s, [(i / 30, "A") for i in range(4)])
    out = _feed(s, [(0.2 + i / 30, "B") for i in range(10)])
    assert out[-1] == "B"


def test_smoother_15fps():
    s = LabelSmoother(0.2)
    out = _feed(s, [(k / 15, "A") for k in range(3)] + [(3 / 15, "B")])
    assert out[-1] == "A"
    out = _feed(s, [(k / 15, "B") for k in range(4, 7)])
    assert out[-1] == "B"


def test_smoother_time_backwards():
    s = LabelSmoother(0.2)
    s.update(1.0, "A")
    with pytest.raises(ValueError):
        s.update(0.9, "A")


def test_smoother_reset():
    s = LabelSmoother(0.2)
    s.update(1.0, "A")
    s.reset()
    assert s.update(0.0, "B") == "B"
