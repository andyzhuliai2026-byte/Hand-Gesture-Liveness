import pytest

from config import load_config
from liveness import INSUFFICIENT_ORDER, REASON_TEXT, score
from liveness.fusion import fuse
from tests.test_quality import good, seq as _seq


@pytest.fixture
def cfg():
    return load_config("config.yaml")


def mkseq(n=30, n_bad=0, n_two=0, hold=0.9):
    frames = [good(t=0.05 * i, n_hands=2 if i < n_two else 1, hand_score=0.1 if i < n_bad else 0.95)
              for i in range(n)]
    s = _seq(frames, 0.0, frames[-1].t)
    s.hold_match_ratio = hold
    return s


def test_pass(cfg):
    r = score(mkseq(), cfg)
    assert (r.verdict, r.total, r.reasons) == ("PASS", pytest.approx(1.0), [])
    assert r.debug["placeholder_scores"] is True and r.debug["n_window"] == 30


def test_low_quality_count(cfg):
    assert score(mkseq(n=10), cfg).reasons == ["LOW_QUALITY"]


def test_low_quality_ratio(cfg):
    r = score(mkseq(n=40, n_bad=20), cfg)  # 20 qualified >= 15, ratio 0.5 < 0.6
    assert (r.verdict, r.total, r.reasons) == ("INSUFFICIENT", None, ["LOW_QUALITY"])


def test_two_hands(cfg):
    assert score(mkseq(n=30, n_two=9), cfg).reasons == ["TWO_HANDS"]


def test_hold_broken(cfg):
    assert score(mkseq(hold=0.5), cfg).reasons == ["HOLD_BROKEN"]
    assert score(mkseq(hold=None), cfg).reasons == ["HOLD_BROKEN"]


def test_insufficient_order(cfg):
    r = score(mkseq(n=10, n_two=5, hold=0.1), cfg)
    assert r.reasons == ["LOW_QUALITY", "TWO_HANDS", "HOLD_BROKEN"]
    assert r.reasons == [c for c in INSUFFICIENT_ORDER if c in r.reasons]


def _s(p=1.0, t=1.0, n=1.0):
    return {"temporal": 1.0, "planarity": p, "texture": t, "nonrigid": n}


def test_fuse_floors(cfg):
    f = cfg["fusion"]
    assert fuse(_s(p=0.3), [], [], f)[0::2] == ("FAIL", ["PLANAR"])
    assert fuse(_s(t=0.2), [], [], f)[0::2] == ("FAIL", ["SCREEN"])
    assert fuse(_s(p=0.3, t=0.2), [], [], f)[2] == ["PLANAR", "SCREEN"]


def test_fuse_total_below_pass(cfg):
    v, total, reasons = fuse(_s(0.55, 0.55, 0.55), [], [], cfg["fusion"])
    assert (v, reasons) == ("FAIL", ["PLANAR"]) and total == pytest.approx(0.55)


def test_fuse_order(cfg):
    f = cfg["fusion"]
    assert fuse(_s(), ["HOLD_BROKEN", "LOW_QUALITY"], ["PLANAR"], f) == ("INSUFFICIENT", None, ["LOW_QUALITY", "HOLD_BROKEN"])
    assert fuse(_s(), [], ["PLANAR", "PRERECORDED"], f) == ("FAIL", None, ["PRERECORDED", "PLANAR"])


def test_fuse_none_score_raises(cfg):
    with pytest.raises(ValueError):
        fuse(_s(p=None), [], [], cfg["fusion"])


def test_reason_text():
    assert len(REASON_TEXT) == 13 and all(v.isascii() for v in REASON_TEXT.values())
