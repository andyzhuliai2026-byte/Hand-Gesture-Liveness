"""Record -> save_sample -> load_sample -> session.replay gives the same result (PROMPT.md 7.4 roundtrip)."""
import re

import numpy as np
import pytest

import gesture
import liveness
import session as S
from config import config_sha8, load_config
from gesture_templates import template
from liveness.frame import Frame, empty_frame, save_sample
from tests.test_planarity import project, rot
from tests.test_score import fist_to_v
from tests.test_texture import _skin
from tools import evaluate as E

WH = (1280, 720)
FPS = 30
CONFIG = "config.yaml"


def hand(t, g, roi_mean=120.0):
    """Template hand projected at 1600 px/m: palm width ~0.11 W, so the 5.5 quality gate passes."""
    w = template(g)
    p = np.c_[w[:, 0] * 1600 + 640, w[:, 1] * 1600 + 360, w[:, 2] * 1600]
    return Frame(t, 1, True, p.astype(np.float32), w.astype(np.float32), 0.95, g, roi_mean, 0.0, WH)


def live_hand(t, t_rev, t_hold, roi_mean, cfg, rng):
    """test_score.live_round geometry for the real session: FIST, 0.3 s joint-angle move to V_SIGN starting
    0.6 s after the reveal, +-20 deg tilt during HOLD, palm pitched 10 deg; skin-like 128 px ROI block."""
    a = 0.0 if t_rev is None else float(np.clip((t - t_rev - 0.6) / 0.3, 0, 1))
    tilt = 0.0 if t_hold is None else 20.0 * np.sin(2 * np.pi * (t - t_hold) / 1.5)
    w = fist_to_v(a) @ (rot((0, 1, 0), tilt) @ rot((1, 0, 0), 10.0)).T
    px = project(w + [0.0, 0.0, 0.40])
    px[:, :2] += rng.normal(0, 1.0, (21, 2))
    w, px = w.astype(np.float32), px.astype(np.float32)
    fr = Frame(t, 1, True, px, w, 0.95, gesture.classify(w, px, cfg["gesture"]), roi_mean, 0.0, WH)
    return fr, np.clip(_skin(rng), 0, 255).astype(np.uint8)


def play_round(cfg, target, gesture, roi_mean=120.0, countdown="OPEN_PALM", live=False):
    """space, `countdown` hand until the reveal, then `gesture` (None = no hand) until RESULT.
    -> (in-round frames, RESULT event data). One 8x8 ROI block per frame, value = frame index.
    live: the V_SIGN round of live_hand instead (passes the M3 checks); target, gesture, countdown unused."""
    s = S.Session(cfg, lambda seq: liveness.score(seq, cfg), target="V_SIGN" if live else target)
    s.key("space")
    rng = np.random.default_rng(0)
    frames, i, t_rev, t_hold = [], 0, None, None
    while True:
        t = i / FPS
        if live:
            fr, roi = live_hand(t, t_rev, t_hold, roi_mean, cfg, rng)
        else:
            g = countdown if s.state in (S.IDLE, S.COUNTDOWN) else gesture
            fr, roi = hand(t, g, roi_mean) if g else empty_frame(t, WH), np.full((8, 8), i % 256, np.uint8)
        out = s.update(fr, roi)
        t_rev = t if t_rev is None and s.state not in (S.IDLE, S.COUNTDOWN) else t_rev
        t_hold = t if t_hold is None and s.state == S.HOLD else t_hold
        if out.in_round:
            frames.append(fr)
        res = [e for e in out.events if e.kind == "RESULT"]
        if res:
            return frames, res[0].data
        i += 1


def _plain(x):
    """numpy scalars -> python, so meta.json can be written."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x.item() if isinstance(x, np.generic) else x


def write_sample(d, cls, frames, data, sha8, variant="na", discard=False):
    """Sample dir in the 0.4 format (frames.npz + meta.json; no video)."""
    res, seq = data["result"], data["sequence"]
    meta = {"schema": 1, "class": cls, "medium": "na", "variant": variant, "lighting": "na", "hand": "na",
            "targets": data["targets"], "t0": data["t0"], "t_hold_start": data["t_hold_start"],
            "t_hold_end": data["t_hold_end"], "hold_match_ratio": data["hold_match_ratio"], "challenge_len": 1,
            "frame_wh": list(WH), "mediapipe_version": "test", "model_sha256": "test", "config_sha8": sha8,
            "live": {"verdict": res.verdict, "total": res.total, "scores": res.scores, "reasons": res.reasons,
                     "debug": res.debug},
            "discard": discard, "notes": "", "source": "synthetic", "n_frames": len(frames), "fps_nominal": 30.0,
            "created": "2026-11-04T00:00:00", "camera_index": 0}
    save_sample(d, frames, _plain(meta), seq.roi if seq else None, seq.roi_t if seq else None)
    return meta


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    cfg = load_config(CONFIG)
    frames, data = play_round(cfg, "V_SIGN", "V_SIGN", live=True)
    d = tmp_path_factory.mktemp("rt") / "real" / "20261104-000000"
    meta = write_sample(d, "real", frames, data, config_sha8(CONFIG))
    return d, meta, data


def edited_config(tmp_path, key, value):
    """config.yaml with the value of `key` replaced (by key, so a [calib] retune keeps this working),
    written to tmp_path."""
    text, n = re.subn(rf"(?m)^(\s*{key}:\s*)\S+", rf"\g<1>{value}", open(CONFIG, encoding="utf-8").read())
    assert n == 1
    p = tmp_path / "config.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_roundtrip_same_config(recorded):
    d, meta, data = recorded
    assert data["result"].verdict == "PASS" and data["sequence"].roi.shape[0] > 0
    frames, meta2, res, n_relabel = E.replay_sample(d, load_config(CONFIG))
    assert n_relabel == 0  # live labels come from classify on the same float32 landmarks frames.npz stores
    assert res.verdict == meta["live"]["verdict"] and res.reasons == meta["live"]["reasons"]
    assert all(np.allclose(res.debug[k], v, rtol=1e-5) for k, v in meta["live"]["debug"].items()
               if isinstance(v, float))
    assert meta2["hold_match_ratio"] == data["hold_match_ratio"] == res.debug["hold_match_ratio"]
    assert E.roundtrip(meta2["live"], res, meta2["config_sha8"] == config_sha8(CONFIG), n_relabel) == ("full", [])


def test_roundtrip_config_changed_compares_debug_only(recorded, tmp_path):
    d, meta, _ = recorded
    p = edited_config(tmp_path, "hold_match_min", 1.01)
    _, meta2, res, n_relabel = E.replay_sample(d, load_config(p))
    assert config_sha8(p) != meta2["config_sha8"] and res.reasons == ["HOLD_BROKEN"] and n_relabel == 0
    assert E.roundtrip(meta2["live"], res, False, n_relabel) == ("debug", [])
    assert "verdict" in E.roundtrip(meta2["live"], res, True)[1]


def test_roundtrip_gesture_change_is_label_drift(recorded, tmp_path):
    """A gesture key that moves labels: the replayed verdict follows the relabeled frames (recorded PASS ->
    INSTANT_SWITCH) and roundtrip reports drift, not debug mismatches. Session-side relabeling: test_evaluate."""
    d, meta, _ = recorded
    p = edited_config(tmp_path, "pip_extended_deg", 120)
    _, meta2, res, n_relabel = E.replay_sample(d, load_config(p))
    assert meta["live"]["verdict"] == "PASS" and n_relabel > 0
    assert (res.verdict, res.reasons) == ("FAIL", ["INSTANT_SWITCH"])
    assert E.roundtrip(meta2["live"], res, False, n_relabel) == ("drift", [f"{n_relabel} frames relabeled"])
    assert E.roundtrip(meta2["live"], None, False, n_relabel) == ("drift", [f"{n_relabel} frames relabeled",
                                                                           "no result"])
    same = f"labels differ from the recording under the same config ({n_relabel} frames)"
    assert same in E.roundtrip(meta2["live"], res, True, n_relabel)[1]
    assert E.roundtrip(meta2["live"], None, True, n_relabel) == ("replay", ["no result", same])
