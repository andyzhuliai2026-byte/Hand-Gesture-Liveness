"""Record -> save_sample -> load_sample -> session.replay gives the same result (PROMPT.md 7.4 roundtrip)."""
import numpy as np
import pytest

import liveness
import session as S
from config import config_sha8, load_config
from gesture_templates import template
from liveness.frame import Frame, empty_frame, save_sample
from tools import evaluate as E

WH = (1280, 720)
FPS = 30
CONFIG = "config.yaml"


def hand(t, g, roi_mean=120.0):
    """Template hand projected at 1600 px/m: palm width ~0.11 W, so the 5.5 quality gate passes."""
    w = template(g)
    p = np.c_[w[:, 0] * 1600 + 640, w[:, 1] * 1600 + 360, w[:, 2] * 1600]
    return Frame(t, 1, True, p.astype(np.float32), w.astype(np.float32), 0.95, g, roi_mean, 0.0, WH)


def play_round(cfg, target, gesture, roi_mean=120.0, countdown="OPEN_PALM"):
    """space, `countdown` hand until the reveal, then `gesture` (None = no hand) until RESULT.
    -> (in-round frames, RESULT event data). One 8x8 ROI block per frame, value = frame index."""
    s = S.Session(cfg, lambda seq: liveness.score(seq, cfg), target=target)
    s.key("space")
    frames, i = [], 0
    while True:
        t = i / FPS
        g = countdown if s.state in (S.IDLE, S.COUNTDOWN) else gesture
        fr = hand(t, g, roi_mean) if g else empty_frame(t, WH)
        out = s.update(fr, np.full((8, 8), i % 256, np.uint8))
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
    frames, data = play_round(cfg, "V_SIGN", "V_SIGN")
    d = tmp_path_factory.mktemp("rt") / "real" / "20261104-000000"
    meta = write_sample(d, "real", frames, data, config_sha8(CONFIG))
    return d, meta, data


def test_roundtrip_same_config(recorded):
    d, meta, data = recorded
    assert data["result"].verdict == "PASS" and data["sequence"].roi.shape[0] > 0
    frames, meta2, res = E.replay_sample(d, load_config(CONFIG))
    assert res.verdict == meta["live"]["verdict"] and res.reasons == meta["live"]["reasons"]
    assert all(np.allclose(res.debug[k], v, rtol=1e-5) for k, v in meta["live"]["debug"].items()
               if isinstance(v, float))
    assert meta2["hold_match_ratio"] == data["hold_match_ratio"] == res.debug["hold_match_ratio"]
    assert E.roundtrip(meta2["live"], res, meta2["config_sha8"] == config_sha8(CONFIG)) == ("full", [])


def test_roundtrip_config_changed_compares_debug_only(recorded, tmp_path):
    d, meta, _ = recorded
    text = open(CONFIG, encoding="utf-8").read()
    assert "hold_match_min: 0.7 " in text
    p = tmp_path / "config.yaml"
    p.write_text(text.replace("hold_match_min: 0.7 ", "hold_match_min: 1.01"), encoding="utf-8")
    _, meta2, res = E.replay_sample(d, load_config(p))
    assert config_sha8(p) != meta2["config_sha8"] and res.reasons == ["HOLD_BROKEN"]
    assert E.roundtrip(meta2["live"], res, False) == ("debug", [])
    assert "verdict" in E.roundtrip(meta2["live"], res, True)[1]
