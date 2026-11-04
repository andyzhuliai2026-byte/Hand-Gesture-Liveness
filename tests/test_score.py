import dataclasses
import json

import numpy as np
import pytest

from config import load_config
import gesture_templates as gt
from liveness import INSUFFICIENT_ORDER, REASON_TEXT, nonrigid, planarity, score
from liveness.frame import Frame, Sequence, empty_frame
from liveness.fusion import fuse
from tests.test_planarity import flatten, project, rot
from tests.test_texture import _grating, _skin

WH = (1280, 720)
FPS = 30.0
T0, T_MOVE, MOVE_S = 1.0, 1.6, 0.3   # countdown ends at T0; FIST -> V_SIGN over [T_MOVE, T_MOVE + MOVE_S]
HOLD0 = T_MOVE + MOVE_S + 0.5        # after the 0.5 s stable hold
HOLD1 = HOLD0 + 1.5


@pytest.fixture
def cfg():
    return load_config("config.yaml")


def fist_to_v(a: float) -> np.ndarray:
    """FIST (a=0) -> V_SIGN (a=1) by joint angles, as a finger moves; built like gesture_templates._build."""
    pts = np.zeros((21, 3))
    for k in range(4):
        uncurl = k < 2  # index, middle
        flex = (1 - a) * np.array(gt._CURL) + a * np.array(gt._EXT) if uncurl else gt._CURL
        spread = (1 - a) * gt._SPREAD[k] + a * gt._V_SPREAD[k] if uncurl else gt._SPREAD[k]
        pts[gt.FINGERS[k + 1], :] = gt._finger(k, flex, spread)
    pts[gt.FINGERS[0], :] = [gt._T1, *map(np.array, gt._T_FOLDED)]
    return pts - pts.mean(axis=0)


def test_fist_to_v_endpoints():
    assert np.allclose(fist_to_v(0.0), gt.template("FIST")) and np.allclose(fist_to_v(1.0), gt.template("V_SIGN"))


def live_round(tilt=20.0, pitch=10.0, cut=False, flat=False, bad=(0, 0), two=(0, 0), gap=(0, 0), hold=0.95, seed=0,
               fps=None, t_move=None, tilt_fn=None, jump=None, screen=False):
    """Synthetic round on the pinhole camera of test_planarity: FIST -> V_SIGN, then HOLD with a +-tilt deg turn.

    pitch: palm normal this far off the optical axis (a squarely facing palm sits in the 2D singularity, see below).

    cut: one-frame switch (photo swap). flat: from HOLD on, the image comes from the flattened template (a photo
    swapped in after a live transition) while lm_world stays 3D (MediaPipe hallucinates depth).
    bad / two / gap: [lo, hi) seconds with hand_score 0.1 / 2 hands / no hand.
    t_move: transition start. tilt_fn: t -> y-turn deg, replaces the HOLD sine. jump: from this t on the image
    shifts 300 px (track switch). screen: ROI blocks carry a 4 px grating (screen replay, test_texture).
    """
    fps = FPS if fps is None else fps  # ponytail: read at call time, so probes that patch FPS / T_MOVE still work
    t_move = T_MOVE if t_move is None else t_move
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, HOLD1 + 1e-9, 1 / fps)
    a = (t >= t_move + MOVE_S).astype(float) if cut else np.clip((t - t_move) / MOVE_S, 0, 1)
    frames, roi, roi_t = [], [], []
    base = rot((1, 0, 0), pitch)
    for tk, ak in zip(t, a):
        if gap[0] <= tk < gap[1]:
            frames.append(empty_frame(float(tk), WH))
            continue
        deg = tilt_fn(tk) if tilt_fn else (tilt * np.sin(2 * np.pi * (tk - HOLD0) / 1.5) if tk >= HOLD0 else 0.0)
        r = rot((0, 1, 0), deg) @ base
        w = fist_to_v(ak)
        px = project((flatten(w) if flat and tk >= HOLD0 else w) @ r.T + [0.0, 0.0, 0.40])
        if jump is not None and tk >= jump:
            px[:, 0] += 300.0
        px[:, :2] += rng.normal(0, 1.0, (21, 2))
        frames.append(Frame(float(tk), 2 if two[0] <= tk < two[1] else 1, True, px.astype(np.float32),
                            (w @ r.T).astype(np.float32), 0.1 if bad[0] <= tk < bad[1] else 0.95, "UNKNOWN",
                            120.0, 0.0, WH))
        if tk >= T0 and (not roi_t or tk - roi_t[-1] >= 0.1 - 1e-9):
            roi.append(np.clip(_skin(rng) + (_grating(4.0, 10.0, 0.0) if screen else 0.0), 0, 255).astype(np.uint8))
            roi_t.append(float(tk))
    return Sequence(frames, np.stack(roi), np.array(roi_t), ["V_SIGN"], T0, HOLD0, float(t[-1]), hold, WH)


def test_live_round_passes(cfg):
    r = score(live_round(), cfg)
    assert (r.verdict, r.reasons) == ("PASS", []) and r.total >= cfg["fusion"]["pass_total"]
    assert all(v is not None for v in r.scores.values())
    assert not (r.debug["veto_prerecorded"] or r.debug["veto_instant_switch"] or r.debug["veto_planar"])
    assert r.debug["n_window"] == 88  # frames in [T0, hold end]; c726dd7 test_pass pinned n_window too


def test_debug_keys_and_json(cfg):
    d = score(live_round(), cfg).debug
    need = {"t_r", "F", "max_ext_rate_observed", "gap_max_s", "t_switch", "r", "R", "rho", "n_pairs",
            "normal_angles", "M", "n_roi", "qualified_ratio", "two_hands_ratio", "hold_match_ratio",  # PROMPT 5.6
            "veto_prerecorded", "veto_instant_switch", "veto_planar"}                             # tools/evaluate
    assert need <= d.keys()
    json.dumps(d)  # sessions.jsonl line


def test_flat_photo_is_planar(cfg):
    r = score(live_round(flat=True), cfg)
    assert (r.verdict, r.total, r.reasons) == ("FAIL", None, ["PLANAR"]) and r.debug["veto_planar"]


def test_photo_switch_is_instant(cfg):
    r = score(live_round(cut=True), cfg)
    assert (r.verdict, r.total, r.reasons) == ("FAIL", None, ["INSTANT_SWITCH"]) and r.debug["veto_instant_switch"]


# today's [calib] values as literals, so a later recalibration cannot flip the characterization test below
PINNED_GESTURE = {"pip_extended_deg": 160, "pip3d_min_deg": 110, "thumb_tip_ratio": 0.75, "ok_tip_ratio": 0.3,
                  "ext_pip_deg": [90, 170], "ext_pip3d_deg": [100, 140], "ext_thumb_ratio": [0.4, 0.9],
                  "smooth_window_s": 0.2}


def test_facing_palm_2d_step_known_limitation(cfg):
    """Known limitation (PLAN dev-8, DESIGN section 7), not the desired outcome: at pitch 0 the 2D PIP angle steps
    across the band in one frame (the step is governed by yaw: the flexion plane contains the line of sight), and
    e = min(e2d, e3d) inherits the step, so 5.1(a) sees no mid-band frame and a live transition reads as a switch."""
    pinned = {**cfg, "gesture": PINNED_GESTURE, "temporal": {**cfg["temporal"], "max_ext_rate": 41.0}}
    r = score(live_round(pitch=0.0, seed=0), pinned)
    assert (r.verdict, r.reasons, r.debug["mid_band_hit"]) == ("FAIL", ["INSTANT_SWITCH"], False)


def test_veto_flag_survives_insufficient(cfg):
    # photo swap hidden in a 0.2 s detection gap: HAND_LOST decides, the 7.4 veto hit rate still sees the switch
    cut = T_MOVE + MOVE_S
    r = score(live_round(cut=True, gap=(cut - 0.1, cut + 0.1)), cfg)
    assert (r.verdict, r.reasons) == ("INSUFFICIENT", ["HAND_LOST"]) and r.debug["veto_instant_switch"]


def test_no_tilt_needs_rotation(cfg):
    r = score(live_round(tilt=0.0), cfg)
    assert (r.verdict, r.reasons, r.scores["planarity"]) == ("INSUFFICIENT", ["NEED_ROTATION"], None)


def test_low_quality_ratio(cfg):
    r = score(live_round(bad=(T0, T0 + 1.2)), cfg)  # 36 of 88 window frames fail -> ratio 0.59 < 0.6
    assert (r.verdict, r.total, r.reasons) == ("INSUFFICIENT", None, ["LOW_QUALITY"])


def test_two_hands(cfg):
    assert score(live_round(two=(T0, T0 + 0.7)), cfg).reasons == ["TWO_HANDS"]


def test_hold_broken(cfg):
    assert score(live_round(hold=0.5), cfg).reasons == ["HOLD_BROKEN"]
    assert score(live_round(hold=None), cfg).reasons == ["HOLD_BROKEN"]


def test_insufficient_order(cfg):
    r = score(live_round(bad=(T0, T0 + 1.2), two=(T0, T0 + 0.7), hold=0.1), cfg)
    assert r.reasons == ["LOW_QUALITY", "TWO_HANDS", "HOLD_BROKEN"]
    assert r.reasons == [c for c in INSUFFICIENT_ORDER if c in r.reasons]


def test_rotation_before_hold_needs_rotation(cfg):
    # 5.3 pairs come from HOLD only: a turn during the stable hold before HOLD, none inside it
    pre = lambda tk: 20.0 * np.sin(2 * np.pi * (tk - HOLD0 + 0.5) / 0.5) if HOLD0 - 0.5 <= tk < HOLD0 else 0.0
    r = score(live_round(tilt_fn=pre), cfg)
    assert (r.verdict, r.reasons, r.debug["n_pairs"]) == ("INSUFFICIENT", ["NEED_ROTATION"], 0)


def test_rotation_on_failed_frames_needs_rotation(cfg):
    # 5.5: a quick 20 deg tilt inside HOLD, seen only on hand_score 0.1 frames, gives no planarity pairs
    b0, b1 = HOLD0 + 0.5, HOLD0 + 0.9
    quick = lambda tk: 20.0 * np.sin(np.pi * (tk - b0) / (b1 - b0)) if b0 <= tk < b1 else 0.0
    r = score(live_round(tilt_fn=quick, bad=(b0, b1)), cfg)
    assert r.debug["qualified_ratio"] >= cfg["quality"]["qualified_ratio_min"]
    assert r.debug["n_qualified"] >= cfg["quality"]["qualified_min_frames"]
    assert (r.verdict, r.reasons, r.debug["n_pairs"]) == ("INSUFFICIENT", ["NEED_ROTATION"], 0)


def test_track_switch_bounds_window(cfg):
    # 5.0: a 300 px palm jump mid-transition starts W at the switch, after the change -> no transition seen
    r = score(live_round(jump=T_MOVE + 0.2), cfg)
    assert (r.verdict, r.reasons) == ("INSUFFICIENT", ["NO_TRANSITION"]) and r.debug["t_switch"] is not None


def test_prerecorded(cfg):
    # target shown 0.2 s before t0 (t_r < 0.3 s): temporal's deciding code, not every fired code, reaches fuse
    r = score(live_round(t_move=0.5), cfg)
    assert (r.verdict, r.total, r.reasons) == ("FAIL", None, ["PRERECORDED"]) and r.debug["veto_prerecorded"]


def test_prerecorded_flag_survives_insufficient(cfg):
    r = score(live_round(t_move=0.5, two=(T0, T0 + 0.7)), cfg)
    assert (r.verdict, r.reasons) == ("INSUFFICIENT", ["TWO_HANDS"]) and r.debug["veto_prerecorded"]


def test_low_quality_count(cfg):
    # 5.5 count rule: 7.5 fps, ratio >= 0.6 but fewer than 15 qualified frames
    r = score(live_round(fps=7.5, bad=(T0, T0 + 1.1)), cfg)
    assert r.debug["qualified_ratio"] >= cfg["quality"]["qualified_ratio_min"]
    assert r.debug["n_qualified"] < cfg["quality"]["qualified_min_frames"]
    assert (r.verdict, r.total, r.reasons) == ("INSUFFICIENT", None, ["LOW_QUALITY"])


def test_failed_frames_leave_nonrigid_and_texture(cfg):
    # 5.5: failed frames never enter 5.2 / 5.4 (45 HOLD frames, 15 failed; 30 ROI blocks, 5 failed)
    assert score(live_round(bad=(HOLD0, HOLD0 + 0.5)), cfg).debug["n_nonrigid"] == 30
    assert score(live_round(bad=(T0, T0 + 0.5)), cfg).debug["n_roi"] == 25


def test_screen_roi_fails_texture(cfg):
    r = score(live_round(screen=True), cfg)
    assert (r.verdict, r.reasons) == ("FAIL", ["SCREEN"])


def test_no_roi_is_low_quality(cfg):
    seq = dataclasses.replace(live_round(), roi=np.zeros((0, 0, 0), np.uint8), roi_t=np.zeros(0))
    r = score(seq, cfg)
    assert (r.verdict, r.reasons, r.scores["texture"]) == ("INSUFFICIENT", ["LOW_QUALITY"], None)


def test_nonrigid_none_is_low_quality(cfg, monkeypatch):
    # 5.2 has no reason code of its own; a None score must not reach fuse() (it raises on None)
    monkeypatch.setattr(nonrigid, "evaluate", lambda lm, c: (None, [], {"r": None, "n_frames": 0}))
    r = score(live_round(), cfg)
    assert (r.verdict, r.reasons, r.scores["nonrigid"]) == ("INSUFFICIENT", ["LOW_QUALITY"], None)



def test_planarity_score_reaches_fuse(cfg, monkeypatch):
    # 5.3 score routing: a low score with no PLANAR code fails through planarity_floor, not the veto
    real = planarity.evaluate
    monkeypatch.setattr(planarity, "evaluate", lambda *a: (0.2,) + real(*a)[1:])
    r = score(live_round(), cfg)
    assert (r.verdict, r.reasons, r.debug["veto_planar"]) == ("FAIL", ["PLANAR"], False)


def test_nonrigid_score_reaches_fuse(cfg, monkeypatch):
    # 5.2 score routing: nonrigid 0.0 contributes nothing to the 5.6 total
    real = nonrigid.evaluate
    monkeypatch.setattr(nonrigid, "evaluate", lambda *a: (0.0,) + real(*a)[1:])
    r, f = score(live_round(), cfg), cfg["fusion"]
    assert r.scores["nonrigid"] == 0.0
    assert r.total == pytest.approx(f["w_planarity"] * r.scores["planarity"] + f["w_texture"] * r.scores["texture"])

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


def test_fuse_weighted_loss_reason():
    # [brief 5.6] initial values, fixed here so calibration cannot move the case; reason = argmax w_k * (1 - score_k)
    f = {"w_planarity": 0.65, "w_texture": 0.20, "w_nonrigid": 0.15, "planarity_floor": 0.4, "texture_floor": 0.3,
         "pass_total": 0.6}
    assert fuse(_s(0.79, 0.31, 0.1), [], [], f)[0::2] == ("FAIL", ["SCREEN"])  # losses .1365 / .138 / .135
    assert fuse(_s(0.78, 0.4, 0.0), [], [], f)[0::2] == ("FAIL", ["STATIC"])   # losses .143 / .12 / .15
    assert fuse(_s(1.0, 0.3, 1.0), [], [], f)[0::2] == ("PASS", [])            # score == floor clears it (5.6: >=)
    assert fuse(_s(0.4, 1.0, 1.0), [], [], f)[0::2] == ("PASS", [])            # total .61


def test_fuse_order(cfg):
    f = cfg["fusion"]
    assert fuse(_s(), ["HOLD_BROKEN", "LOW_QUALITY"], ["PLANAR"], f) == ("INSUFFICIENT", None, ["LOW_QUALITY", "HOLD_BROKEN"])
    assert fuse(_s(), [], ["PLANAR", "PRERECORDED"], f) == ("FAIL", None, ["PRERECORDED", "PLANAR"])


def test_fuse_none_score_raises(cfg):
    with pytest.raises(ValueError):
        fuse(_s(p=None), [], [], cfg["fusion"])


def test_reason_text():
    assert len(REASON_TEXT) == 13 and all(v.isascii() for v in REASON_TEXT.values())
