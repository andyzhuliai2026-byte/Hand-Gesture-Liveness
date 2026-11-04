"""Synthetic tests for liveness.temporal (PROMPT.md 5.0 track switch, 5.1)."""
import json

import numpy as np
import pytest

from liveness import temporal
from liveness.frame import NONE, UNKNOWN

CFG = {"temporal": {"prerecorded_s": 0.3, "match_frames": 3, "window_s": 1.0, "max_gap_s": 0.15,
                    "f_delta": 0.5, "mid_band": [0.2, 0.8], "max_ext_rate": 15.0, "switch_jump": 1.0}}
T0 = 1.0
FIST = np.zeros(5)
OPEN = np.ones(5)
OK_E = np.array([0.3, 0.1, 1.0, 1.0, 1.0])  # thumb, index curled into the ring; others extended


def make_seq(fps, e_from, e_to, t_start, dur, target="OPEN_PALM", from_label="FIST",
             dropout=None, appear=None, t_end=3.0):
    """Linear e_from -> e_to over [t_start, t_start+dur] (dur 0 = hard cut). Label = target once e == e_to."""
    t = np.arange(0.0, t_end, 1.0 / fps)
    a = np.clip((t - t_start) / dur, 0, 1) if dur > 0 else (t >= t_start).astype(float)
    e = e_from + a[:, None] * (e_to - e_from)
    labels = np.where(a >= 1.0, target, np.where(a <= 0.0, from_label, UNKNOWN)).astype(object)
    present = np.ones(len(t), bool)
    if dropout is not None:
        present &= ~((t >= dropout[0]) & (t < dropout[1]))
    if appear is not None:
        present &= t >= appear
    e[~present] = np.nan
    labels[~present] = NONE
    return t, present, list(labels), e


def run(seq, target="OPEN_PALM", t_switch=()):
    t, present, labels, e = seq
    score, reasons, debug = temporal.evaluate(t, present, labels, e, target, T0, list(t_switch), CFG)
    json.dumps(debug, allow_nan=False)
    return score, reasons, debug


FPS = pytest.mark.parametrize("fps", [30, 15])


@FPS
def test_smooth_transition_passes(fps):
    score, reasons, d = run(make_seq(fps, FIST, OPEN, T0 + 0.5, 0.2))
    assert (score, reasons) == (1.0, [])
    assert d["F"] == [0, 1, 2, 3, 4] and d["max_ext_rate_observed"] <= 15.0
    assert d["t_r"] == pytest.approx(0.7, abs=1.5 / fps)


@FPS
def test_open_palm_to_ok_passes(fps):
    score, reasons, d = run(make_seq(fps, OPEN, OK_E, T0 + 0.6, 0.2, target="OK", from_label="OPEN_PALM"),
                            target="OK")
    assert (score, reasons) == (1.0, [])
    assert d["F"] == [0, 1]  # thumb 0.7, index 0.9; middle..pinky unchanged


@FPS
def test_hard_cut_is_instant_switch(fps):
    score, reasons, d = run(make_seq(fps, FIST, OPEN, T0 + 0.5, 0.0))
    assert (score, reasons) == (0.0, ["INSTANT_SWITCH"])
    assert d["mid_band_hit"] is False


def test_hard_cut_rate_exceeds_limit_at_30fps():
    d = run(make_seq(30, FIST, OPEN, T0 + 0.5, 0.0))[2]
    assert d["max_ext_rate_observed"] > 27  # a one-frame cut reads ~1/dt; real smoke transitions reach 31.7 /s too (PLAN dev-8)


@FPS
def test_dropout_mid_transition_is_hand_lost(fps):
    score, reasons, d = run(make_seq(fps, FIST, OPEN, T0 + 0.5, 0.4, dropout=(T0 + 0.55, T0 + 0.85)))
    assert (score, reasons) == (None, ["HAND_LOST"])
    assert d["gap_max_s"] > 0.3


@FPS
def test_target_throughout_is_no_transition(fps):
    seq = make_seq(fps, OPEN, OPEN, 0.0, 0.0, appear=T0 + 0.5)
    score, reasons, d = run(seq)
    assert (score, reasons) == (None, ["NO_TRANSITION"])
    assert d["F"] == []


@FPS
def test_fast_reaction_is_prerecorded(fps):
    score, reasons, d = run(make_seq(fps, FIST, OPEN, T0 - 0.1, 0.2))
    assert (score, reasons) == (0.0, ["PRERECORDED"])
    assert d["t_r"] == pytest.approx(0.1, abs=1.5 / fps)


def test_reaction_time_direction():
    """Earlier transition -> smaller t_r; crossing prerecorded_s flips PASS to PRERECORDED."""
    late = run(make_seq(30, FIST, OPEN, T0 + 0.3, 0.2))
    early = run(make_seq(30, FIST, OPEN, T0 + 0.0, 0.2))
    assert late[2]["t_r"] > early[2]["t_r"]
    assert late[0] == 1.0 and early[0] == 0.0


def test_switch_bounds_window():
    """A track switch at the first target frame drops the pre-switch frames: W holds only t_match."""
    t, p, labels, e = seq = make_seq(30, FIST, OPEN, T0 + 0.5, 0.2)
    k = labels.index("OPEN_PALM")
    score, reasons, d = run(seq, t_switch=[0.2, float(t[k]), 2.9])
    assert (score, reasons) == (None, ["NO_TRANSITION"])
    assert d["t_match"] == d["t_switch"] == d["W"][0] == float(t[k]) and d["F"] == []
    assert run(seq)[0] == 1.0  # control: same sequence without the switch passes


@pytest.mark.parametrize("seq", [
    (np.zeros(0), np.zeros(0, bool), [], np.zeros((0, 5))),
    (np.arange(5) / 30, np.zeros(5, bool), [NONE] * 5, np.full((5, 5), np.nan)),
    # target run of 2 < match_frames frames, timed after t0 so only the run length rejects it
    (T0 + 0.5 + np.arange(4) / 30, np.ones(4, bool), ["FIST", "OPEN_PALM", "OPEN_PALM", "FIST"], np.ones((4, 5))),
])
def test_degenerate_inputs_do_not_raise(seq):
    score, reasons, d = run(seq)
    assert score is None and reasons == ["NO_TRANSITION"] and d["t_match"] is None


def test_first_run_long_enough_is_the_match():
    """A 2-frame target run, one other frame, then a 3-frame run: t_match = first frame of the 3-frame run."""
    t = T0 + 0.5 + np.arange(8) / 30
    labels = ["FIST", "OPEN_PALM", "OPEN_PALM", UNKNOWN, "OPEN_PALM", "OPEN_PALM", "OPEN_PALM", "FIST"]
    assert temporal.first_match(t, np.ones(8, bool), labels, "OPEN_PALM", T0, CFG) == float(t[4])
    d = run((t, np.ones(8, bool), labels, np.ones((8, 5))))[2]
    assert d["t_match"] == float(t[4])


def seq_e(e, present=None, fps=30):
    """Sequence from a per-frame extension array on t = k/fps: OPEN_PALM when e == 1, FIST when e == 0."""
    t = np.arange(len(e)) / fps
    e = np.array(e, float)
    present = np.ones(len(t), bool) if present is None else np.asarray(present, bool)
    labels = np.where((e >= 1).all(1), "OPEN_PALM", np.where((e <= 0).all(1), "FIST", UNKNOWN)).astype(object)
    e[~present], labels[~present] = np.nan, NONE
    return t, present, list(labels), e


T30 = np.arange(0, 3.5, 1 / 30)
CUT = np.where(T30[:, None] >= T0 + 0.6, 1.0, 0.0) * np.ones((1, 5))   # FIST -> OPEN cut at T0+0.6
DROP_100MS = ~((T30 >= T0 + 0.5) & (T30 < T0 + 0.6))


def test_cut_hidden_in_dropout_fails_on_midband_alone():
    """(a) on its own: the cut spans a 100 ms dropout, so the rate 1.0/0.133 = 7.5/s passes (b)."""
    score, reasons, d = run(seq_e(CUT, DROP_100MS))
    assert (score, reasons) == (0.0, ["INSTANT_SWITCH"])
    assert d["mid_band_hit"] is False and d["max_ext_rate_observed"] <= 15.0 and d["gap_max_s"] <= 0.15


def test_midband_frame_but_too_fast_fails_on_rate_alone():
    """(b) on its own: 0 -> 0.3 -> 1.0 over two frames has a mid-band frame but a 21/s step."""
    e = np.zeros((len(T30), 5))
    k = np.searchsorted(T30, T0 + 0.5)
    e[k], e[k + 1:] = 0.3, 1.0
    score, reasons, d = run(seq_e(e))
    assert (score, reasons) == (0.0, ["INSTANT_SWITCH"])
    assert d["mid_band_hit"] is True and d["max_ext_rate_observed"] > 15.0


def test_non_F_finger_in_band_does_not_satisfy_midband():
    """A static thumb parked at 0.5 is not in F, so it cannot provide the intermediate state."""
    e = CUT.copy()
    e[:, 0] = 0.5
    t, p, _, e = seq_e(e, DROP_100MS)
    labels = [NONE if not q else ("OPEN_PALM" if x >= T0 + 0.6 else "FIST") for x, q in zip(t, p)]
    score, reasons, d = run((t, p, labels, e))
    assert (score, reasons) == (0.0, ["INSTANT_SWITCH"]) and 0 not in d["F"] and d["mid_band_hit"] is False


def test_static_start_pose_in_band_known_limitation():
    """Known limitation (PLAN S5 ②), not the desired outcome: a photo switch whose start pose has an F finger
    (thumb e 0.75) already in mid_band passes 5.1(a), since (a) counts any detected frame in W. Review evidence F1:
    the target pose at t_match satisfies (a) as well (74% of real V_SIGN frames have a mid-band finger)."""
    t = T30
    e = np.where((t >= T0 + 0.8)[:, None], [0.05, 0, 0, 0, 0], [0.75, 0.95, 0.95, 0.95, 0.95])
    p = ~((t >= T0 + 0.7) & (t < T0 + 0.8))
    labels = [NONE if not q else ("FIST" if x >= T0 + 0.8 else UNKNOWN) for x, q in zip(t, p)]
    e[~p] = np.nan
    score, reasons, d = run((t, p, labels, e), target="FIST")
    assert (score, reasons, d["mid_band_hit"]) == (1.0, [], True)


def test_transition_started_before_reveal_is_no_transition():
    """W starts at t0: a transition anticipated 0.5 s before the reveal leaves no change inside W."""
    a = np.clip((T30 - (T0 - 0.5)) / 0.9, 0, 1)
    score, reasons, d = run(seq_e(a[:, None] * np.ones((1, 5))))
    assert (score, reasons) == (None, ["NO_TRANSITION"]) and d["W"][0] == T0


def test_very_slow_transition_is_bounded_by_window_s():
    """W starts at t_match - window_s: a 2.3 s ramp changes < f_delta inside the last 1 s."""
    a = np.clip((T30 - (T0 + 0.1)) / 2.3, 0, 1)
    score, reasons, d = run(seq_e(a[:, None] * np.ones((1, 5))))
    assert (score, reasons) == (None, ["NO_TRANSITION"])
    assert d["W"][0] == pytest.approx(d["t_match"] - 1.0)


def test_gap_from_W_start_to_first_detection_is_ignored():
    """Hand enters 0.3 s after t0, then a smooth transition: the W-start gap is not HAND_LOST."""
    a = np.clip((T30 - (T0 + 0.6)) / 0.2, 0, 1)
    score, reasons, d = run(seq_e(a[:, None] * np.ones((1, 5)), T30 >= T0 + 0.3))
    assert (score, reasons) == (1.0, []) and d["gap_max_s"] <= 0.15


def test_prerecorded_beats_empty_window():
    """Check order: PRERECORDED decides even though W has no transition evidence."""
    score, reasons, d = run(seq_e(np.where(T30[:, None] >= T0 - 0.05, 1.0, 0.0) * np.ones((1, 5))))
    assert (score, reasons) == (0.0, ["PRERECORDED"]) and d["fired"] == ["PRERECORDED", "NO_TRANSITION"]


def test_absence_breaks_target_run():
    """One target frame, 0.5 s without a hand, then target: t_match is after the absence -> HAND_LOST."""
    a = np.clip((T30 - (T0 + 0.3)) / 0.2, 0, 1)
    k = int(np.flatnonzero(a >= 1)[0])
    p = np.ones(len(T30), bool)
    p[k + 1:k + 16] = False
    score, reasons, d = run(seq_e(a[:, None] * np.ones((1, 5)), p))
    assert (score, reasons) == (None, ["HAND_LOST"]) and d["t_match"] == pytest.approx(T30[k + 16])


def test_target_run_across_reveal_absence_is_not_prerecorded():
    """Target shown in the countdown, hand gone 1.1 s across the reveal, back in target: the pre-t0 run is
    broken by the absence, so t_match is the re-entry and nothing changed in W."""
    p = ~((T30 >= 0.5) & (T30 < T0 + 0.6))
    score, reasons, d = run(seq_e(np.ones((len(T30), 5)), p))
    assert (score, reasons) == (None, ["NO_TRANSITION"]) and d["t_match"] >= T0 + 0.6


def test_inputs_not_mutated():
    seq = make_seq(30, FIST, OPEN, T0 + 0.5, 0.2)
    copies = [np.array(x, copy=True) for x in seq]
    run(seq)
    for a, b in zip(seq, copies):
        np.testing.assert_array_equal(np.asarray(a), b)


# ---------------------------------------------------------------- track_switches (5.0)

def hand_at(cx, cy, width=100.0):
    """(21,3) landmarks with palm centre (cx, cy) and pixel palm width `width`."""
    lm = np.zeros((21, 3))
    lm[:, 0], lm[:, 1] = cx, cy
    lm[5, 0], lm[17, 0] = cx - width / 2, cx + width / 2
    lm[0, 1], lm[9, 1] = cy + width / 4, cy - width / 4
    lm[13, 0] = cx  # exact: mean of P0,P5,P9,P13,P17 x = cx
    return lm


def track(centres, present=None, fps=30):
    n = len(centres)
    t = np.arange(n) / fps
    present = np.ones(n, bool) if present is None else np.asarray(present)
    lm = np.stack([hand_at(*c) for c in centres])
    lm[~present] = np.nan
    return t, present, lm


def test_jump_over_one_palm_width_is_switch():
    t, p, lm = track([(300, 300)] * 5 + [(450, 300)] * 5)
    assert temporal.track_switches(t, p, lm, CFG) == [pytest.approx(t[5])]


def test_reentry_far_away_is_switch():
    p = [True] * 4 + [False] * 4 + [True] * 4
    t, p, lm = track([(300, 300)] * 4 + [(0, 0)] * 4 + [(300, 450)] * 4, p)
    assert temporal.track_switches(t, p, lm, CFG) == [pytest.approx(t[8])]


def test_small_motion_and_near_reentry_are_not_switches():
    centres = [(300 + 5 * i, 300) for i in range(10)]
    p = [True] * 4 + [False] * 3 + [True] * 3
    assert temporal.track_switches(*track(centres), CFG) == []
    assert temporal.track_switches(*track(centres, p), CFG) == []


@pytest.mark.parametrize("k, expect", [(0.9, 0), (1.1, 1)])
def test_switch_threshold_is_one_palm_width(k, expect):
    """Boundary: 0.9 palm widths is not a switch, 1.1 is, for consecutive frames and for re-entry."""
    t, p, lm = track([(300, 300)] * 3 + [(300 + 100 * k, 300)] * 3)
    assert len(temporal.track_switches(t, p, lm, CFG)) == expect
    t, p, lm = track([(300, 300)] * 3 + [(0, 0)] * 3 + [(300, 300 + 100 * k)] * 3, [1, 1, 1, 0, 0, 0, 1, 1, 1])
    assert len(temporal.track_switches(t, p.astype(bool), lm, CFG)) == expect


def test_switch_reference_width_is_last_known_frame():
    """Re-entry 150 px away at 3x the scale: 1.5 old widths (switch) though only 0.5 new widths."""
    lm = np.stack([hand_at(300, 300, 100)] * 3 + [np.full((21, 3), np.nan)] * 3 + [hand_at(450, 300, 300)] * 3)
    p = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1], bool)
    assert temporal.track_switches(np.arange(9) / 30, p, lm, CFG) == [pytest.approx(6 / 30)]


def test_track_switches_degenerate():
    assert temporal.track_switches(np.zeros(0), np.zeros(0, bool), np.zeros((0, 21, 3)), CFG) == []
    t, p, lm = track([(300, 300)] * 3, [False] * 3)
    assert temporal.track_switches(t, p, lm, CFG) == []
    t, p, lm = track([(300, 300)])
    assert temporal.track_switches(t, p, lm, CFG) == []
