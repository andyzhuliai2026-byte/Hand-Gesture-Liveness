import copy
import random

import numpy as np
import pytest

import liveness
import session as S
from config import load_config
from gesture_templates import template
from liveness import LivenessResult
from liveness.frame import GESTURES, Frame, empty_frame

WH = (1280, 720)
FPS = 30


@pytest.fixture
def cfg():
    return load_config("config.yaml")


@pytest.fixture
def cfg_raw(cfg):
    """Smoothing window shorter than a frame: smooth label == raw label, so gap timing is exact."""
    c = copy.deepcopy(cfg)
    c["gesture"]["smooth_window_s"] = 1e-3
    return c


def px(w):
    """Orthographic projection of world landmarks to pixel coordinates."""
    return np.c_[w[:, 0] * 800 + 640, w[:, 1] * 800 + 360, w[:, 2] * 800]


def hand(t, g, raw=None):
    w = template(g)
    return Frame(t, 1, True, px(w).astype(np.float32), w.astype(np.float32), 0.95, raw or g, 120.0, 0.0, WH)


class FakeScore:
    def __init__(self):
        self.calls = []

    def __call__(self, seq):
        self.calls.append(seq)
        return LivenessResult("PASS", 1.0, {"temporal": 1.0}, [], {})


class Run:
    """Drives a Session with synthetic frames at a fixed rate; t = i / fps."""

    def __init__(self, s, fps=FPS):
        self.s, self.fps, self.i, self.outs = s, fps, 0, []

    @property
    def t(self):
        return self.i / self.fps

    def step(self, g=None, raw=None, roi=None):
        fr = hand(self.t, g, raw) if g else empty_frame(self.t, WH)
        out = self.s.update(fr, roi)
        self.i += 1
        self.outs.append(out)
        return out

    def until(self, pred, g=None, **kw):
        while True:
            out = self.step(g, **kw)
            if pred(out):
                return out

    def ev(self, kind):
        return [e for o in self.outs for e in o.events if e.kind == kind]

    def start(self, g=None):
        """space, then countdown frames of g up to and including the REVEAL frame."""
        self.s.key("space")
        return self.until(lambda o: o.phase == S.CHALLENGE, g)

    def to_hold(self, target, g_on_entry=None):
        """Target frames until the stable hold is about to complete; the completing frame is g_on_entry
        (None = no hand, absorbed by the gap tolerance). Needs cfg_raw."""
        while self.s.state == S.CHALLENGE:
            st = self.s._round.stable_start
            done = st is not None and self.t - st >= self.s._s["stable_hold_s"]
            self.step(g_on_entry if done else target)
        assert self.s.state == S.HOLD


def test_countdown_then_reveal(cfg):
    run = Run(S.Session(cfg, FakeScore(), target="FIST"))
    run.start()
    assert all(o.phase == S.COUNTDOWN for o in run.outs[:-1]) and run.outs[-2].t < 1.0
    reveal = run.ev("REVEAL")[0]
    assert run.outs[-1].t == 1.0 and reveal.data["t0"] == 1.0 and run.ev("ROUND_START")[0].t == 0.0


def _three_targets(cfg, seed):
    run = Run(S.Session(cfg, FakeScore(), seed=seed))
    for _ in range(3):
        run.start()
        run.s.key("r")
        run.step()
    return [e.data["targets"][0] for e in run.ev("REVEAL")]


def test_fixed_target_and_seed(cfg):
    run = Run(S.Session(cfg, FakeScore(), target="FIST"))
    assert run.start().target == "FIST" and run.ev("REVEAL")[0].data["targets"] == ["FIST"]
    assert _three_targets(cfg, 7) == _three_targets(cfg, 7)


def test_reveal_excludes_current_gesture(cfg):
    for seed in range(20):
        run = Run(S.Session(cfg, FakeScore(), seed=seed))
        assert run.start("OPEN_PALM").target != "OPEN_PALM"


def test_reveal_excludes_by_extension(cfg):
    # labels say UNKNOWN, only e matches: OPEN_PALM e = 1,1,1,1,1 also matches OK (thumb, index not judged)
    for seed in range(20):
        run = Run(S.Session(cfg, FakeScore(), seed=seed))
        run.s.key("space")
        run.until(lambda o: o.phase == S.CHALLENGE, "OPEN_PALM", raw="UNKNOWN")
        assert run.outs[-1].target not in ("OPEN_PALM", "OK")


def test_reveal_all_excluded_draws_from_all(cfg, monkeypatch):
    monkeypatch.setattr(S, "_excluded", lambda *a: set(GESTURES))
    run = Run(S.Session(cfg, FakeScore(), seed=3))
    assert run.start("OPEN_PALM").target == random.Random(3).choice(sorted(GESTURES))


def test_first_match(cfg):
    run = Run(S.Session(cfg, FakeScore(), target="V_SIGN"))
    run.start()
    for g in ("V_SIGN", "V_SIGN", None, "V_SIGN", "V_SIGN", "OPEN_PALM"):
        assert not run.step(g).first_match
    t_seg = run.t
    outs = [run.step("V_SIGN") for _ in range(3)]
    assert [o.first_match for o in outs] == [False, False, True] and run.s._round.t_match == t_seg


def test_first_match_timeout(cfg):
    fake = FakeScore()
    run = Run(S.Session(cfg, fake, target="V_SIGN"))
    run.start()
    out = run.until(lambda o: o.phase == S.RESULT)  # no hand at all
    assert 5.0 < out.t <= 5.0 + 1 / FPS and run.outs[-2].t <= 5.0
    assert (out.verdict, out.reasons, out.scores, out.total) == ("FAIL", ["TIMEOUT"], {}, None)
    assert run.ev("RESULT")[0].data["sequence"] is None and fake.calls == []

    run = Run(S.Session(cfg, FakeScore(), target="V_SIGN"))  # wrong gesture throughout: 4.0 s branch, WRONG_GESTURE
    run.start("OPEN_PALM")
    out = run.until(lambda o: o.phase == S.RESULT, "OPEN_PALM")
    assert 5.0 < out.t <= 5.0 + 1 / FPS and out.reasons == ["WRONG_GESTURE"]


def _first_match_late(cfg, seg_start):
    """No hand until seg_start (frame index), then V_SIGN; t0 = 1.0, so the first-match deadline is t = 5.0."""
    run = Run(S.Session(cfg, FakeScore(), target="V_SIGN"))
    run.start()
    while run.i < seg_start and run.s.state == S.CHALLENGE:
        run.step()
    for _ in range(3):
        if run.s.state != S.CHALLENGE:
            break
        run.step("V_SIGN")
    return run


def test_first_match_run_started_by_deadline(cfg):
    run = _first_match_late(cfg, 150)  # run starts at t = 150/30 = 5.0 == t0 + 4.0: still in time
    assert run.outs[-3].t == 5.0 and run.outs[-1].first_match and run.s._round.t_match == 5.0
    assert run.s.state == S.CHALLENGE and not run.ev("RESULT")
    run.until(lambda o: o.phase != S.CHALLENGE, "V_SIGN")
    assert run.s.state == S.HOLD

    run = _first_match_late(cfg, 151)  # run starts at 5.033 > 5.0: timeout on that frame
    out = run.outs[-1]
    assert out.phase == S.RESULT and out.t == 151 / FPS and out.reasons == ["TIMEOUT"] and not out.first_match


def test_no_hold_without_first_match(cfg):
    # raw V V U repeating: never 3 in a row, yet the 0.2 s majority vote stays V_SIGN all along
    run = Run(S.Session(cfg, FakeScore(), target="V_SIGN"))
    run.s.key("space")
    while run.s.state != S.RESULT:
        run.step("V_SIGN", raw=("V_SIGN", "V_SIGN", "UNKNOWN")[run.i % 3])
    ch = [o for o in run.outs if o.phase == S.CHALLENGE]
    assert all(o.smooth_label == "V_SIGN" for o in ch) and max(o.stable_progress for o in ch) == 1.0
    assert not run.ev("HOLD_START") and run.outs[-1].reasons == ["TIMEOUT"] and not run.outs[-1].first_match


def test_wrong_gesture(cfg):
    run = Run(S.Session(cfg, FakeScore(), target="V_SIGN"))
    run.start()
    for _ in range(3):
        run.step("V_SIGN")
    out = run.until(lambda o: o.phase == S.RESULT, "OPEN_PALM")
    assert out.first_match and 6.0 < out.t <= 6.0 + 1 / FPS and out.reasons == ["WRONG_GESTURE"]


def test_wrong_gesture_uses_smoothed_labels(cfg):
    c = copy.deepcopy(cfg)
    c["session"]["wrong_gesture_frac"] = 0.9
    c["gesture"]["smooth_window_s"] = 0.21  # exactly 7 frames at 30 fps, clear of float edges
    run = Run(S.Session(c, FakeScore(), target="V_SIGN"))
    run.s.key("space")
    while run.s.state != S.RESULT:  # raw OPEN_PALM x4, UNKNOWN x3: raw 57 %, smoothed 100 % OPEN_PALM
        run.step("OPEN_PALM", raw="OPEN_PALM" if run.i % 7 < 4 else "UNKNOWN")
    last = [o for o in run.outs if o.t > run.outs[-1].t - 1.0]
    assert all(o.smooth_label == "OPEN_PALM" for o in last) and run.outs[-1].reasons == ["WRONG_GESTURE"]


def test_wrong_gesture_half_boundary(cfg_raw):
    # 4 fps: the timeout frame t = 5.25 sees 4.5, 4.75, 5.0, 5.25 in its 1 s window; exactly 2 are OPEN_PALM
    run = Run(S.Session(cfg_raw, FakeScore(), target="V_SIGN"), fps=4)
    run.start()
    while run.s.state == S.CHALLENGE:
        run.step("OPEN_PALM" if run.t in (4.5, 4.75) else None)
    out = run.outs[-1]
    assert out.t == 5.25 and out.reasons == ["WRONG_GESTURE"]


def test_stable_gap_tolerance(cfg_raw):
    run = Run(S.Session(cfg_raw, FakeScore(), target="OPEN_PALM"))
    run.start()
    t_s = run.t
    for g in ["OPEN_PALM"] * 9 + ["FIST"] * 2:  # 0.3 s match, 2 frames (0.067 s) off -> kept
        run.step(g)
    run.until(lambda o: o.phase == S.HOLD, "OPEN_PALM")
    assert t_s + 0.5 - 1e-9 <= run.ev("HOLD_START")[0].t <= t_s + 0.5 + 1 / FPS

    run = Run(S.Session(cfg_raw, FakeScore(), target="OPEN_PALM"))
    run.start()
    for g in ["OPEN_PALM"] * 9 + ["FIST"] * 5:  # 5 frames (0.167 s) off -> cleared
        run.step(g)
    assert run.s._round.stable_start is None
    t_restart = run.t
    run.until(lambda o: o.phase == S.HOLD, "OPEN_PALM")
    assert run.ev("HOLD_START")[0].t >= t_restart + 0.5 - 1e-9


def _hold_round(cfg_raw, hold_frames, fake=None, roi=False):
    fake = fake or FakeScore()
    run = Run(S.Session(cfg_raw, fake, target="OPEN_PALM"))
    run.start()
    run.to_hold("OPEN_PALM")
    for g in hold_frames:
        run.step(g)
    out = run.until(lambda o: o.phase != S.HOLD)
    return run, out, fake


def test_hold_full_length_and_ratio(cfg_raw):
    run, out, fake = _hold_round(cfg_raw, ["OPEN_PALM"] * 7 + ["FIST"] * 3)
    t_hs = run.ev("HOLD_START")[0].t
    hold_outs = [o for o in run.outs if o.phase == S.HOLD]
    assert all(o.t < t_hs + 1.5 for o in hold_outs) and out.phase == S.RESULT and out.t >= t_hs + 1.5
    hold_end = run.ev("HOLD_END")[0]
    assert hold_end.data["hold_match_ratio"] == pytest.approx(0.7) == fake.calls[0].hold_match_ratio
    assert [e.kind for e in out.events] == ["HOLD_END", "RESULT"] and out.verdict == "PASS"

    run, out, fake = _hold_round(cfg_raw, [])  # entry frame and the whole HOLD without a hand
    assert run.ev("HOLD_END")[0].data["hold_match_ratio"] == 0.0 == fake.calls[0].hold_match_ratio


def test_sequence_frames(cfg_raw):
    run, out, fake = _hold_round(cfg_raw, ["OPEN_PALM"] * 10)
    seq = fake.calls[0]
    assert seq.frames[0].t == run.ev("ROUND_START")[0].t and seq.frames[-1].t == seq.t_hold_end == out.t
    assert len(seq.frames) == sum(o.in_round for o in run.outs)
    assert (seq.t0, seq.targets, seq.t_hold_start) == (1.0, ["OPEN_PALM"], run.ev("HOLD_START")[0].t)


def _roi(i):
    return np.full((8, 8), i % 256, np.uint8)


def test_roi_interval(cfg_raw):
    fake = FakeScore()
    run = Run(S.Session(cfg_raw, fake, target="OPEN_PALM"))
    run.s.key("space")
    while run.s.state != S.HOLD:
        run.step("OPEN_PALM", roi=_roi(run.i))
    while run.s.state == S.HOLD:
        run.step("OPEN_PALM", roi=_roi(run.i))
    seq = fake.calls[0]
    assert len(seq.roi_t) > 10 and np.all(np.diff(seq.roi_t) >= 0.1) and seq.roi_t[0] >= seq.t0
    assert seq.roi.shape[1:] == (8, 8) and list(seq.roi[:, 0, 0]) == [round(t * FPS) for t in seq.roi_t]


def test_roi_max_blocks(cfg_raw):
    c = copy.deepcopy(cfg_raw)
    c["session"]["hold_s"] = 12.0  # long enough for 80 blocks
    fake = FakeScore()
    run = Run(S.Session(c, fake, target="OPEN_PALM"), fps=8)  # 0.125 s apart: every block accepted
    run.start("OPEN_PALM")
    sent = []
    while run.s.state != S.RESULT:
        give = len(sent) < 80
        sent += [run.i] if give else []
        run.step("OPEN_PALM", roi=_roi(run.i) if give else None)
    seq = fake.calls[0]
    assert len(seq.roi) == 70 and list(seq.roi[:, 0, 0]) == sent[10:]


def test_external_roi(cfg_raw):
    ts = np.arange(0, 6, 0.1)
    ext = np.stack([_roi(i) for i in range(len(ts))])
    fake = FakeScore()
    run = Run(S.Session(cfg_raw, fake, target="OPEN_PALM", external_roi=(ext, ts)))
    run.start()
    run.to_hold("OPEN_PALM")
    run.until(lambda o: o.phase == S.RESULT, "OPEN_PALM", roi=_roi(255))  # per-frame roi ignored
    seq = fake.calls[0]
    keep = (ts >= seq.t0) & (ts <= seq.t_hold_end)
    assert np.array_equal(seq.roi_t, ts[keep]) and np.array_equal(seq.roi, ext[keep])


def test_result_then_idle(cfg):
    run = Run(S.Session(cfg, FakeScore(), target="V_SIGN"))
    run.start()
    res = run.until(lambda o: o.phase == S.RESULT)
    run.s.key("space")
    assert run.step().phase == S.RESULT and len(run.ev("ROUND_START")) == 1
    out = run.until(lambda o: o.phase == S.IDLE)
    assert out.t >= res.t + 3.0 and run.outs[-2].t < res.t + 3.0 and [e.kind for e in out.events] == ["ROUND_END"]
    assert not out.in_round and out.target is None


def test_r_key(cfg):
    fake = FakeScore()
    run = Run(S.Session(cfg, fake, target="V_SIGN"))
    run.start()
    run.s.key("r")
    out = run.step()
    assert out.phase == S.IDLE and [e.kind for e in out.events] == ["ROUND_DISCARD"] and fake.calls == []

    run.start()
    run.until(lambda o: o.phase == S.RESULT)
    run.s.key("r")
    out = run.step()
    assert out.phase == S.IDLE and [e.kind for e in out.events] == ["ROUND_END"]


def test_set_t0_shifts_timeout(cfg):
    run = Run(S.Session(cfg, FakeScore(), target="V_SIGN"))
    run.start()
    run.s.set_t0(1.2)
    out = run.until(lambda o: o.phase == S.RESULT)
    assert 5.2 < out.t <= 5.2 + 1 / FPS and run.ev("REVEAL")[0].data["t0"] == 1.0
    assert out.reasons == ["TIMEOUT"]


def test_replay(cfg_raw):
    seen = []

    def real(seq):
        seen.append(seq)
        return liveness.score(seq, cfg_raw)

    ts = np.arange(0, 4, 0.1)
    ext = np.stack([_roi(i) for i in range(len(ts))])
    run = Run(S.Session(cfg_raw, real, target="OPEN_PALM"))
    run.start()
    run.to_hold("OPEN_PALM")
    run.until(lambda o: o.phase == S.RESULT, "OPEN_PALM")
    orig = run.ev("RESULT")[0].data
    frames = orig["sequence"].frames

    out, events = S.replay(frames, cfg_raw, real, orig["targets"], orig["t0"], ext, ts)
    rep = [e for e in events if e.kind == "RESULT"][0].data
    assert out.phase == S.RESULT and rep["result"].verdict == orig["result"].verdict
    assert rep["result"].reasons == orig["result"].reasons and rep["hold_match_ratio"] == orig["hold_match_ratio"]
    keep = (ts >= orig["t0"]) & (ts <= orig["t_hold_end"])
    assert np.array_equal(seen[-1].roi, ext[keep]) and np.array_equal(seen[-1].roi_t, ts[keep])


def test_hold_entry_frame_counts(cfg_raw):
    run = Run(S.Session(cfg_raw, FakeScore(), target="OPEN_PALM"))
    run.start()
    run.to_hold("OPEN_PALM", g_on_entry="OPEN_PALM")
    out = run.until(lambda o: o.phase != S.HOLD, "FIST")
    n = sum(o.phase == S.HOLD for o in run.outs)  # non-target frames: HOLD outputs minus the entry, plus the end frame
    assert out.phase == S.RESULT and run.ev("HOLD_END")[0].data["hold_match_ratio"] == pytest.approx(1 / (n + 1))


def test_stable_gap_accumulates_across_breaks(cfg_raw):
    run = Run(S.Session(cfg_raw, FakeScore(), target="OPEN_PALM"))
    run.start()
    for g in ["OPEN_PALM"] * 9 + ["FIST"] * 2 + ["OPEN_PALM"] * 3 + ["FIST"] * 2:  # 4 off frames: 0.133 s > 0.1
        run.step(g)
    assert run.s._round.first_match and run.s._round.stable_start is None
    assert run.s.state == S.CHALLENGE and not run.ev("HOLD_START")


def test_replay_applies_t0(cfg_raw):
    fake = FakeScore()
    run = Run(S.Session(cfg_raw, fake, target="OPEN_PALM"))
    run.s.key("space")
    while run.s.state != S.CHALLENGE:
        run.step("OPEN_PALM", roi=_roi(run.i))
    t_rev = run.outs[-1].t
    run.s.set_t0(t_rev + 0.02)  # the reveal-frame block is now before t0
    while run.s.state != S.RESULT:
        run.step("OPEN_PALM", roi=_roi(run.i))
    live = fake.calls[0]
    assert live.t0 == t_rev + 0.02 and live.roi_t[0] > live.t0

    # stored blocks plus the reveal-frame one: replay must drop it with the given t0
    ext = np.concatenate([_roi(round(t_rev * FPS))[None], live.roi]), np.r_[t_rev, live.roi_t]
    rep = FakeScore()
    S.replay(live.frames, cfg_raw, rep, ["OPEN_PALM"], t_rev + 0.02, *ext)
    got = rep.calls[0]
    assert (got.t0, got.t_hold_start, got.t_hold_end, got.hold_match_ratio) == \
        (live.t0, live.t_hold_start, live.t_hold_end, live.hold_match_ratio)
    assert np.array_equal(got.roi, live.roi) and np.array_equal(got.roi_t, live.roi_t)


def test_stable_hold_boundaries(cfg_raw):
    # 2 fps: every t is exact in binary. first_match_frames 1 so the stable hold alone decides the entry frame.
    c = copy.deepcopy(cfg_raw)
    c["session"]["first_match_frames"] = 1
    run = Run(S.Session(c, FakeScore(), target="OPEN_PALM"), fps=2)
    run.start()  # reveal at 1.0, no hand
    run.step("OPEN_PALM")  # 1.5: stable_start
    out = run.step("OPEN_PALM")  # 2.0 == stable_start + 0.5
    assert out.t == 2.0 and out.phase == S.HOLD and run.ev("HOLD_START")[0].t == 2.0

    run = Run(S.Session(c, FakeScore(), target="OPEN_PALM"), fps=2)
    run.start()
    run.step("OPEN_PALM")  # 1.5: first match
    while run.t < 5.5:
        run.step()  # no hand: stable cleared
    run.step("OPEN_PALM")  # 5.5: stable_start
    out = run.step("OPEN_PALM")  # 6.0 == t0 + 5.0: completes in time
    assert out.t == 6.0 and out.phase == S.HOLD and not run.ev("RESULT")
