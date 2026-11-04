import re
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import hud  # noqa: E402
from config import load_config  # noqa: E402
from gesture import analyze  # noqa: E402
from gesture_templates import template  # noqa: E402
from liveness import REASON_TEXT  # noqa: E402  (session.py takes it from here)
from liveness.frame import empty_frame  # noqa: E402

CFG = {
    "gesture": load_config(str(ROOT / "config.yaml"))["gesture"],
    "session": {"countdown_s": 1.0, "first_match_max_s": 4.0, "challenge_timeout_s": 5.0, "stable_hold_s": 0.5,
                "hold_s": 1.5, "result_s": 3.0},
    "quality": {"min_palm_frac": 0.08},
    "fusion": {"w_planarity": 0.65, "w_texture": 0.20, "w_nonrigid": 0.15, "planarity_floor": 0.4,
               "texture_floor": 0.3, "pass_total": 0.6},
    "planarity": {"min_normal_deg": 12.0},
}
STATS = {"fps": 30.0, "proc_ms": 8.3, "recording": False}
SIZES = ((640, 480), (1280, 720), (1920, 1080))


@dataclass
class Out:  # SessionOutput (SPEC_M2 0.3) stub
    phase: str = "IDLE"
    t: float = 10.0
    t_in_phase: float = 0.4
    phase_len: float | None = None
    prompt: str = ""
    target: str | None = None
    targets: list = field(default_factory=list)
    stable_progress: float = 0.0
    first_match: bool = False
    hint: str | None = None
    verdict: str | None = None
    reasons: list = field(default_factory=list)
    reason_text: str | None = None
    scores: dict = field(default_factory=dict)
    total: float | None = None
    raw_label: str = "NONE"
    smooth_label: str = "NONE"
    e: np.ndarray | None = None
    palm_ratio: float | None = None
    quality_ok: bool | None = None
    in_round: bool = False
    events: list = field(default_factory=list)


def roty(p, deg):
    """+deg turns the camera-facing side (-z) of the hand toward screen-right (+x)."""
    a = np.radians(deg)
    r = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
    return p @ r.T


def frame(W, H, gid="THUMBS_UP", yaw=0.0, nan=False):
    w = roty(template(gid), yaw)
    px = np.c_[w[:, 0] * 2.2 * H + W / 2, w[:, 1] * 2.2 * H + H * 0.55, w[:, 2] * 2.2 * H]
    fr = replace(empty_frame(10.0, (W, H)), n_hands=1, present=True, lm_px=px.astype(np.float32),
                 lm_world=w.astype(np.float32), hand_score=0.95, raw_label=gid, roi_mean=120.0, roi_sat_ratio=0.01)
    if nan:
        fr.lm_px[:] = np.nan
        fr.lm_world[:] = np.nan
    return fr


def result_event(debug):
    res = SimpleNamespace(verdict="PASS", total=0.8, scores={}, reasons=[], debug=debug)
    return SimpleNamespace(kind="RESULT", t=10.0, data={"result": res})


SCORES = {"temporal": 1.0, "planarity": 0.82, "texture": 0.71, "nonrigid": 0.64}
LIVE = [Out(), Out(phase="COUNTDOWN", phase_len=1.0, prompt="Show one hand"),
        Out(phase="CHALLENGE", phase_len=5.0, target="THUMBS_UP", raw_label="FIST", smooth_label="FIST",
            hint="Move closer", palm_ratio=0.05, quality_ok=False),
        Out(phase="CHALLENGE", phase_len=5.0, target="THUMBS_UP", first_match=True, stable_progress=0.6),
        Out(phase="HOLD", phase_len=1.5, target="THUMBS_UP", stable_progress=1.0, e=np.full(5, np.nan))]
VERDICTS = [("PASS", [], SCORES, 0.77), ("FAIL", ["PLANAR"], {**SCORES, "planarity": 0.18}, 0.29),
            ("INSUFFICIENT", ["NEED_ROTATION"], {**SCORES, "planarity": None}, None),
            ("FAIL", ["TIMEOUT"], {}, None), ("FAIL", ["WRONG_GESTURE"], {}, None),
            ("FAIL", ["INSTANT_SWITCH"], {**SCORES, "temporal": 0.0}, None)]
RESULTS = [Out(phase="RESULT", phase_len=3.0, target="V_SIGN", verdict=v, reasons=r, reason_text=r[0] if r else None,
               scores=s, total=t, events=[result_event({"t_r": 0.42, "n": np.nan, "angles": [21.5, 18.2, 16.9, 3.0],
                                                         "placeholder_scores": True})])
           for v, r, s, t in VERDICTS]


@pytest.mark.parametrize("W,H", SIZES)
def test_every_phase_verdict_debug_renders(W, H):
    img = np.random.default_rng(0).integers(0, 256, (H, W, 3), dtype=np.uint8)
    src = img.copy()
    for out in LIVE + RESULTS:
        for debug in (False, True):
            for fr in (frame(W, H), None, frame(W, H, nan=True), empty_frame(10.0, (W, H))):
                pose = None if fr is None or not fr.present else analyze(fr.lm_world, fr.lm_px, CFG["gesture"])
                res = hud.draw(img, fr, pose, out, STATS, debug, CFG, hud.HudState())
                assert res.shape == img.shape and res.dtype == np.uint8
                assert np.array_equal(img, src)
    for phase in ("CHALLENGE", "HOLD", "RESULT"):  # every Optional field None, NaN numbers (t is always a frame time)
        none_out = Out(phase=phase, phase_len=None, target=None, verdict=None, scores={"planarity": None},
                       total=float("nan"), e=None, palm_ratio=float("nan"), quality_ok=None, reason_text=None)
        hud.draw(img, None, None, none_out, {"fps": None, "proc_ms": float("nan"), "recording": False}, True, CFG,
                 hud.HudState())


def test_deterministic():
    img = np.full((720, 1280, 3), 90, np.uint8)
    fr = frame(1280, 720, yaw=5.0)
    pose = analyze(fr.lm_world, fr.lm_px, CFG["gesture"])
    for out in (LIVE[4], RESULTS[0]):
        a = hud.draw(img, fr, pose, out, STATS, True, CFG, hud.HudState())
        b = hud.draw(img, fr, pose, out, STATS, True, CFG, hud.HudState())
        assert np.array_equal(a, b)


def test_palm_yaw_sign():
    for gid in ("OPEN_PALM", "THUMBS_UP", "FIST"):
        for p in (template(gid), template(gid) * [-1, 1, 1], roty(template(gid), 180)):  # right, mirrored, back
            d = hud.palm_yaw_deg(roty(p, 10)) - hud.palm_yaw_deg(p)
            assert d == pytest.approx(10, abs=1)
    nan = template("FIST")
    nan[5] = np.nan
    assert hud.palm_yaw_deg(nan) is None
    assert hud.palm_yaw_deg(np.zeros((21, 3))) is None


def test_tilt_state_machine():
    st, W, H = hud.HudState(), 1280, 720
    ui = lambda phase, yaw, nan=False: hud.build_ui(Out(phase=phase), frame(W, H, yaw=yaw, nan=nan), None, STATS,
                                                     CFG, False, st)
    assert ui("CHALLENGE", 5)["tilt_deg"] is None
    u = ui("HOLD", 5)                      # baseline = first HOLD frame
    assert (u["tilt_deg"], u["tilt_min_deg"], u["tilt_max_deg"]) == pytest.approx((0, 0, 0), abs=1e-6)
    u = ui("HOLD", 15)
    assert (u["tilt_deg"], u["tilt_min_deg"], u["tilt_max_deg"]) == pytest.approx((10, 0, 10), abs=1e-6)
    u = ui("HOLD", -3)
    assert (u["tilt_deg"], u["tilt_min_deg"], u["tilt_max_deg"]) == pytest.approx((-8, -8, 10), abs=1e-6)
    assert u["tilt_range_deg"] == pytest.approx(18)
    u = ui("HOLD", 0, nan=True)            # lost frame: no live value, range kept
    assert u["tilt_deg"] is None and u["tilt_min_deg"] == pytest.approx(-8)
    assert u["tilt_target_deg"] == pytest.approx(18.0)
    u = ui("RESULT", 0)
    assert u["tilt_deg"] is None and u["tilt_min_deg"] is None and u["tilt_range_deg"] is None and st.yaw0 is None
    u = ui("HOLD", 20)                     # next round: new baseline
    assert u["tilt_deg"] == pytest.approx(0, abs=1e-6) and u["tilt_max_deg"] == pytest.approx(0, abs=1e-6)


def test_result_debug_values_and_keys(monkeypatch):
    st, out = hud.HudState(), RESULTS[0]
    dbg = out.events[0].data["result"].debug
    assert hud.build_ui(out, None, None, STATS, CFG, True, st)["debug_values"] is dbg
    assert hud.build_ui(replace(out, events=[]), None, None, STATS, CFG, True, st)["debug_values"] is dbg
    assert hud.build_ui(Out(), None, None, STATS, CFG, False, st)["debug_values"] is None and st.result_debug is None
    live = hud.build_ui(Out(phase="CHALLENGE"), None, None, STATS, CFG, True, st)["debug_values"]
    g = CFG["gesture"]                     # labels from config.yaml, formatted like hud.build_ui
    assert f"pip2d >{g['pip_extended_deg']:g}" in live and f"thumb tip >{g['thumb_tip_ratio']:g}" in live
    assert f"tip gap <{g['ok_tip_ratio']:g}" in live
    assert set(hud.build_ui(Out(), None, None, STATS, CFG, False, st)) == set(_reference_keys())
    texts, orig = [], cv2.putText
    monkeypatch.setattr(cv2, "putText", lambda img, s, *a, **k: (texts.append(s), orig(img, s, *a, **k))[1])
    img = np.zeros((720, 1280, 3), np.uint8)
    hud.draw(img, None, None, out, {**STATS, "recording": True}, False, CFG, hud.HudState())
    assert "back to start" in texts and "discard sample" in texts and "Back to start in" in texts
    assert not {"new round", "new", "SPACE"} & set(texts)
    texts.clear()
    hud.draw(img, None, None, LIVE[1], STATS, False, CFG, hud.HudState())   # FU8: SPACE only in IDLE
    assert "SPACE" not in texts and "reset" in texts
    texts.clear()
    hud.draw(img, None, None, Out(), STATS, False, CFG, hud.HudState())
    assert "SPACE" in texts and "Press SPACE to start" in texts


def _spy_texts(monkeypatch):
    texts, orig = [], cv2.putText
    monkeypatch.setattr(cv2, "putText", lambda img, s, *a, **k: (texts.append(s), orig(img, s, *a, **k))[1])
    return texts


@pytest.mark.parametrize("W,H", SIZES)
def test_result_reason_text_drawn_once(monkeypatch, W, H):
    """PROMPT 3.1: RESULT line 2 is the reason-code text, drawn once (the footer shows the verdict line, not the
    prompt session.py also sets to it); IDLE draws its prompt. Ported from c726dd7 test_ui_result_has_no_top_left_prompt."""
    texts, img = _spy_texts(monkeypatch), np.zeros((H, W, 3), np.uint8)
    for code, sc in (("TIMEOUT", {}), ("PLANAR", {**SCORES, "planarity": 0.18})):
        reason = REASON_TEXT[code]
        out = Out(phase="RESULT", phase_len=3.0, target="FIST", verdict="FAIL", reasons=[code], reason_text=reason,
                  prompt=reason, scores=sc, total=0.29 if sc else None)
        texts.clear()
        hud.draw(img, empty_frame(10.0, (W, H)), None, out, STATS, False, CFG, hud.HudState())
        runs = [(i, j) for i in range(len(texts)) for j in range(i + 1, len(texts) + 1)
                if " ".join(texts[i:j]) == reason]   # the whole string, or wrapped lines that join back to it
        assert len(runs) == 1, (code, texts)
    texts.clear()
    hud.draw(img, empty_frame(10.0, (W, H)), None, Out(prompt="Press SPACE to start now"), STATS, False, CFG,
             hud.HudState())
    assert "Press SPACE to start now" in texts   # out.prompt reaches the IDLE footer (not the built-in fallback)


def test_challenge_countdown_one_number(monkeypatch):
    """FU2: one number to challenge_s for the whole CHALLENGE, unchanged by the first_match flip; 'MATCH IN' counts
    to first_match_s; both read 0.0 exactly at their deadline (never earlier)."""
    texts, img = _spy_texts(monkeypatch), np.zeros((720, 1280, 3), np.uint8)

    def shown(tp, matched):
        texts.clear()
        lab = "THUMBS_UP" if matched else "NONE"
        out = Out(phase="CHALLENGE", t=10 + tp, t_in_phase=tp, target="THUMBS_UP", first_match=matched,
                  raw_label=lab, smooth_label=lab)
        hud.draw(img, None, None, out, STATS, False, CFG, hud.HudState())
        nums = [s for s in texts if re.fullmatch(r"\d+\.\d", s)]
        assert len(nums) == 1
        return float(nums[0]), [s for s in texts if s.startswith("MATCH IN") or s == "MATCHED"]

    seq = [shown(i / 30, i >= 60)[0] for i in range(151)]           # first match at t0 + 2.0 s
    assert seq[0] == 5.0 and seq[-1] == 0.0
    assert all(0 <= a - b <= 0.1 + 1e-9 for a, b in zip(seq, seq[1:]))
    for tp in (1.0, 2.0, 3.97):
        assert shown(tp, False)[0] == shown(tp, True)[0]
    assert shown(2.0, True)[1] == ["MATCHED"] and shown(2.0, False)[1] == ["MATCH IN 2.0 s"]
    assert shown(3.98, False)[1] == ["MATCH IN 0.1 s"] and shown(4.0, False)[1] == ["MATCH IN 0.0 s"]
    assert shown(4.98, True)[0] == 0.1 and shown(5.0, True)[0] == 0.0


def _hold_ui(lo, hi, d):
    out = Out(phase="HOLD", t=10.0, t_in_phase=0.5, target="THUMBS_UP", stable_progress=1.0)
    ui = hud.build_ui(out, None, None, STATS, CFG, False, hud.HudState())
    return dict(ui, tilt_deg=d, tilt_min_deg=lo, tilt_max_deg=hi, tilt_range_deg=hi - lo)


def test_hold_both_sides_while_short(monkeypatch):
    """RC1: no per-frame side choice - still-hand jitter never changes the card words or swaps an arrowhead's side;
    both arrowheads stay until the target is reached, then KEEP HOLDING and none."""
    words, tips, track, fill = [], [], hud._track, cv2.fillPoly
    monkeypatch.setattr(hud, "_track", lambda c, s, *a, **k: (words.append(s), track(c, s, *a, **k))[1])
    monkeypatch.setattr(cv2, "fillPoly", lambda img, pts, col, *a, **k: (
        tips.append(np.asarray(pts[0]).reshape(-1, 2)[0] / 16) if tuple(col) == hud.AMBER else None,
        fill(img, pts, col, *a, **k))[1])
    img, fr = np.full((720, 1280, 3), 60, np.uint8), frame(1280, 720)
    hand = {"present": True, "n_hands": 1, "hand_score": 0.95, "lm_px": fr.lm_px, "lm_world": fr.lm_world}
    rng, lo, hi, prev, seen = np.random.default_rng(3), 0.0, 0.0, None, set()
    for i in range(45):                    # still hand: tilt is jitter around 0, the running range grows a little
        d = float(rng.normal(0, 0.6)) if i else 0.0
        lo, hi = min(lo, d), max(hi, d)
        words.clear(), tips.clear()
        hud.render(img, hand, _hold_ui(lo, hi, d))
        assert len(tips) == 2
        cur = sorted(tips, key=lambda p: p[0])
        if prev is not None:
            assert max(float(np.hypot(*(p - q))) for p, q in zip(cur, prev)) < 20
        prev, seen = cur, seen | {w for w in words if w.startswith(("TURN", "KEEP"))}
    assert seen == {"TURN LEFT / RIGHT"}
    words.clear(), tips.clear()
    hud.render(img, hand, _hold_ui(-9.0, 9.0, 0.0))   # 18 deg = the adapter's target: reached
    assert not tips and "KEEP HOLDING" in words


REAL_DEBUG = {"t_r": 0.4183127, "F": 0.8611904, "gap_max_s": 0.06693, "t_switch": [], "R": 0.09436018, "n_pairs": 10,
              "normal_angles": [24.71938, 22.03517, 19.86624, 18.40215, 16.93378, 15.27741, 14.61902, 13.80357,
                                12.95184, 12.20431], "rho": None, "M": float("nan"), "hold_match_ratio": 0.9512195}


@pytest.mark.parametrize("W,H", [(640, 480), (1280, 720)])
def test_debug_values_fit_their_plate(monkeypatch, W, H):
    """F2: PROMPT-realistic debug values: no '...', every string inside the debug-values plate, lists as
    count + range."""
    drawn, plates, put, plate, dvp = [], [], hud.put, hud._plate, hud._debug_values_plate
    monkeypatch.setattr(hud, "put", lambda img, s, org, col, kind="sans", px=20, w=400, *a: (
        drawn.append((s, org[0] + hud.text_rect(s, kind, px, w)[0], org[1] + hud.text_rect(s, kind, px, w)[1],
                      *hud.text_rect(s, kind, px, w)[2:])), put(img, s, org, col, kind, px, w, *a))[1])
    monkeypatch.setattr(hud, "_plate", lambda c, x0, y0, x1, y1, accent=hud.CYAN: (
        plates.append((x0, y0, x1, y1)), plate(c, x0, y0, x1, y1, accent))[1])
    inside = []

    def wrapped(c, y0, *a):
        n0, p0 = len(drawn), len(plates)
        dvp(c, y0, *a)
        inside.extend((s, plates[p0]) for s in drawn[n0:])
    monkeypatch.setattr(hud, "_debug_values_plate", wrapped)
    out = replace(RESULTS[0], events=[result_event(REAL_DEBUG)])
    hud.draw(np.zeros((H, W, 3), np.uint8), None, None, out, STATS, True, CFG, hud.HudState())
    assert inside and not [d for d in drawn if "..." in d[0]]
    for (s, x, y, w, h), (x0, y0, x1, y1) in inside:
        assert x0 <= x and x + w <= x1 and y0 <= y and y + h <= y1, (s, (x, y, w, h), (x0, y0, x1, y1))
    texts = {d[0][0] for d in inside}
    assert "10: 12.2..24.72" in texts and "--" in texts and "0.4183" in texts


def test_dropout_keeps_hand_layer():
    """RC4: one dropped detection frame (absent / NaN) renders like the present frame; after HAND_HOLD_S it is gone."""
    img, fr = np.full((720, 1280, 3), 90, np.uint8), frame(1280, 720)
    out = Out(phase="CHALLENGE", t=10.0, target="THUMBS_UP", first_match=True, stable_progress=0.5)
    st = hud.HudState()
    hud.draw(img, fr, None, out, STATS, False, CFG, st)
    for dt, lost in ((1 / 30, empty_frame(10.0, (1280, 720))), (2 / 30, None)):
        o = replace(out, t=10.0 + dt)
        assert np.array_equal(hud.draw(img, lost, None, o, STATS, False, CFG, st),
                              hud.draw(img, fr, None, o, STATS, False, CFG, hud.HudState()))
    o = replace(out, t=10.0 + hud.HAND_HOLD_S + 0.05)
    assert np.array_equal(hud.draw(img, None, None, o, STATS, False, CFG, st),
                          hud.draw(img, None, None, o, STATS, False, CFG, hud.HudState()))


def test_lost_match_after_first_match(monkeypatch):
    """U1: first_match stays set in M2; once neither the raw nor the smoothed label is the target the CHALLENGE
    screen says so (no MATCHED / Match found, SEEING chip, amber MATCH LOST); a raw-only or smooth-only flicker keeps
    the matched screen."""
    texts, img = _spy_texts(monkeypatch), np.full((720, 1280, 3), 80, np.uint8)
    fr = frame(1280, 720, gid="FIST")

    def shown(raw, smooth):
        texts.clear()
        out = Out(phase="CHALLENGE", t=14.8, t_in_phase=4.8, target="THUMBS_UP", first_match=True,
                  stable_progress=0.0 if smooth != "THUMBS_UP" else 0.4, raw_label=raw, smooth_label=smooth)
        hud.draw(img, fr, None, out, STATS, False, CFG, hud.HudState())
        return list(texts)

    lost = shown("FIST", "FIST")
    assert not [s for s in lost if "MATCHED" in s or s.startswith(("Match found", "MATCH FOUND"))]
    assert "Lost it - show THUMBS_UP and hold steady" in lost and "SEEING FIST" in lost and "MATCH LOST" in lost
    assert "0.2" in lost                   # the number keeps counting
    for raw, smooth in (("THUMBS_UP", "FIST"), ("FIST", "THUMBS_UP")):
        kept = shown(raw, smooth)
        assert "MATCHED" in kept and "Match found. Hold steady" in kept and "MATCH LOST" not in kept


def _base_yaw(gid="THUMBS_UP"):
    return hud.palm_yaw_deg(template(gid))


def test_tilt_edge_on_baseline(monkeypatch):
    """YAW-FOLD: the folded yaw jumps 180 deg edge-on; the tilt is wrapped mod 180, so a 4.5 deg turn from an
    edge-on baseline stays a 4.5 deg turn (card still TURN), and a +-30 deg sweep around yaw0 = +88 is exact."""
    W, H = 1280, 720
    for yaw0, turn in ((-86.0, -4.5), (-88.0, 4.5)):
        st, psi = hud.HudState(), yaw0 - _base_yaw()
        hold = lambda d: hud.build_ui(Out(phase="HOLD", t_in_phase=0.5, target="THUMBS_UP", stable_progress=1.0),
                                      frame(W, H, yaw=psi + d), None, STATS, CFG, False, st)
        assert hold(0)["tilt_deg"] == pytest.approx(0, abs=1e-6)
        u = hold(turn)
        assert u["tilt_deg"] == pytest.approx(turn, abs=0.5) and u["tilt_range_deg"] < 5
    words, track = [], hud._track
    monkeypatch.setattr(hud, "_track", lambda c, s, *a, **k: (words.append(s), track(c, s, *a, **k))[1])
    fr = frame(W, H, yaw=psi + turn)
    hud.render(np.full((H, W, 3), 60, np.uint8), {"present": True, "n_hands": 1, "lm_px": fr.lm_px,
                                                  "lm_world": fr.lm_world}, u)
    assert "TURN LEFT / RIGHT" in words and "KEEP HOLDING" not in words
    st, psi, prev = hud.HudState(), 88.0 - _base_yaw(), None
    for d in list(np.arange(0.0, -30.5, -0.5)) + list(np.arange(-30.0, 30.5, 0.5)):
        u = hud.build_ui(Out(phase="HOLD"), frame(W, H, yaw=psi + d), None, STATS, CFG, False, st)
        assert abs(u["tilt_deg"] - d) < 1
        if prev is not None:
            assert abs(u["tilt_deg"] - prev) < 2
        prev = u["tilt_deg"]


def _hint_run(hints, frames=None, phase="CHALLENGE"):
    """Shown hint per frame at 30 fps for a list of session hints (frames: None = no hand that frame)."""
    st, W, H = hud.HudState(), 1280, 720
    frames = frames or [frame(W, H)] * len(hints)
    return [hud.build_ui(Out(phase=phase, t=10.0 + i / 30, target="THUMBS_UP", hint=h), f, None, STATS, CFG, False,
                         st)["hint"] for i, (h, f) in enumerate(zip(hints, frames))]


def test_hint_debounce(monkeypatch):
    """HINT-FLICKER: the per-frame session hint is debounced for display: a one-frame hint never shows, a hint
    toggling every 2 frames gives one steady banner, a dropped frame inside a hint keeps it; it clears HINT_OFF_S
    after the session stops sending it; IDLE resets it."""
    mc = "Move closer"
    assert set(_hint_run([None] * 10 + [mc] + [None] * 30)) == {None}
    assert set(_hint_run((["Show only one hand"] + [None] * 8) * 6)) == {None}      # isolated blips 0.3 s apart
    tog = _hint_run(([mc] * 2 + [None] * 2) * 20)
    assert tog[:8] == [None] * 8 and set(tog[8:]) == {mc}         # on at the first hint frame >= 0.2 s, then steady
    steady = [mc] * 20 + [None] + [mc] * 10
    shown = _hint_run(steady, [frame(1280, 720)] * 20 + [None] + [frame(1280, 720)] * 10)
    assert set(shown[7:]) == {mc}
    off = _hint_run([mc] * 20 + [None] * 30)
    assert set(off[7:37]) == {mc} and set(off[39:]) == {None}     # last seen at frame 19, held 0.6 s (18 frames)
    assert _hint_run([mc] * 20, phase="IDLE") == [None] * 20
    texts, img, st = _spy_texts(monkeypatch), np.zeros((720, 1280, 3), np.uint8), hud.HudState()
    for i, h in enumerate(([mc] * 2 + [None] * 2) * 5):   # the strip reads the filtered value
        texts.clear()
        hud.draw(img, frame(1280, 720), None, Out(phase="CHALLENGE", t=10 + i / 30, target="THUMBS_UP", hint=h,
                                                  palm_ratio=0.081, quality_ok=h is None), STATS, False, CFG, st)
    assert mc in texts and hud.BORDERLINE in texts


def _reference_keys():
    return ("phase t t_in_phase phase_len prompt target stable_progress first_match hint verdict reasons reason_text "
            "scores total raw_label smooth_label e palm_ratio quality_ok fps proc_ms debug recording tilt_deg "
            "tilt_range_deg tilt_target_deg debug_values tilt_min_deg tilt_max_deg protocol fusion "
            "palm_min_ratio").split()


@pytest.mark.parametrize("out", [LIVE[4], RESULTS[1]], ids=["HOLD", "RESULT"])
def test_render_p95_720p(out):
    img = np.random.default_rng(1).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    fr, st = frame(1280, 720, yaw=7.0), hud.HudState()
    hand = {"present": True, "n_hands": 1, "hand_score": 0.95, "lm_px": fr.lm_px, "lm_world": fr.lm_world}
    ui = hud.build_ui(out, fr, None, STATS, CFG, False, st)
    ts = []
    for i in range(210):
        ui["t"], ui["t_in_phase"] = 10.0 + i / 30, (i % 45) / 30
        t0 = time.perf_counter()
        hud.render(img, hand, ui)
        ts.append((time.perf_counter() - t0) * 1000)
    p50, p95 = np.percentile(ts[10:], [50, 95])
    print(f"render {out.phase} 720p p50 {p50:.2f} ms p95 {p95:.2f} ms")
    assert p95 < 8.0
