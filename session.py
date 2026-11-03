"""Challenge state machine (PROMPT.md 3). Pure logic: Frame + keys in, phase / prompt / events out.

Every timer runs on Frame.t, so --video --headless replays a round deterministically. No camera, no UI.
"""
from __future__ import annotations

import random
from collections import Counter, deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

import gesture
from liveness import REASON_TEXT, LivenessResult, quality
from liveness.frame import GESTURES, Frame, Sequence, palm_width_px

IDLE, COUNTDOWN, CHALLENGE, HOLD, VERIFYING, RESULT = "IDLE", "COUNTDOWN", "CHALLENGE", "HOLD", "VERIFYING", "RESULT"
PROMPTS = {IDLE: "Press SPACE to start", COUNTDOWN: "Show one hand", CHALLENGE: "Make the gesture shown above",
           HOLD: "Hold it. Slowly tilt your hand left and right"}
PASS_TEXT = "Liveness verified"
_IN_ROUND = (COUNTDOWN, CHALLENGE, HOLD)


@dataclass
class SessionEvent:
    kind: str      # ROUND_START | REVEAL | HOLD_START | HOLD_END | RESULT | ROUND_DISCARD | ROUND_END
    t: float
    data: dict     # REVEAL: {"t0", "targets"}; HOLD_START: {"t"}; HOLD_END: {"t", "hold_match_ratio"};
                   # RESULT: {"result", "sequence", "targets", "t0", "t_hold_start", "t_hold_end", "hold_match_ratio"}


@dataclass
class SessionOutput:            # one per frame; keys mirror the HUD ui contract
    phase: str
    t: float
    t_in_phase: float
    phase_len: float | None
    prompt: str
    target: str | None
    targets: list[str]
    stable_progress: float
    first_match: bool
    hint: str | None
    verdict: str | None
    reasons: list[str]
    reason_text: str | None
    scores: dict
    total: float | None
    raw_label: str
    smooth_label: str
    e: np.ndarray | None
    palm_ratio: float | None
    quality_ok: bool | None
    in_round: bool
    events: list[SessionEvent]


def _excluded(labels, last_e, tol: float) -> set[str]:
    """Gestures the user already (nearly) holds: known labels seen in the lookback, plus every gesture
    whose judged fingers all sit within tol of the last detected extension e."""
    out = {lb for lb in labels if lb in GESTURES}
    if last_e is not None:
        out |= {g for g, te in gesture.TARGET_E.items()
                if all(abs(e - x) < tol for e, x in zip(last_e, te) if x is not None)}
    return out


class _Round:
    """Everything one round owns; dropped on IDLE."""

    def __init__(self, t: float, roi_max_blocks: int):
        self.round_start = t
        self.frames: list[Frame] = []
        self.smooth: list[str] = []     # smoothed label per round frame
        self.last_e: np.ndarray | None = None
        self.t0: float | None = None
        self.targets: list[str] = []
        self.run, self.run_t = 0, None  # consecutive raw == target detected frames, first t of that run
        self.first_match, self.t_match = False, None
        self.stable_start: float | None = None
        self.gap = 0.0
        self.t_hold_start: float | None = None
        self.t_hold_end: float | None = None
        self.hold_present = self.hold_hit = 0
        self.hold_match_ratio: float | None = None
        self.roi: deque[tuple[float, np.ndarray]] = deque(maxlen=roi_max_blocks)
        self.result: LivenessResult | None = None
        self.t_result: float | None = None


class Session:
    def __init__(self, cfg: dict, score_fn: Callable[[Sequence], LivenessResult], target: str | None = None,
                 seed: int | None = None, external_roi: tuple[np.ndarray, np.ndarray] | None = None):
        if target is not None and target not in GESTURES:
            raise ValueError(f"unknown target {target!r}")
        self._s, self._g, self._q = cfg["session"], cfg["gesture"], cfg["quality"]
        self._score_fn = score_fn
        self._fixed = target
        self._rng = random.Random(seed)  # one generator per session: a fixed seed fixes the whole target sequence
        self._ext = external_roi
        self._smoother = gesture.LabelSmoother(self._g["smooth_window_s"])
        self._state = IDLE
        self._round: _Round | None = None
        self._idle_t: float | None = None
        self._keys: list[str] = []

    @property
    def state(self) -> str:
        return self._state

    def key(self, name: str) -> None:
        if name not in ("space", "r"):
            raise ValueError(f"unknown key {name!r}")
        self._keys.append(name)

    def set_t0(self, t: float) -> None:
        if self._state == CHALLENGE:
            self._round.t0 = float(t)

    # ------------------------------------------------------------------ per frame

    def update(self, fr: Frame, roi: np.ndarray | None = None) -> SessionOutput:
        events: list[SessionEvent] = []
        if self._idle_t is None:
            self._idle_t = fr.t
        for k in self._keys:
            self._apply_key(k, fr, events)
        self._keys.clear()

        smooth = self._smoother.update(fr.t, fr.raw_label)
        e = gesture.analyze(fr.lm_world, fr.lm_px, self._g).e if fr.present else None
        in_round = self._state in _IN_ROUND
        r = self._round
        if in_round:
            r.frames.append(fr)
            r.smooth.append(smooth)
            if e is not None:
                r.last_e = e
        if self._state == COUNTDOWN:
            if fr.t >= r.round_start + self._s["countdown_s"]:
                self._reveal(fr, events)
                self._challenge(fr, smooth, roi, events)
        elif self._state == CHALLENGE:
            self._challenge(fr, smooth, roi, events)
        elif self._state == HOLD:
            self._hold(fr, roi, events)
        elif self._state == RESULT:
            if fr.t >= r.t_result + self._s["result_s"]:
                events.append(SessionEvent("ROUND_END", fr.t, {}))
                self._to_idle(fr)
        return self._output(fr, smooth, e, in_round, events)

    def _apply_key(self, k: str, fr: Frame, events: list) -> None:
        if k == "space" and self._state == IDLE:
            self._round = _Round(fr.t, self._s["roi_max_blocks"])
            self._smoother.reset()
            self._state = COUNTDOWN
            events.append(SessionEvent("ROUND_START", fr.t, {}))
        elif k == "r" and self._state in _IN_ROUND:
            events.append(SessionEvent("ROUND_DISCARD", fr.t, {}))
            self._to_idle(fr)
        elif k == "r" and self._state == RESULT:
            events.append(SessionEvent("ROUND_END", fr.t, {}))
            self._to_idle(fr)

    def _to_idle(self, fr: Frame) -> None:
        self._state, self._round, self._idle_t = IDLE, None, fr.t

    def _reveal(self, fr: Frame, events: list) -> None:
        r, s = self._round, self._s
        if self._fixed is not None:
            target = self._fixed
        else:
            labels = [lb for f, sm in zip(r.frames, r.smooth) if f.t > fr.t - s["reveal_lookback_s"]
                      for lb in (f.raw_label, sm)]
            cand = sorted(set(GESTURES) - _excluded(labels, r.last_e, s["reveal_e_tol"])) or sorted(GESTURES)
            target = self._rng.choice(cand)
        r.targets, r.t0 = [target], fr.t
        self._state = CHALLENGE
        events.append(SessionEvent("REVEAL", fr.t, {"t0": r.t0, "targets": list(r.targets)}))

    def _take_roi(self, fr: Frame, roi) -> None:
        r = self._round
        if roi is None or fr.t < r.t0:
            return
        if r.roi and fr.t - r.roi[-1][0] < self._s["roi_interval_s"]:
            return
        r.roi.append((fr.t, np.array(roi, np.uint8)))  # deque maxlen drops the oldest

    def _challenge(self, fr: Frame, smooth: str, roi, events: list) -> None:
        r, s = self._round, self._s
        target = r.targets[0]
        self._take_roi(fr, roi)
        if fr.present and fr.raw_label == target:
            if r.run == 0:
                r.run_t = fr.t
            r.run += 1
            if r.run >= s["first_match_frames"] and not r.first_match:
                r.first_match, r.t_match = True, r.run_t
        else:
            r.run = 0
        if smooth == target:
            if r.stable_start is None:
                r.stable_start, r.gap = fr.t, 0.0
        elif r.stable_start is not None:
            r.gap += fr.t - r.frames[-2].t
            if r.gap > s["stable_gap_tol_s"]:
                r.stable_start = None

        # 5.1: t_match is the first frame of the run, so a run that started by t0 + first_match_max_s may finish later
        deadline = r.t0 + s["first_match_max_s"]
        no_first_match = (not r.first_match and fr.t > deadline and (r.run == 0 or r.run_t > deadline))
        if (r.first_match and r.stable_start is not None and fr.t - r.stable_start >= s["stable_hold_s"]
                and fr.t <= r.t0 + s["challenge_timeout_s"]):
            r.t_hold_start = fr.t
            self._state = HOLD
            events.append(SessionEvent("HOLD_START", fr.t, {"t": fr.t}))
            self._hold_count(fr)
        elif no_first_match or fr.t > r.t0 + s["challenge_timeout_s"]:
            self._timeout(fr, "WRONG_GESTURE" if self._wrong_gesture(fr) else "TIMEOUT", events)

    def _wrong_gesture(self, fr: Frame) -> bool:
        r, s = self._round, self._s
        win = [sm for f, sm in zip(r.frames, r.smooth) if f.t > fr.t - s["wrong_gesture_window_s"]]
        counts = Counter(sm for sm in win if sm in GESTURES and sm != r.targets[0])
        return bool(counts) and counts.most_common(1)[0][1] >= s["wrong_gesture_frac"] * len(win)

    def _timeout(self, fr: Frame, reason: str, events: list) -> None:
        self._enter_result(fr, LivenessResult("FAIL", None, {}, [reason], {}), None, events)

    def _hold_count(self, fr: Frame) -> None:
        r = self._round
        if fr.present:
            r.hold_present += 1
            r.hold_hit += fr.raw_label == r.targets[0]

    def _hold(self, fr: Frame, roi, events: list) -> None:
        r = self._round
        self._take_roi(fr, roi)
        self._hold_count(fr)
        if fr.t < r.t_hold_start + self._s["hold_s"]:
            return
        r.t_hold_end = fr.t
        r.hold_match_ratio = r.hold_hit / r.hold_present if r.hold_present else 0.0
        events.append(SessionEvent("HOLD_END", fr.t, {"t": fr.t, "hold_match_ratio": r.hold_match_ratio}))
        # VERIFYING: synchronous, never visible as a phase
        roi_arr, roi_t = self._roi_arrays()
        seq = Sequence(list(r.frames), roi_arr, roi_t, list(r.targets), r.t0, r.t_hold_start, r.t_hold_end,
                       r.hold_match_ratio, r.frames[0].frame_wh)
        self._enter_result(fr, self._score_fn(seq), seq, events)

    def _roi_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """ROI blocks (internal buffer, or external_roi) kept to [t0, t_hold_end]; (0, 0, 0) when none."""
        r = self._round
        empty = np.zeros((0, 0, 0), np.uint8), np.zeros(0, np.float64)
        if self._ext is not None:
            blocks, ts = np.asarray(self._ext[0]), np.asarray(self._ext[1], np.float64)
        elif r.roi:
            blocks, ts = np.stack([b for _, b in r.roi]), np.array([t for t, _ in r.roi], np.float64)
        else:
            return empty
        keep = (ts >= r.t0) & (ts <= r.t_hold_end)
        if not keep.any():
            return empty
        return blocks[keep].astype(np.uint8), ts[keep]

    def _enter_result(self, fr: Frame, res: LivenessResult, seq: Sequence | None, events: list) -> None:
        r = self._round
        r.result, r.t_result = res, fr.t
        self._state = RESULT
        events.append(SessionEvent("RESULT", fr.t, {
            "result": res, "sequence": seq, "targets": list(r.targets), "t0": r.t0,
            "t_hold_start": r.t_hold_start, "t_hold_end": r.t_hold_end, "hold_match_ratio": r.hold_match_ratio}))

    def _output(self, fr: Frame, smooth: str, e, in_round: bool, events: list) -> SessionOutput:
        st, r, s = self._state, self._round, self._s
        phase_t = {IDLE: self._idle_t, COUNTDOWN: r and r.round_start, CHALLENGE: r and r.t0,
                   HOLD: r and r.t_hold_start, RESULT: r and r.t_result}[st]
        phase_len = {IDLE: None, COUNTDOWN: s["countdown_s"], CHALLENGE: s["challenge_timeout_s"],
                     HOLD: s["hold_s"], RESULT: s["result_s"]}[st]
        if st == CHALLENGE:
            progress = (0.0 if r.stable_start is None
                        else float(np.clip((fr.t - r.stable_start) / s["stable_hold_s"], 0.0, 1.0)))
        else:
            progress = 1.0 if st in (HOLD, RESULT) and r.t_hold_start is not None else 0.0
        res = r.result if st == RESULT else None
        reason_text = (REASON_TEXT[res.reasons[0]] if res.reasons else PASS_TEXT) if res else None
        return SessionOutput(
            phase=st, t=fr.t, t_in_phase=fr.t - phase_t, phase_len=phase_len,
            prompt=reason_text if res else PROMPTS[st],
            target=r.targets[0] if r and r.targets else None, targets=list(r.targets) if r else [],
            stable_progress=progress, first_match=bool(r and r.first_match),
            hint=quality.frame_hint(fr, self._q) if st in (CHALLENGE, HOLD) else None,
            verdict=res.verdict if res else None, reasons=list(res.reasons) if res else [],
            reason_text=reason_text, scores=dict(res.scores) if res else {}, total=res.total if res else None,
            raw_label=fr.raw_label, smooth_label=smooth, e=e,
            palm_ratio=palm_width_px(fr.lm_px) / fr.frame_wh[0] if fr.present else None,
            quality_ok=quality.frame_ok(fr, self._q) if fr.present else None,
            in_round=in_round, events=events)


def replay(frames: list[Frame], cfg: dict, score_fn, targets: list[str], t0: float | None,
           roi: np.ndarray | None = None, roi_t: np.ndarray | None = None) -> tuple[SessionOutput, list[SessionEvent]]:
    """Feed recorded frames through a fresh Session: space before the first frame, fixed target = targets[0],
    set_t0(t0) right after the REVEAL event when t0 is not None, stored ROI blocks used instead of per-frame crops.
    Returns the last output and every event (the RESULT event carries the LivenessResult)."""
    s = Session(cfg, score_fn, target=targets[0], external_roi=(roi, roi_t) if roi is not None else None)
    s.key("space")
    out, events = None, []
    for fr in frames:
        out = s.update(fr)
        events += out.events
        if t0 is not None and any(ev.kind == "REVEAL" for ev in out.events):
            s.set_t0(t0)
    return out, events
