"""M4 calibration report from the calib half (PROMPT.md 5.1, 5.3, 5.6, 4, 7.1, 7.5). Never edits config.yaml.

    .venv/bin/python -m tools.calibrate [--data data] [--config config.yaml] [--out reports] [--export-fixture]
                                        [--fixture tests/fixtures/gestures_real.npz]
Freezing is manual: a human copies the proposals into config.yaml, cites the evidence in its comments, commits.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import gesture
from config import ConfigError, config_sha8, load_config
from liveness.frame import GESTURES, load_sample
from tools import evaluate as E

EXT_RATE_MARGIN = 1.3                               # [brief 5.1] max_ext_rate = real p99 x this, >= real p100
BRIEF_INITIAL = {"planarity": 0.4, "texture": 0.3}  # [brief 5.6] fusion floor initial values
FLOOR_RAISE_MAX = 0.2                               # [brief 5.6] a floor never exceeds initial + this
TAU_FACTOR = 2.0                                    # [brief 5.3] calib real R p5 >= this x tau_hard
R_CLASSES = ("real", "attack_photo", "attack_switch")
DIST_KEYS = ("r", "M", "R", "z_spread", "max_ext_rate_observed")
AUC_KEYS = ("nonrigid", "texture", "planarity")
FINGERS = ("T", "I", "M", "R", "P")
EXPECTED = gesture.TARGET_E  # [brief 4] table: thumb..pinky 1 extended / 0 curled / None = * (not checked)
FIXTURE = "tests/fixtures/gestures_real.npz"


def _pct(x, q) -> float:
    return float(np.percentile(x, q))


def _up(v: float, nd: int) -> float:
    """Round up to nd decimals, exact on values already on the grid."""
    r = round(v, nd)
    return r if r >= v else round(r + 10 ** -nd, nd)


def _down(v: float, nd: int) -> float:
    r = round(v, nd)
    return r if r <= v else round(r - 10 ** -nd, nd)


def propose_max_ext_rate(x: list[float]) -> float:
    """[brief 5.1] max(1.3 x p99, p100), rounded up to 0.1."""
    return _up(max(EXT_RATE_MARGIN * _pct(x, 99), max(x)), 1)


def propose_floor(k: str, scores: list[float]) -> float:
    """[brief 5.6] largest floor with calib FRR = 0 (min real score), capped at initial + 0.2, rounded down."""
    return _down(min(min(scores), BRIEF_INITIAL[k] + FLOOR_RAISE_MAX), 2)


def _real(ss):
    """real, variant != still, round reached the liveness check (TIMEOUT / WRONG_GESTURE rounds carry no debug)."""
    return [s for s in E._valid(ss) if s["cls"] == "real" and s["meta"]["variant"] != "still"]


# ---------------------------------------------------------------- sections

def sec_ext_rate(ss, cfg) -> list[str]:
    x = E._finite(s["res"].debug["max_ext_rate_observed"] for s in _real(ss))
    L = ["## 1. temporal.max_ext_rate [brief 5.1]", "",
         "x = debug max_ext_rate_observed of real calib samples; proposed = max(1.3 x p99, max), rounded up to 0.1.", ""]
    if not x:
        return L + ["no data"]
    return L + [E._row(["key", "current", "proposed", "n", "p50", "p99", "max"]), E._row(["---"] * 7),
                E._row(["temporal.max_ext_rate", cfg["temporal"]["max_ext_rate"], f"{propose_max_ext_rate(x):.1f}",
                        len(x), f"{_pct(x, 50):.3f}", f"{_pct(x, 99):.3f}", f"{max(x):.3f}"])]


def sec_floors(ss, cfg) -> list[str]:
    reached = [s for s in _real(ss) if s["res"].total is not None]  # reached step 3 of 5.6
    L = ["## 2. fusion floors [brief 5.6]", "",
         "real calib samples with a fused total; proposed = min(min score, initial + 0.2), rounded down to 0.01.", ""]
    if not reached:
        return L + ["no data"]
    L += [E._row(["key", "current", "proposed", "n", "min score"]), E._row(["---"] * 5)]
    for k in ("planarity", "texture"):
        x = E._finite(s["res"].scores[k] for s in reached)
        L.append(E._row([f"fusion.{k}_floor", cfg["fusion"][f"{k}_floor"], f"{propose_floor(k, x):.2f}",
                         len(x), f"{min(x):.3f}"]) if x else E._row([f"fusion.{k}_floor", "no data", "", 0, ""]))
    return L


def sec_tau(ss, cfg) -> list[str]:
    x = E._finite(s["res"].debug["R"] for s in _real(ss))
    L = ["## 3. planarity.tau_hard constraint [brief 5.3]", ""]
    if not x:
        return L + ["no data"]
    p5, need = _pct(x, 5), TAU_FACTOR * cfg["planarity"]["tau_hard"]
    L += [E._row(["n", "real R p5", "2 x tau_hard", "status"]), E._row(["---"] * 4),
          E._row([len(x), f"{p5:.4f}", f"{need:.4f}", "PASS" if p5 >= need else "VIOLATED"])]
    if p5 < need:
        L += ["", f"VIOLATED: lower planarity.tau_hard to <= {_down(p5 / 2, 4):.4f} (p5 / 2), or act on OPEN_PALM (section 4)."]
    return L


def sec_r_by_gesture(ss) -> list[str]:
    head = ["target", *(f"{c} n: min / p5 / p50 / p95 / max" for c in R_CLASSES), "flag"]
    L = ["## 4. R by target gesture [brief 5.3]", "", E._row(head), E._row(["---"] * len(head))]
    ss = E._valid(ss)  # (real, still) is never in the calib half: assign_halves puts it in "special"
    for g in GESTURES:
        x = {c: E._finite(s["res"].debug["R"] for s in ss if s["cls"] == c and s["meta"]["targets"][0] == g)
             for c in R_CLASSES}
        cells = [f"{len(v)}: " + " / ".join(f"{_pct(v, q):.4f}" for q in (0, 5, 50, 95, 100)) if v else "-"
                 for v in x.values()]
        overlap = x["real"] and x["attack_photo"] and min(x["real"]) <= max(x["attack_photo"])
        L.append(E._row([g, *cells, "OVERLAP" if overlap else ""]))
    return L


def sec_distributions(ss) -> list[str]:
    head = ["class", "n", *(f"{k} n: p5 / p50 / p95" for k in DIST_KEYS)]
    L = ["## 5. Distributions (report only)", "", E._row(head), E._row(["---"] * len(head))]
    by = {c: [s for s in E._valid(ss) if s["cls"] == c] for c in E.CLASSES}
    L += [E._row([c, len(v), *(E._pcts([s["res"].debug[k] for s in v]) for k in DIST_KEYS)]) for c, v in by.items()]
    sc = {c: {k: E._finite(s["res"].scores[k] for s in v) for k in AUC_KEYS} for c, v in by.items()}
    L += ["", E._row(["AUC real vs", *AUC_KEYS]), E._row(["---"] * 4)]
    L += [E._row([c, *(E._auc_cell(E._auc(sc["real"][k], sc[c][k])) for k in AUC_KEYS)]) for c in E.ATTACKS]
    return L + ["", "nonrigid.c/s, texture.c/s, planarity.c/s: PROMPT gives no formula — decide from these distributions "
                    "and write the reason in config.yaml comments"]


def sec_gestures(gs, cfg) -> list[str]:
    gc = cfg["gesture"]
    poses = {g: [] for g in GESTURES}
    hits = {g: 0 for g in GESTURES}
    for s in gs:
        frames, meta, _, _ = load_sample(s["dir"])
        tgt = meta["targets"][0]
        for fr in frames:
            if fr.present and fr.t >= meta["t0"] + E.GESTURE_SETTLE_S:
                p = gesture.analyze(fr.lm_world, fr.lm_px, gc)
                poses[tgt].append(p)
                hits[tgt] += p.label == tgt  # classify() == analyze().label
    L = ["## 6. Gesture thresholds (gestures calib half, frames t >= t0 + 1 s, present) [brief 4, 7.3]", "",
         E._row(["target", "n frames", "raw acc"]), E._row(["---"] * 3)]
    L += [E._row([g, len(poses[g]), E._f(hits[g] / len(poses[g]) if poses[g] else None)]) for g in GESTURES]
    L += ["", "margin = value - threshold. Extended rows want margin > 0 (p5): both pip2d and pip_deg must pass.",
          "Curled fingers: a frame is curled when EITHER feature is below its threshold, so read the curled "
          "pip2d/pip_deg rows together; the binding row is min(2d,3d), p95 of min(pip2d - pip_extended_deg, "
          "pip_deg - pip3d_min_deg), and wants margin < 0.", "",
          E._row(["target", "finger", "expected", "feature", "stat", "value", "threshold", "margin"]),
          E._row(["---"] * 8)]

    def row(g, finger, exp, feat, vals, thr):
        x = E._finite(vals)
        stat = "p5" if exp else "p95"
        v = _pct(x, 5 if exp else 95) if x else None
        L.append(E._row([g, finger, exp, feat, stat, E._f(v), thr, E._f(None if v is None else v - thr)]))

    for g in GESTURES:
        ps = poses[g]
        for i, exp in enumerate(EXPECTED[g]):
            if exp is None:
                continue
            exp = int(exp)
            if i == 0:
                row(g, "T", exp, "thumb_tip_ratio", [p.thumb_tip_ratio for p in ps], gc["thumb_tip_ratio"])
                continue
            row(g, FINGERS[i], exp, "pip2d_deg", [p.pip2d_deg[i - 1] for p in ps], gc["pip_extended_deg"])
            row(g, FINGERS[i], exp, "pip_deg", [p.pip_deg[i - 1] for p in ps], gc["pip3d_min_deg"])
            if not exp:  # dev-5 AND rule: curled iff either margin < 0
                row(g, FINGERS[i], exp, "min(2d,3d)", [min(p.pip2d_deg[i - 1] - gc["pip_extended_deg"],
                                                           p.pip_deg[i - 1] - gc["pip3d_min_deg"]) for p in ps], 0)
    # OK ring: tip gap below ok_tip_ratio on OK frames (p95), above it on OPEN_PALM (p5): the only other target with
    # middle, ring and pinky extended, so the only one the OK rule can take (gesture.analyze checks OK first)
    row("OK", "T-I", 0, "tip_gap_ratio", [p.tip_gap_ratio for p in poses["OK"]], gc["ok_tip_ratio"])
    row("OPEN_PALM", "T-I", 1, "tip_gap_ratio", [p.tip_gap_ratio for p in poses["OPEN_PALM"]], gc["ok_tip_ratio"])
    return L


FOOTER = ["## 7. Freeze procedure [brief 7.5]", "",
          "1. Apply the proposals by hand in config.yaml, each with its calib evidence (this report) in the comment.",
          "2. Commit config.yaml (frozen: no threshold may change after this from test-half results).",
          "3. Run `.venv/bin/python -m tools.evaluate --split test` — at most twice; the second run only after "
          "DESIGN.md records what changed and its calib evidence."]


# ---------------------------------------------------------------- main

def export_fixture(gs, path: Path) -> None:
    """[brief 7.1] keypoints only: no roi, no video, no hand_score."""
    t, present, lm_px, lm_world, sample, target, t0 = [], [], [], [], [], [], []
    for i, s in enumerate(gs):
        frames, meta, _, _ = load_sample(s["dir"])
        t += [f.t for f in frames]
        present += [f.present for f in frames]
        lm_px += [f.lm_px for f in frames]
        lm_world += [f.lm_world for f in frames]
        sample += [i] * len(frames)
        target.append(meta["targets"][0])
        t0.append(meta["t0"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, t=np.array(t, np.float64), present=np.array(present, bool),
                        lm_px=np.array(lm_px, np.float32).reshape(-1, 21, 3),
                        lm_world=np.array(lm_world, np.float32).reshape(-1, 21, 3),
                        sample=np.array(sample, np.int32), target=np.array(target, "<U16"), t0=np.array(t0, np.float64))
    print(f"fixture -> {path} ({path.stat().st_size} bytes, {len(gs)} samples, {len(t)} frames)")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=".venv/bin/python -m tools.calibrate", description=__doc__.split("\n")[0])
    p.add_argument("--data", default="data")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", default="reports")
    p.add_argument("--export-fixture", action="store_true", help="write the calib-half gesture keypoints fixture")
    p.add_argument("--fixture", default=FIXTURE)
    return p


def main(argv=None) -> int:
    a = _parser().parse_args(argv)
    try:
        cfg, sha8 = load_config(a.config), config_sha8(a.config)
    except ConfigError as e:              # same contract as main.py: error message, exit 2
        print(f"error: {e}", file=sys.stderr)
        return 2
    samples = E.discover(Path(a.data), cfg["session"]["challenge_len"])
    E.assign_halves(samples)
    calib = [s for s in samples if s["half"] == "calib"]
    live = [s for s in calib if s["cls"] != E.GESTURE_CLASS]
    gs = [s for s in calib if s["cls"] == E.GESTURE_CLASS]
    for s in live:
        s["res"] = E.replay_sample(s["dir"], cfg)[2]

    L = ["# Calibration: calib half", "", f"config {a.config} sha8 {sha8} | samples {len(live)} liveness, "
         f"{len(gs)} gestures | proposals only, config.yaml is not modified", ""]
    for sec in (sec_ext_rate(live, cfg), sec_floors(live, cfg), sec_tau(live, cfg), sec_r_by_gesture(live),
                sec_distributions(live), sec_gestures(gs, cfg), FOOTER):
        L += sec + [""]
    text = "\n".join(L)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = E._fresh(out / f"calib_{sha8}.md")
    path.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"report -> {path}", file=sys.stderr)
    if a.export_fixture:
        if not gs:  # never replace a good fixture with an empty one (e.g. a mistyped --data)
            print(f"no calib-half gestures samples under {a.data}: fixture {a.fixture} not written", file=sys.stderr)
            return 1
        export_fixture(gs, Path(a.fixture))
    return 0


if __name__ == "__main__":
    sys.exit(main())
