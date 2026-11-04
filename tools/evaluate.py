"""Offline evaluation (PROMPT.md 7.3-7.5): replay recorded rounds through session + liveness.score.

Reads frames.npz + meta.json only; never the video, never MediaPipe. raw_label of every present frame is
recomputed with the current gesture config before replay and gesture stats (PLAN S7), never taken from the recording.
    .venv/bin/python -m tools.evaluate [--data data] [--split calib (default)|test|all] [--challenge-len 1] [--gestures]
                                       [--config config.yaml] [--out reports]
Exit code: --split test -> 0 when every 7.5 criterion that is not WARN-only holds, else 1 (a criterion with
n = 0 does not hold, no samples at all -> 1). attack_replay is WARN-only per 7.5 (a target: missing it is not a
delivery failure, but the failure report is required), so a miss or n = 0 there keeps exit 0 while
failure_<sha8>.md is still written. 2, with no report, when config.yaml fails validation, or when this run is
`test` / `all` with test-half samples and <out>/test_runs.json already holds 2 such runs (`calib` never counts nor
is refused). `calib` and `all` otherwise -> 0.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

import gesture
import session
from config import ConfigError, config_sha8, load_config
from gesture import LabelSmoother
from liveness import score
from liveness.frame import GESTURES, load_sample

CLASSES = ("real", "attack_photo", "attack_switch", "attack_replay")
ATTACKS = CLASSES[1:]
GESTURE_CLASS = "gestures"
SPECIAL = {"still": "NEED_ROTATION", "preplay": "PRERECORDED", "handheld": "TWO_HANDS"}  # [brief 7.3] variant -> expected
SPECIAL_ROWS = {("real", "still"), ("attack_replay", "preplay"), ("attack_photo", "handheld"),  # [brief 7.3]
                ("attack_switch", "handheld"), ("attack_replay", "handheld")}  # other variants are split normally
SPECIAL_MIN = 0.75           # [brief 7.5] expected hit rate of a variant row, else WARN
GESTURE_SETTLE_S = 1.0       # [brief 7.3] frames with t >= t0 + this count as the target gesture
VERDICTS = ("PASS", "FAIL", "INSUFFICIENT")
SCORE_KEYS = ("temporal", "nonrigid", "planarity", "texture")
LOSS_KEYS = ("planarity", "texture", "nonrigid")  # the fused scores (5.6)
DEBUG_KEYS = ("t_r", "gap_max_s", "R", "rho", "M", "qualified_ratio", "two_hands_ratio", "hold_match_ratio")
VETO_KEYS = {"PRERECORDED": "veto_prerecorded", "INSTANT_SWITCH": "veto_instant_switch", "PLANAR": "veto_planar"}
UNREACHED = ("TIMEOUT", "WRONG_GESTURE")  # the round never reached the liveness check
DIST_KEYS = ("max_ext_rate_observed", "R", "M", "r", "z_spread")  # [brief 5.1, 5.3] report only, z never scored
MEDIUM_ROWS = {"attack_replay/phone": ("phone",), "attack_replay/tablet+monitor": ("tablet", "monitor")}  # [brief 5.4]
TEST_RUNS_MAX = 2            # [brief 7.5] the test half runs at most twice
DEBUG_PRINT = ("t_r", "F", "max_ext_rate_observed", "gap_max_s", "t_switch", "r", "R", "rho", "n_pairs",  # [brief 5.6]
               "normal_angles", "M", "n_roi", "qualified_ratio", "two_hands_ratio", "hold_match_ratio")


# ---------------------------------------------------------------- samples

def discover(data: Path, challenge_len: int) -> list[dict]:
    """Sample dirs data/<class>/<name>/ with frames.npz + meta.json, sorted by name; discard and other
    challenge lengths skipped. -> [{"cls", "dir", "meta"}]"""
    out = []
    for cls in CLASSES + (GESTURE_CLASS,):
        root = Path(data) / cls
        dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
        for d in dirs:
            if not ((d / "frames.npz").is_file() and (d / "meta.json").is_file()):
                continue
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            if not meta["discard"] and meta["challenge_len"] == challenge_len:
                out.append({"cls": cls, "dir": d, "meta": meta})
    return out


def assign_halves(samples: list[dict]) -> None:
    """7.3 split: per class in name order, special (class, variant) rows aside, 1-based odd -> calib, even -> test."""
    count = Counter()
    for s in samples:
        if (s["cls"], s["meta"]["variant"]) in SPECIAL_ROWS:
            s["half"] = "special"
            continue
        count[s["cls"]] += 1
        s["half"] = "calib" if count[s["cls"]] % 2 else "test"


def relabel(frames: list, gcfg: dict) -> tuple[list, int]:
    """raw_label of present frames recomputed with gcfg (absent frames keep theirs). -> (frames, n changed)"""
    out = [replace(f, raw_label=gesture.classify(f.lm_world, f.lm_px, gcfg)) if f.present else f for f in frames]
    return out, sum(a.raw_label != b.raw_label for a, b in zip(frames, out))


def replay_sample(d, cfg: dict):
    """-> (relabeled frames, meta, LivenessResult of the replayed RESULT event or None when there is none,
    number of present frames whose label changed). Session and score() both see the current config's labels."""
    frames, meta, roi, roi_t = load_sample(d)
    frames, n_changed = relabel(frames, cfg["gesture"])
    _, events = session.replay(frames, cfg, lambda seq: score(seq, cfg), meta["targets"], meta["t0"], roi, roi_t)
    res = [e.data["result"] for e in events if e.kind == "RESULT"]
    return frames, meta, res[0] if res else None, n_changed


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=1e-5, equal_nan=True))


def roundtrip(live: dict, res, same_config: bool, n_relabel: int = 0) -> tuple[str, list[str]]:
    """meta["live"] vs the replayed result (None: no RESULT). Same config sha8: verdict, reasons, scores, and
    no relabeled frame; config changed: labels relabeled -> "drift" (expected, not a mismatch), else only the
    threshold-free debug values both sides have. -> ("full" | "debug" | "drift" | "replay", fields)"""
    if not same_config and n_relabel:
        return "drift", [f"{n_relabel} frames relabeled"] + (["no result"] if res is None else [])
    same_labels = [f"labels differ from the recording under the same config ({n_relabel} frames)"] \
        if same_config and n_relabel else []
    if res is None:
        return "replay", ["no result"] + same_labels
    if same_config:
        diffs = [k for k in ("verdict", "reasons") if live[k] != getattr(res, k)] + same_labels
        keys = sorted(set(live["scores"]) | set(res.scores))
        return "full", diffs + [f"scores.{k}" for k in keys if k not in live["scores"] or k not in res.scores
                                or not _close(live["scores"][k], res.scores[k])]
    common = [k for k in DEBUG_KEYS if k in live["debug"] and k in res.debug]
    return "debug", [f"debug.{k}" for k in common if not _close(live["debug"][k], res.debug[k])]


# ---------------------------------------------------------------- statistics

def _auc(pos, neg) -> float | None:
    """P(pos > neg) + P(tie) / 2 by the rank sum (Mann-Whitney U); None unless both sides have a value."""
    if len(pos) < 1 or len(neg) < 1:
        return None
    _, inv, counts = np.unique(np.r_[np.asarray(pos, float), np.asarray(neg, float)],
                               return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inv]  # 1-based, ties averaged
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def _unreached(res) -> bool:
    return bool(res.reasons) and res.reasons[0] in UNREACHED


def _valid(ss: list[dict]) -> list[dict]:
    """Samples whose round reached the liveness check."""
    return [s for s in ss if s["res"] is not None and not _unreached(s["res"])]


def _finite(xs) -> list[float]:
    return [float(x) for x in xs if isinstance(x, (int, float, np.number)) and not isinstance(x, bool)
            and np.isfinite(x)]


def class_stats(ss: list[dict], count_unreached: bool = False) -> dict:
    """n counts samples that reached the liveness check, timeouts and missing results listed apart;
    count_unreached (real) puts them in n as well, as not passed."""
    valid = _valid(ss)
    verdicts = Counter(s["res"].verdict for s in valid)
    return {
        "n": len(ss) if count_unreached else len(valid), "n_reached": len(valid), **{v: verdicts[v] for v in VERDICTS},
        "scores": {k: [s["res"].scores[k] for s in valid if k in s["res"].scores and s["res"].scores[k] is not None]
                   for k in SCORE_KEYS},
        "insufficient": Counter(r for s in valid if s["res"].verdict == "INSUFFICIENT" for r in s["res"].reasons),
        "fail": Counter(r for s in valid if s["res"].verdict == "FAIL" for r in s["res"].reasons),
        "debug": {k: _finite(s["res"].debug.get(k) for s in valid) for k in DIST_KEYS},
        # hard-veto hit rate from debug, independent of the 5.6 order; n/a until M3 writes the keys
        "veto": {c: None if not valid or any(k not in s["res"].debug for s in valid)
                 else sum(bool(s["res"].debug[k]) for s in valid) / len(valid) for c, k in VETO_KEYS.items()},
        "unreached": Counter(s["res"].reasons[0] for s in ss if s["res"] is not None and _unreached(s["res"])),
        "no_result": sum(s["res"] is None for s in ss),
    }


def _variant_hit(variant: str, res) -> bool:
    code = SPECIAL[variant]
    return code in res.reasons or (code in VETO_KEYS and VETO_KEYS[code] in res.debug
                                   and bool(res.debug[VETO_KEYS[code]]))


def r_by_target(ss: list[dict]) -> dict:
    """5.3 debug R of the samples that reached the check, by target gesture (meta targets[0]) and class."""
    out = {g: {c: [] for c in CLASSES} for g in GESTURES}
    for s in _valid(ss):
        out[s["meta"]["targets"][0]][s["cls"]].append(s["res"].debug.get("R"))
    return out


def gesture_stats(ss: list[dict], cfg: dict) -> dict:
    """Present frames with t >= t0 + 1 s: raw label 6x6 confusion (rows = target), raw and smoothed accuracy;
    labels recomputed with cfg["gesture"], never the recorded ones."""
    idx = {g: i for i, g in enumerate(GESTURES)}
    m, other = np.zeros((6, 6), int), Counter()
    n, hit = Counter(), Counter()
    for s in ss:
        frames, meta, _, _ = load_sample(s["dir"])
        frames, _ = relabel(frames, cfg["gesture"])
        tgt, t_from = meta["targets"][0], meta["t0"] + GESTURE_SETTLE_S
        sm = LabelSmoother(cfg["gesture"]["smooth_window_s"])
        for fr in frames:
            smooth = sm.update(fr.t, fr.raw_label)  # fed every frame, as the session does
            if fr.t < t_from or not fr.present:
                continue
            if fr.raw_label in idx:
                m[idx[tgt], idx[fr.raw_label]] += 1
            else:
                other[tgt] += 1
            n[tgt] += 1
            hit[tgt] += smooth == tgt
    total = sum(n.values())
    return {"matrix": m, "other": other, "n": n, "n_samples": len(ss),
            "per_gesture_raw": {g: m[i, i] / n[g] if n[g] else None for g, i in idx.items()},
            "per_gesture": {g: hit[g] / n[g] if n[g] else None for g in GESTURES},
            "smooth_acc": sum(hit.values()) / total if total else None}


def criteria(st: dict, gest: dict | None) -> list[dict]:
    """7.5 on the test half; a criterion with n = 0 does not hold; attack_replay only WARNs."""
    out = []

    def add(name, n, actual, op, target, warn_only=False):
        ok = n > 0 and actual is not None and (actual >= target if op == ">=" else actual <= target)
        gap = None if actual is None else max(0, target - actual) if op == ">=" else max(0, actual - target)
        out.append({"name": name, "n": n, "actual": actual, "target": f"{op} {_f(target)}", "gap": gap, "ok": ok,
                    "warn_only": warn_only})

    n = st["real"]["n"]
    add("real PASS", n, st["real"]["PASS"], ">=", -(-9 * n // 10))   # ceil(0.9 n), integer math
    add("real INSUFFICIENT", n, st["real"]["INSUFFICIENT"], "<=", n // 10)
    for c, tenths in (("attack_photo", 1), ("attack_switch", 1), ("attack_replay", 2)):
        add(f"{c} PASS", st[c]["n"], st[c]["PASS"], "<=", tenths * st[c]["n"] // 10, warn_only=c == "attack_replay")
    per = gest["per_gesture"].values()
    add("gestures smoothed accuracy", gest["n_samples"], gest["smooth_acc"], ">=", 0.95)
    add("gestures worst per-gesture smoothed accuracy", gest["n_samples"], None if None in per else min(per),
        ">=", 0.90)
    return out


# ---------------------------------------------------------------- rendering

def _f(x) -> str:
    if x is None:
        return "n/a"
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def _ms(v: list) -> str:
    return f"{np.mean(v):.3f} ± {np.std(v):.3f}" if v else "n/a"


def _row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _counts(c: Counter) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(c.items())) or "-"


def _pcts(v: list) -> str:
    """'n: p5 / p50 / p95' of the finite values, '-' when none."""
    v = _finite(v)
    return f"{len(v)}: " + " / ".join(f"{x:.4g}" for x in np.percentile(v, [5, 50, 95])) if v else "-"  # r, z ~ 1e-3


def _auc_cell(a) -> str:
    return "n/a" if a is None else f"{a:.3f}" + (" (<0.7)" if a < 0.7 else "")


def _name(s: dict) -> str:
    return f"{s['cls']}/{s['dir'].name}"


def _dcell(k: str, v) -> str:
    """One DEBUG_PRINT value: '-' when missing; F ints, t_switch 2 decimals, normal_angles 'n=.. max=..'."""
    if v is None:
        return "-"
    if k == "F":
        return ",".join(str(int(x)) for x in v) or "-"
    if k == "t_switch":
        return ",".join(f"{x:.2f}" for x in (v if isinstance(v, (list, tuple)) else [v])) or "-"
    if k == "normal_angles":
        return f"n={len(v)} max={max(v):.4g}" if len(v) else "-"
    return f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)


def debug_rows(ss: list[dict]) -> list[str]:
    """5.6 debug set per replayed sample (gestures class excluded)."""
    lines = [_row(["sample", "class", "verdict", *DEBUG_PRINT]), _row(["---"] * (len(DEBUG_PRINT) + 3))]
    for s in ss:
        r = s["res"]
        dbg = r.debug if r is not None else {}
        lines.append(_row([_name(s), s["cls"], r.verdict if r is not None else "no result",
                           *(_dcell(k, dbg.get(k)) for k in DEBUG_PRINT)]))
    return lines


def _largest_loss(scores: dict, fcfg: dict) -> str:
    if any(k not in scores or scores[k] is None for k in LOSS_KEYS):
        return "n/a"
    loss = {k: fcfg[f"w_{k}"] * (1.0 - scores[k]) for k in LOSS_KEYS}
    k = max(loss, key=lambda key: loss[key])
    return f"{k} {loss[k]:.3f}"


def class_table(st: dict) -> list[str]:
    head = (["class", "n", *VERDICTS, "pass rate", *SCORE_KEYS, *(f"AUC {k}" for k in SCORE_KEYS),
             "INSUFFICIENT reasons", "FAIL reasons", *(f"veto {c}" for c in VETO_KEYS), *UNREACHED, "no result"])
    lines = [_row(head), _row(["---"] * len(head))]
    for c in st:  # CLASSES, then the attack_replay medium rows (5.4, report only)
        r = st[c]
        auc = (["-"] * 4 if c == "real" else
               [_auc_cell(_auc(st["real"]["scores"][k], r["scores"][k])) for k in SCORE_KEYS])
        lines.append(_row([c, r["n"], *(r[v] for v in VERDICTS), _f(r["PASS"] / r["n"] if r["n"] else None),
                           *(_ms(r["scores"][k]) for k in SCORE_KEYS), *auc, _counts(r["insufficient"]), _counts(r["fail"]),
                           *(_f(r["veto"][v]) for v in VETO_KEYS), *(r["unreached"][u] for u in UNREACHED),
                           r["no_result"]]))
    return lines


def render(split, sha8, sel, st, special, mismatches, gest, crit, warns, by_target, drift=()) -> str:
    L = [f"# Evaluation: split {split}", "",
         f"config sha8 {sha8} | time {time.strftime('%Y-%m-%dT%H:%M:%S')} | samples {len(sel)}", "",
         "## Classes", "",
         "n: real counts every sample (TIMEOUT / WRONG_GESTURE / no result count as not passed); attack rows "
         "count samples that reached the liveness check (TIMEOUT / WRONG_GESTURE listed apart as not reached).", "",
         *class_table(st), "", "attack_replay/<medium> rows: report only, never in the 7.5 criteria.", "",
         "## R by target gesture (5.3)", "", "cell: n: p5 / p50 / p95 of finite debug R", "",
         _row(["target", *CLASSES]), _row(["---"] * (len(CLASSES) + 1)),
         *(_row([g, *(_pcts(by_target[g][c]) for c in CLASSES)]) for g in GESTURES), "",
         "## Debug distributions", "",
         "cell: n: p5 / p50 / p95 over samples that reached the check; attack_switch is judged by "
         "max_ext_rate_observed (5.1); z_spread is reported, never scored (5.3).", "",
         _row(["class", "n reached", *DIST_KEYS]), _row(["---"] * (len(DIST_KEYS) + 2)),
         *(_row([c, st[c]["n_reached"], *(_pcts(st[c]["debug"][k]) for k in DIST_KEYS)]) for c in CLASSES), "",
         "## Debug per sample (5.6)", "", "every replayed sample of the split, variant rows included; - = missing", "",
         *debug_rows([s for s in sel if s["cls"] != GESTURE_CLASS]), "",
         "## Variant rows (7.5, report only)", ""]
    if special:
        L += [_row(["variant", "expected", "n", "hit rate", "status"]), _row(["---"] * 5)]
        L += [_row([v, SPECIAL[v], n, _f(rate), "WARN" if rate is None or rate < SPECIAL_MIN else "ok"])
              for v, (n, rate) in special.items()]
    else:
        L.append("none")
    if gest is not None:  # --gestures only
        L += ["", "## Gestures (frames t >= t0 + 1 s, present)", "",
              _row(["target \\ raw", *GESTURES, "other", "raw acc", "smoothed acc"]), _row(["---"] * 10)]
        L += [_row([g, *gest["matrix"][i], gest["other"][g], _f(gest["per_gesture_raw"][g]),
                    _f(gest["per_gesture"][g])]) for i, g in enumerate(GESTURES)]
        L += ["", f"smoothed per-frame accuracy {_f(gest['smooth_acc'])}"]
    L += ["", "## Roundtrip", ""]
    L += [f"- {name} ({branch}): {', '.join(diffs)}" for name, branch, diffs in mismatches] or \
         ["Roundtrip: all consistent"]
    L += ["", "## Label drift (gesture config changed)", ""]
    L += [f"- {name}: {', '.join(diffs)}" for name, _, diffs in drift] or ["none"]
    if crit:
        L += ["", "## Criteria (7.5, test half)", "", _row(["criterion", "n", "actual", "target", "status"]),
              _row(["---"] * 5)]
        L += [_row([c["name"], c["n"], _f(c["actual"]), c["target"],
                    ("ok" if c["ok"] else "WARN" if c["warn_only"] else "FAIL") + (" (n = 0)" if not c["n"] else "")])
              for c in crit]
    L += ["", "## WARN", ""] + ([f"- WARN {w}" for w in warns] or ["none"])
    return "\n".join(L) + "\n"


def render_failure(sha8, crit, test_samples, calib_st, fcfg) -> str:
    L = [f"# Failure report {sha8}", "", "## ① 未达标判据", "",
         _row(["criterion", "n", "actual", "target", "gap"]), _row(["---"] * 5)]
    L += [_row([c["name"] + (" (WARN only)" if c["warn_only"] else ""), c["n"], _f(c["actual"]), c["target"],
                _f(c["gap"])]) for c in crit if not c["ok"]]
    L += ["", "## ② 误通过的攻击样本与误拒的真手样本", "",
          _row(["sample", "verdict", "reasons", *SCORE_KEYS, "largest loss"]), _row(["---"] * 8)]
    for s in test_samples:
        r = s["res"]
        if r is None and s["cls"] == "real":  # counted as not passed (D16)
            L.append(_row([_name(s), "no result", f"{s['relabel']} frames relabeled" if s["relabel"] else "-",
                           *["n/a"] * len(SCORE_KEYS), "n/a"]))
        elif r is not None and (r.verdict == "PASS") == (s["cls"] in ATTACKS):
            L.append(_row([_name(s), r.verdict, ", ".join(r.reasons) or "-",
                           *(_f(r.scores[k]) if k in r.scores else "n/a" for k in SCORE_KEYS),
                           _largest_loss(r.scores, fcfg)]))
    L += ["", "## ③ calib 半区各检查项 AUC（real 对各类攻击，< 0.7 视为该特征在当前数据上无效）", "",
          _row(["attack", *SCORE_KEYS]), _row(["---"] * 5)]
    L += [_row([c, *(_auc_cell(_auc(calib_st["real"]["scores"][k], calib_st[c]["scores"][k])) for k in SCORE_KEYS)])
          for c in ATTACKS]
    L += ["", "## ④ 下一步建议", "", "（由 M4 分析后填写，不超过 3 条）"]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main

def _fresh(path: Path) -> Path:
    """path, or path-2, path-3, ... : an existing report is never overwritten (7.5 keeps both test runs)."""
    p, k = path, 1
    while p.exists():
        k += 1
        p = path.with_name(f"{path.stem}-{k}{path.suffix}")
    return p


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=".venv/bin/python -m tools.evaluate", description=__doc__.split("\n")[0])
    p.add_argument("--data", default="data")
    p.add_argument("--split", choices=("calib", "test", "all"), default="calib")
    p.add_argument("--challenge-len", type=int, choices=(1, 2), default=1)
    p.add_argument("--gestures", action="store_true", help="6x6 confusion matrix and per-gesture accuracy")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", default="reports")
    return p


def main(argv=None) -> int:
    a = _parser().parse_args(argv)
    try:
        cfg, sha8 = load_config(a.config), config_sha8(a.config)
    except ConfigError as e:              # same contract as main.py: error message, exit 2
        print(f"error: {e}", file=sys.stderr)
        return 2
    samples = discover(Path(a.data), a.challenge_len)
    if not samples:
        print("no samples" + (": nothing to judge on the test split" if a.split == "test" else ""))
        return 1 if a.split == "test" else 0
    assign_halves(samples)
    out = Path(a.out)
    ledger, n_test = out / "test_runs.json", sum(s["half"] == "test" for s in samples)
    counted = a.split in ("test", "all") and n_test > 0  # every run that shows test-half results [brief 7.5]
    runs = json.loads(ledger.read_text(encoding="utf-8")) if counted and ledger.is_file() else []
    if counted and len(runs) >= TEST_RUNS_MAX:
        print(f"refused: --split {a.split} includes the test half and {ledger} already records {len(runs)} "
              f"test-half runs (7.5: at most {TEST_RUNS_MAX}; DESIGN.md must record what changed and its calib "
              "evidence before the 2nd)", file=sys.stderr)
        return 2
    for s in samples:
        s["res"], s["relabel"] = replay_sample(s["dir"], cfg)[2:] if s["cls"] != GESTURE_CLASS else (None, 0)

    sel = [s for s in samples if a.split == "all" or s["half"] in (a.split, "special")]
    live = [s for s in sel if s["cls"] != GESTURE_CLASS]
    rts = [(_name(s), *roundtrip(s["meta"]["live"], s["res"], s["meta"]["config_sha8"] == sha8, s["relabel"]))
           for s in live]
    mismatches = [r for r in rts if r[1] != "drift" and r[2]]
    drift = [r for r in rts if r[1] == "drift"]

    def stats(ss):
        res = {c: class_stats([s for s in ss if s["cls"] == c and s["half"] != "special"], c == "real")
               for c in CLASSES}
        rep = [s for s in ss if s["cls"] == "attack_replay" and s["half"] != "special"]
        res.update({row: class_stats([s for s in rep if s["meta"]["medium"] in media])
                    for row, media in MEDIUM_ROWS.items()})
        return res

    st = stats(sel)
    special = {}
    for v in SPECIAL:
        vs = [s for s in live if s["half"] == "special" and s["meta"]["variant"] == v]
        valid = [s for s in vs if s["res"] is not None and not _unreached(s["res"])]
        if vs:
            special[v] = (len(valid), sum(_variant_hit(v, s["res"]) for s in valid) / len(valid) if valid else None)
    gest = (gesture_stats([s for s in sel if s["cls"] == GESTURE_CLASS and s["half"] != "special"], cfg)
            if a.gestures or a.split == "test" else None)
    crit = criteria(st, gest) if a.split == "test" else []

    warns = [f"variant {v}: {SPECIAL[v]} hit rate {_f(rate)} < {SPECIAL_MIN}" for v, (_, rate) in special.items()
             if rate is None or rate < SPECIAL_MIN]
    warns += [f"{c['name']}: {_f(c['actual'])} (target {c['target']})" for c in crit if c["warn_only"] and not c["ok"]]
    # ponytail: visibility only; whether these leave n (D16) or keep the recorded HOLD timing is the user's call
    warns += [f"{_name(s)}: {s['relabel']} frames relabeled and the replay has no RESULT (the recording ends at the "
              "record-time HOLD end); " + ("counted as not passed" if s["cls"] == "real" and s["half"] != "special"
                                           else "left out of n") for s in live if s["res"] is None and s["relabel"]]
    warns += [f"{_name(s)}: meta medium {s['meta']['medium']} is in no attack_replay/<medium> row" for s in sel
              if s["cls"] == "attack_replay" and s["half"] != "special"
              and not any(s["meta"]["medium"] in m for m in MEDIUM_ROWS.values())]
    text = render(a.split, sha8, sel, st, special, mismatches, gest if a.gestures else None, crit, warns,
                  r_by_target([s for s in sel if s["half"] != "special"]), drift)
    out.mkdir(parents=True, exist_ok=True)
    report = _fresh(out / f"eval_{a.split}_{sha8}.md")
    report.write_text(text, encoding="utf-8")
    if counted:
        runs.append({"sha8": sha8, "split": a.split, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "n": n_test})
        ledger.write_text(json.dumps(runs, indent=1) + "\n", encoding="utf-8")
    print(text, end="")
    print(f"report -> {report}", file=sys.stderr)
    if any(not c["ok"] for c in crit):
        calib = stats([s for s in samples if s["half"] == "calib"])
        test_samples = [s for s in live if s["half"] == "test"]
        path = _fresh(out / f"failure_{sha8}.md")
        path.write_text(render_failure(sha8, crit, test_samples, calib, cfg["fusion"]), encoding="utf-8")
        print(f"failure report -> {path}", file=sys.stderr)
    return 1 if any(not c["ok"] and not c["warn_only"] for c in crit) else 0


if __name__ == "__main__":
    sys.exit(main())
