"""tools/evaluate.py on a synthetic data tree (PROMPT.md 7.3-7.5)."""
import json
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from config import config_sha8, load_config
from liveness import LivenessResult
from liveness.frame import GESTURES
from tests.test_roundtrip import CONFIG, edited_config, play_round, write_sample
from tools import evaluate as E

SHA = config_sha8(CONFIG)
# real: 02 discard, 04 still -> split over 01 03 05 06 -> calib 01 05, test 03 06 (06 is a timeout)
REAL = [("01", "pass", "na", False), ("02", "pass", "na", True), ("03", "pass", "na", False),
        ("04", "pass", "still", False), ("05", "pass", "na", False), ("06", "timeout", "na", False)]


@pytest.fixture(scope="module")
def rounds():
    cfg = load_config(CONFIG)
    out = {"pass": play_round(cfg, "V_SIGN", "V_SIGN", live=True),
           "dark": play_round(cfg, "V_SIGN", "V_SIGN", roi_mean=10.0, live=True),
           "timeout": play_round(cfg, "V_SIGN", None), "wrong": play_round(cfg, "FIST", "OPEN_PALM")}
    out.update({g: play_round(cfg, g, g) for g in GESTURES})
    return out


def make_tree(root, rounds, photo_test="pass", real06="timeout", dark=("attack_switch", "attack_replay"),
              gesture_test_fist="FIST"):
    """data/real (above; 06 = `real06`), data/attack_photo 01 dark (calib), 02 `photo_test` (test), 03 discard;
    01 dark, 02 dark for each class in `dark`; data/gestures: each gesture twice in a row (calib, test),
    the test-half FIST sample replaced by `gesture_test_fist`."""
    real = [r if r[0] != "06" else ("06", real06, "na", False) for r in REAL]
    photo = [("01", "dark", "na", False), ("02", photo_test, "na", False), ("03", "pass", "na", True)]
    two_dark = [("01", "dark", "na", False), ("02", "dark", "na", False)]
    for cls, rows in (("real", real), ("attack_photo", photo), *((c, two_dark) for c in dark)):
        for name, kind, variant, discard in rows:
            write_sample(root / cls / f"20261104-0000{name}", cls, *rounds[kind], SHA, variant, discard)
    for j in range(2 * len(GESTURES)):
        g = GESTURES[j // 2]
        kind = gesture_test_fist if g == "FIST" and j % 2 else g
        write_sample(root / "gestures" / f"20261104-0000{j:02d}", "gestures", *rounds[kind], SHA)
    return root


def run(data, out, *args):
    return E.main(["--data", str(data), "--config", CONFIG, "--out", str(out), *args])


def test_auc():
    assert E._auc([1.0, 2.0], [0.0, 0.5]) == 1.0 and E._auc([1.0], [1.0]) == 0.5 and E._auc([0.0], [1.0]) == 0.0
    assert E._auc([], [1.0]) is None


def test_split_discard_variant(tmp_path, rounds):
    samples = E.discover(make_tree(tmp_path / "data", rounds), 1)
    E.assign_halves(samples)
    half = {(s["cls"], s["dir"].name[-2:]): s["half"] for s in samples}
    assert {k[1]: v for k, v in half.items() if k[0] == "real"} == {
        "01": "calib", "03": "test", "04": "special", "05": "calib", "06": "test"}
    assert {k[1]: v for k, v in half.items() if k[0] == "attack_photo"} == {"01": "calib", "02": "test"}
    assert E.discover(tmp_path / "data", 2) == []


def test_table_rows_and_variant_row(tmp_path, rounds, capsys):
    assert run(make_tree(tmp_path / "data", rounds), tmp_path / "rep", "--split", "all") == 0
    text = (tmp_path / "rep" / f"eval_all_{SHA}.md").read_text(encoding="utf-8")
    assert all(f"| {c} |" in text for c in E.CLASSES)
    assert "| still |" in text and "WARN" in text and "TIMEOUT" in text
    assert "Roundtrip: all consistent" in text and text.strip() in capsys.readouterr().out


def test_split_test_exit_code_and_failure_report(tmp_path, rounds):
    data, rep = make_tree(tmp_path / "bad", rounds, photo_test="pass"), tmp_path / "rep_bad"
    assert run(data, rep, "--split", "test") == 1  # attack_photo test half: 1 PASS of n=1 > floor(0.1)
    fail = (rep / f"failure_{SHA}.md").read_text(encoding="utf-8")
    assert all(h in fail for h in ("## ① ", "## ② ", "## ③ ", "## ④ 下一步建议"))
    assert "attack_photo/20261104-000002" in fail and "（由 M4 分析后填写，不超过 3 条）" in fail
    data, rep = make_tree(tmp_path / "good", rounds, photo_test="dark", real06="pass"), tmp_path / "rep_good"
    assert run(data, rep, "--split", "test") == 0 and not (rep / f"failure_{SHA}.md").exists()


def test_special_rows_by_class_and_variant(tmp_path, rounds):
    # attack_replay still / moving are split normally; preplay / handheld are special; real still is special
    rows = [("01", "still"), ("02", "moving"), ("03", "preplay"), ("04", "handheld"), ("05", "still")]
    for name, variant in rows:
        write_sample(tmp_path / "data" / "attack_replay" / f"20261104-0000{name}", "attack_replay", *rounds["dark"],
                     SHA, variant)
    write_sample(tmp_path / "data" / "real" / "20261104-000001", "real", *rounds["pass"], SHA, "still")
    samples = E.discover(tmp_path / "data", 1)
    E.assign_halves(samples)
    half = {(s["cls"], s["dir"].name[-2:]): s["half"] for s in samples}
    assert half == {("real", "01"): "special", ("attack_replay", "01"): "calib", ("attack_replay", "02"): "test",
                    ("attack_replay", "03"): "special", ("attack_replay", "04"): "special",
                    ("attack_replay", "05"): "calib"}
    for s in samples:
        s["res"] = E.replay_sample(s["dir"], load_config(CONFIG))[2]
    rep = [s for s in samples if s["cls"] == "attack_replay" and s["half"] != "special"]
    assert E.class_stats(rep)["n"] == 3


def test_split_test_gestures_without_flag(tmp_path, rounds):
    data = make_tree(tmp_path / "data", rounds, photo_test="dark", real06="pass", gesture_test_fist="wrong")
    assert run(data, tmp_path / "rep", "--split", "test") == 1  # FIST test sample: 0 % smoothed accuracy
    text = (tmp_path / "rep" / f"eval_test_{SHA}.md").read_text(encoding="utf-8")
    assert "| gestures worst per-gesture smoothed accuracy | 6 | 0.000 |" in text and "## Gestures" not in text


def test_real_n_counts_timeouts(tmp_path, rounds):
    data = make_tree(tmp_path / "data", rounds, photo_test="dark")  # real test half: 03 pass, 06 timeout
    samples = E.discover(data, 1)
    E.assign_halves(samples)
    test_real = [s for s in samples if s["cls"] == "real" and s["half"] == "test"]
    for s in test_real:
        s["res"] = E.replay_sample(s["dir"], load_config(CONFIG))[2]
    st = E.class_stats(test_real, count_unreached=True)
    assert (st["n"], st["PASS"], st["unreached"]["TIMEOUT"]) == (2, 1, 1)
    assert E.class_stats(test_real)["n"] == 1  # attack rows keep excluding it
    assert run(data, tmp_path / "rep", "--split", "test") == 1  # PASS 1 < ceil(0.9 * 2)


def test_criterion_with_no_samples_fails(tmp_path, rounds):
    data = make_tree(tmp_path / "data", rounds, photo_test="dark", real06="pass", dark=("attack_replay",))
    assert run(data, tmp_path / "rep", "--split", "test") == 1  # attack_switch n = 0
    fail = (tmp_path / "rep" / f"failure_{SHA}.md").read_text(encoding="utf-8")
    assert "| attack_switch PASS | 0 |" in fail


def test_no_samples_test_split(tmp_path, capsys):
    assert run(tmp_path / "nothing", tmp_path / "rep", "--split", "test") == 1
    assert "no samples: nothing to judge on the test split" in capsys.readouterr().out


def test_veto_numpy_bool():
    res = LivenessResult("FAIL", None, {}, ["PRERECORDED"], {
        "veto_prerecorded": np.bool_(True), "veto_instant_switch": np.bool_(False), "veto_planar": np.bool_(True)})
    st = E.class_stats([{"res": res}])
    assert st["veto"] == {"PRERECORDED": 1.0, "INSTANT_SWITCH": 0.0, "PLANAR": 1.0}
    assert E._variant_hit("preplay", LivenessResult("FAIL", None, {}, [], {"veto_prerecorded": np.bool_(True)}))


def test_reports_never_overwritten(tmp_path, rounds):
    data = make_tree(tmp_path / "data", rounds)
    assert run(data, tmp_path / "rep", "--split", "test") == 1
    assert run(data, tmp_path / "rep", "--split", "test") == 1
    names = sorted(p.name for p in (tmp_path / "rep").iterdir())
    assert names == sorted([f"eval_test_{SHA}.md", f"eval_test_{SHA}-2.md", f"failure_{SHA}.md",
                            f"failure_{SHA}-2.md", "test_runs.json"])


def test_third_test_run_refused(tmp_path, rounds, capsys):
    data, rep = make_tree(tmp_path / "data", rounds), tmp_path / "rep"
    assert run(data, rep, "--split", "test") == 1 and run(data, rep, "--split", "test") == 1
    runs = json.loads((rep / "test_runs.json").read_text(encoding="utf-8"))
    assert [(r["sha8"], r["n"]) for r in runs] == [(SHA, 11)] * 2  # test half: real 2, photo 1, switch 1, replay 1, gestures 6
    assert [r["split"] for r in runs] == ["test"] * 2
    assert all("created" in r for r in runs)
    before = sorted(p.name for p in rep.iterdir())
    capsys.readouterr()
    assert run(data, rep, "--split", "test") == 2
    err = capsys.readouterr().err
    assert "test_runs.json" in err and "DESIGN.md" in err
    assert sorted(p.name for p in rep.iterdir()) == before
    # S8 (final integration review): `all` shows the test half, so it is refused too; `calib` never counts nor refused
    assert run(data, rep, "--split", "all") == 2 and sorted(p.name for p in rep.iterdir()) == before
    assert run(data, rep, "--split", "calib") == 0
    assert len(json.loads((rep / "test_runs.json").read_text(encoding="utf-8"))) == 2


def test_test_run_without_test_samples_neither_counts_nor_refused(tmp_path, rounds):
    data, rep = tmp_path / "data", tmp_path / "rep"
    write_sample(data / "real" / "20261104-000001", "real", *rounds["pass"], SHA)  # a lone sample lands in calib
    assert run(data, rep, "--split", "test") == 1 and not (rep / "test_runs.json").exists()
    ledger = json.dumps([{"sha8": SHA, "created": "x", "n": 1}] * 2)
    (rep / "test_runs.json").write_text(ledger, encoding="utf-8")
    assert run(data, rep, "--split", "test") == 1  # not refused (2) with 2 entries
    assert run(data, rep, "--split", "all") == 0   # all without test samples: neither counted nor refused
    assert (rep / "test_runs.json").read_text(encoding="utf-8") == ledger


def test_gestures_confusion(tmp_path, rounds):
    samples = E.discover(make_tree(tmp_path / "data", rounds), 1)
    g = E.gesture_stats([s for s in samples if s["cls"] == "gestures"], load_config(CONFIG))
    m = g["matrix"]
    assert m.shape == (6, 6) and m.trace() == m.sum() > 0 and np.all(np.diag(m) > 0)
    assert g["smooth_acc"] == 1.0 and all(a == 1.0 for a in g["per_gesture"].values())
    assert run(tmp_path / "data", tmp_path / "rep", "--split", "all", "--gestures") == 0
    assert "## Gestures" in (tmp_path / "rep" / f"eval_all_{SHA}.md").read_text(encoding="utf-8")


def test_no_samples(tmp_path, capsys):
    assert run(tmp_path / "nothing", tmp_path / "rep") == 0 and "no samples" in capsys.readouterr().out


def test_r_by_target_and_debug_distributions(tmp_path, rounds):
    data = make_tree(tmp_path / "data", rounds)
    assert run(data, tmp_path / "rep", "--split", "all") == 0
    text = (tmp_path / "rep" / f"eval_all_{SHA}.md").read_text(encoding="utf-8")
    samples = E.discover(data, 1)
    E.assign_halves(samples)
    real = [E.replay_sample(s["dir"], load_config(CONFIG))[2] for s in samples
            if s["cls"] == "real" and s["half"] != "special"]
    r = [x.debug["R"] for x in real if x.reasons[:1] != ["TIMEOUT"]]
    assert len(r) == 3 and E._pcts(r).startswith("3: ")  # real 01 03 05 (06 timeout, 04 still aside)
    photo = [E.replay_sample(s["dir"], load_config(CONFIG))[2].debug["R"] for s in samples
             if s["cls"] == "attack_photo" and s["half"] != "special" and s["dir"].name.endswith("02")]
    assert "## R by target gesture (5.3)" in text and f"| V_SIGN | {E._pcts(r)} | {E._pcts(photo)} | - | - |" in text
    assert E._pcts(photo).startswith("1: ")
    assert "| FIST | - | - | - | - |" in text
    assert "## Debug distributions" in text and "| class | n reached | " + " | ".join(E.DIST_KEYS) + " |" in text
    assert f"| real | 3 | {E._pcts([x.debug['max_ext_rate_observed'] for x in real if x.reasons[:1] != ['TIMEOUT']])}" \
        in text
    dist = text.split("## Debug distributions")[1].split("## Variant rows")[0]
    assert "| attack_replay | 2 | " in dist and "| attack_switch | 2 | " in dist
    lines = text.splitlines()
    head = next(l for l in lines if l.startswith("| class | n | PASS")).split(" | ")
    real_row = next(l for l in lines if l.startswith("| real | 4 | ")).split(" | ")
    assert real_row[head.index("FAIL reasons")] == "-"  # real 06 is a TIMEOUT round, not a FAIL verdict


def test_pcts():
    assert E._pcts([5, 1, None, float("nan"), 3, 2, float("inf"), 4, True]) == "5: 1.2 / 3 / 4.8"
    assert E._pcts([7.25]) == "1: 7.25 / 7.25 / 7.25" and E._pcts([0.0012345]) == "1: 0.001234 / 0.001234 / 0.001234"
    assert E._pcts([]) == E._pcts([None, float("nan")]) == "-"


def test_fail_reasons_column():
    st = E.class_stats([{"res": LivenessResult("FAIL", None, {}, ["INSTANT_SWITCH"], {})},
                        {"res": LivenessResult("FAIL", None, {}, ["INSTANT_SWITCH"], {})},
                        {"res": LivenessResult("FAIL", None, {}, ["HAND_LOST"], {})},
                        {"res": LivenessResult("INSUFFICIENT", None, {}, ["NO_TRANSITION"], {})}])
    assert st["fail"] == Counter(INSTANT_SWITCH=2, HAND_LOST=1) and st["insufficient"] == Counter(NO_TRANSITION=1)
    lines = E.class_table({c: st for c in E.CLASSES})
    assert "FAIL reasons" in lines[0] and "| HAND_LOST 1, INSTANT_SWITCH 2 |" in lines[2]


def test_attack_replay_medium_rows(tmp_path, rounds):
    data = make_tree(tmp_path / "data", rounds)
    for name, medium in (("01", "phone"), ("02", "monitor")):
        p = data / "attack_replay" / f"20261104-0000{name}" / "meta.json"
        meta = json.loads(p.read_text(encoding="utf-8"))
        p.write_text(json.dumps({**meta, "medium": medium}), encoding="utf-8")
    assert run(data, tmp_path / "rep", "--split", "all") == 0
    text = (tmp_path / "rep" / f"eval_all_{SHA}.md").read_text(encoding="utf-8")
    replay = next(l for l in text.splitlines() if l.startswith("| attack_replay |"))
    for row in E.MEDIUM_ROWS:
        line = next(l for l in text.splitlines() if l.startswith(f"| {row} |"))
        assert line.split(" | ")[1] == "1" and len(line.split(" | ")) == len(replay.split(" | "))
        head = next(l for l in text.splitlines() if l.startswith("| class | n | PASS")).split(" | ")
        i = head.index("AUC temporal")
        assert line.split(" | ")[i] == replay.split(" | ")[i] == "0.500 (<0.7)"
    data2 = make_tree(tmp_path / "data2", rounds, photo_test="dark", real06="pass")  # medium rows stay out of 7.5
    assert run(data2, tmp_path / "rep2", "--split", "test") == 0


def test_label_drift_listed_apart(tmp_path, rounds):
    """pip_extended_deg 179.9 breaks V_SIGN: the real V_SIGN rounds (recorded PASS) relabel, never match and give
    no result; they show under label drift, not under roundtrip mismatches."""
    data = make_tree(tmp_path / "data", rounds)
    cfg = edited_config(tmp_path, "pip_extended_deg", 179.9)
    assert E.main(["--data", str(data), "--config", str(cfg), "--out", str(tmp_path / "rep"), "--split", "all"]) == 0
    text = (tmp_path / "rep" / f"eval_all_{config_sha8(cfg)}.md").read_text(encoding="utf-8")
    rt, drift = text.split("## Roundtrip")[1].split("## Label drift (gesture config changed)")
    drift = drift.split("## ")[0]
    assert "Roundtrip: all consistent" in rt
    assert all(f"- real/20261104-0000{n}: " in drift for n in ("01", "03", "05"))
    assert "frames relabeled, no result" in drift and "real/20261104-000006" not in drift  # timeout: no V_SIGN
    warn = text.split("## WARN")[1]
    assert "real/20261104-000003: " in warn and "has no RESULT" in warn and "counted as not passed" in warn
    assert "attack_photo/20261104-000002: " in warn and "left out of n" in warn


def test_failure_report_lists_real_without_result(tmp_path, rounds):
    """A real test sample whose relabeled replay has no RESULT counts as not passed (D16), so failure section ②
    lists it as a rejected real sample with its relabel count."""
    data = make_tree(tmp_path / "data", rounds, photo_test="dark")
    cfg = edited_config(tmp_path, "pip_extended_deg", 179.9)
    assert E.main(["--data", str(data), "--config", str(cfg), "--out", str(tmp_path / "rep"), "--split", "test"]) == 1
    sec2 = (tmp_path / "rep" / f"failure_{config_sha8(cfg)}.md").read_text(encoding="utf-8").split("## ② ")[1]
    row = next(r for r in sec2.split("## ③")[0].splitlines() if r.startswith("| real/20261104-000003 |"))
    assert row.split(" | ")[1] == "no result" and row.split(" | ")[2].endswith("frames relabeled")


def test_same_config_relabel_is_mismatch(tmp_path, rounds):
    frames, data = rounds["pass"]
    k = next(i for i, f in enumerate(frames) if f.present)
    frames = [replace(f, raw_label="UNKNOWN") if i == k else f for i, f in enumerate(frames)]
    write_sample(tmp_path / "data" / "real" / "20261104-000001", "real", frames, data, SHA)
    assert run(tmp_path / "data", tmp_path / "rep", "--split", "all") == 0
    text = (tmp_path / "rep" / f"eval_all_{SHA}.md").read_text(encoding="utf-8")
    assert ("- real/20261104-000001 (full): labels differ from the recording under the same config (1 frames)"
            in text)
    assert text.split("## Label drift (gesture config changed)")[1].split("## ")[0].strip() == "none"


def test_gesture_stats_uses_current_config(tmp_path, rounds):
    """thumb_tip_ratio 5.0 misreads every extended thumb: accuracy follows the config, not the recorded labels."""
    samples = [s for s in E.discover(make_tree(tmp_path / "data", rounds), 1) if s["cls"] == "gestures"]
    cfg = load_config(CONFIG)
    cfg["gesture"]["thumb_tip_ratio"] = 5.0
    g = E.gesture_stats(samples, cfg)
    assert g["per_gesture"]["THUMBS_UP"] == g["per_gesture_raw"]["THUMBS_UP"] == 0.0
    assert g["per_gesture"]["OPEN_PALM"] == 0.0 and g["per_gesture"]["FIST"] == 1.0 and g["smooth_acc"] < 0.95


def test_debug_per_sample(tmp_path, rounds):
    """5.6: the report prints the 15 debug keys for every replayed non-gestures sample of the split."""
    data = make_tree(tmp_path / "data", rounds)
    assert run(data, tmp_path / "rep", "--split", "all") == 0
    text = (tmp_path / "rep" / f"eval_all_{SHA}.md").read_text(encoding="utf-8")
    assert len(E.DEBUG_PRINT) == 15
    sec = text.split("## Debug distributions")[1].split("## Debug per sample (5.6)")[1].split("## Variant rows")[0]
    assert "| sample | class | verdict | " + " | ".join(E.DEBUG_PRINT) + " |" in sec
    rows = {l.split(" | ")[0][2:]: l.split(" | ") for l in sec.splitlines() if l.startswith("| ") and "/" in l[:30]}
    assert "real/20261104-000004" in rows and not any(k.startswith("gestures/") for k in rows)  # variant in, gestures out
    res = E.replay_sample(data / "real" / "20261104-000001", load_config(CONFIG))[2]
    row = rows["real/20261104-000001"]
    assert row[2] == res.verdict == "PASS" and row[3] == f"{res.debug['t_r']:.4g}"
    assert rows["real/20261104-000006"][2:4] == ["FAIL", "-"]  # TIMEOUT round: no liveness debug
    assert E._dcell("F", []) == E._dcell("t_switch", None) == E._dcell("normal_angles", []) == E._dcell("rho", None) == "-"
    assert E._dcell("F", [1, 3]) == "1,3" and E._dcell("t_switch", 1.234) == "1.23"
    assert E._dcell("normal_angles", [3.0, 12.34567]) == "n=2 max=12.35" and E._dcell("n_pairs", 4) == "4"
    assert E._dcell("rho", 0.309512) == "0.3095"   # 4 significant digits
    assert "| no result |" in E.debug_rows([{"cls": "real", "dir": tmp_path / "real" / "x", "res": None}])[2]


def test_bad_config_exits_2(tmp_path, capsys):
    bad = edited_config(tmp_path, "w_nonrigid", ".nan")
    assert E.main(["--data", str(tmp_path / "nothing"), "--config", str(bad), "--out", str(tmp_path / "rep")]) == 2
    assert "error: fusion.w_nonrigid" in capsys.readouterr().err and not (tmp_path / "rep").exists()


def test_report_path_on_stderr(tmp_path, rounds, capsys):
    data, rep = make_tree(tmp_path / "data", rounds), tmp_path / "rep"
    assert run(data, rep, "--split", "calib") == 0
    assert f"report -> {rep / f'eval_calib_{SHA}.md'}" in capsys.readouterr().err
    assert run(data, rep, "--split", "calib") == 0
    assert f"report -> {rep / f'eval_calib_{SHA}-2.md'}" in capsys.readouterr().err


def test_default_split_calib_no_ledger(tmp_path, rounds):
    data, rep = make_tree(tmp_path / "data", rounds), tmp_path / "rep"
    assert run(data, rep) == 0
    assert sorted(p.name for p in rep.iterdir()) == [f"eval_calib_{SHA}.md"]


def test_split_all_counts_in_ledger(tmp_path, rounds):
    data, rep = make_tree(tmp_path / "data", rounds), tmp_path / "rep"
    assert run(data, rep, "--split", "all") == 0
    runs = json.loads((rep / "test_runs.json").read_text(encoding="utf-8"))
    assert [(r["split"], r["n"]) for r in runs] == [("all", 11)]
    assert run(data, rep, "--split", "test") == 1
    before = sorted(p.name for p in rep.iterdir())
    assert run(data, rep, "--split", "all") == 2 and sorted(p.name for p in rep.iterdir()) == before


def test_attack_replay_miss_is_warn_only(tmp_path, rounds):
    """7.5: attack_replay only WARNs -- a miss or n = 0 keeps exit 0 but still writes the failure report."""
    data = make_tree(tmp_path / "miss", rounds, photo_test="dark", real06="pass")
    write_sample(data / "attack_replay" / "20261104-000002", "attack_replay", *rounds["pass"], SHA)  # test half PASS
    assert run(data, tmp_path / "rep_miss", "--split", "test") == 0
    fail = (tmp_path / "rep_miss" / f"failure_{SHA}.md").read_text(encoding="utf-8")
    assert "| attack_replay PASS (WARN only) | 1 | 1 |" in fail and "attack_replay/20261104-000002" in fail
    data = make_tree(tmp_path / "none", rounds, photo_test="dark", real06="pass", dark=("attack_switch",))
    assert run(data, tmp_path / "rep_none", "--split", "test") == 0
    assert "| attack_replay PASS (WARN only) | 0 |" in (tmp_path / "rep_none" / f"failure_{SHA}.md").read_text(
        encoding="utf-8")
