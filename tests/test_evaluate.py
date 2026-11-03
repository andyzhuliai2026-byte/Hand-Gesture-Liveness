"""tools/evaluate.py on a synthetic data tree (PROMPT.md 7.3-7.5)."""
import numpy as np
import pytest

from config import config_sha8, load_config
from liveness import LivenessResult
from liveness.frame import GESTURES
from tests.test_roundtrip import CONFIG, play_round, write_sample
from tools import evaluate as E

SHA = config_sha8(CONFIG)
# real: 02 discard, 04 still -> split over 01 03 05 06 -> calib 01 05, test 03 06 (06 is a timeout)
REAL = [("01", "pass", "na", False), ("02", "pass", "na", True), ("03", "pass", "na", False),
        ("04", "pass", "still", False), ("05", "pass", "na", False), ("06", "timeout", "na", False)]


@pytest.fixture(scope="module")
def rounds():
    cfg = load_config(CONFIG)
    out = {"pass": play_round(cfg, "V_SIGN", "V_SIGN"), "dark": play_round(cfg, "V_SIGN", "V_SIGN", roi_mean=10.0),
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
                            f"failure_{SHA}-2.md"])


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
