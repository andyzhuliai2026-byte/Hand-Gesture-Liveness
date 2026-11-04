"""tools/calibrate.py on a synthetic calib half (PROMPT.md 5.1, 5.3, 5.6, 7.1)."""
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from config import config_sha8, load_config
from liveness.frame import GESTURES
from tests.test_gestures_real import raw_accuracy
from tests.test_roundtrip import CONFIG, edited_config, play_round, write_sample
from tools import calibrate as C
from tools import evaluate as E

SHA = config_sha8(CONFIG)
KEYS = {"t", "present", "lm_px", "lm_world", "sample", "target", "t0"}


@pytest.fixture(scope="module")
def rounds():
    cfg = load_config(CONFIG)
    out = {"pass": play_round(cfg, "V_SIGN", "V_SIGN", live=True),
           "dark": play_round(cfg, "V_SIGN", "V_SIGN", roi_mean=10.0, live=True),
           "timeout": play_round(cfg, "V_SIGN", None),       # FAIL TIMEOUT, debug {}
           "wrong": play_round(cfg, "V_SIGN", "FIST")}       # FAIL WRONG_GESTURE, debug {}
    out.update({g: play_round(cfg, g, g) for g in GESTURES})
    return out


def make_tree(root, rounds, real=True):
    """real 01-03 pass (calib 01, 03) + 04 still (special); each attack class 01-02 dark (calib 01);
    gestures: each gesture twice in a row (calib, test)."""
    for name, variant in (("01", "na"), ("02", "na"), ("03", "na"), ("04", "still")) if real else ():
        write_sample(root / "real" / f"20261104-0000{name}", "real", *rounds["pass"], SHA, variant)
    for c in E.ATTACKS:
        for name in ("01", "02"):
            write_sample(root / c / f"20261104-0000{name}", c, *rounds["dark"], SHA)
    for j in range(2 * len(GESTURES)):
        write_sample(root / "gestures" / f"20261104-0000{j:02d}", "gestures", *rounds[GESTURES[j // 2]], SHA)
    return root


@pytest.fixture(scope="module")
def tree(tmp_path_factory, rounds):
    return make_tree(tmp_path_factory.mktemp("cal") / "data", rounds)


def run(data, out, *args):
    return C.main(["--data", str(data), "--config", CONFIG, "--out", str(out), *args])


def test_rounding():
    assert C._up(41.0, 1) == 41.0 and C._up(4.11, 1) == 4.2 and C._up(0.1 * 3, 1) == 0.4  # strict: 0.30000000000000004
    assert C._down(0.4 + 0.2, 2) == 0.6 and C._down(0.456, 2) == 0.45 and C._down(0.29, 2) == 0.29


def test_report_sections_and_proposals(tree, tmp_path, capsys):
    before = Path(CONFIG).read_bytes()
    assert run(tree, tmp_path / "rep") == 0
    assert Path(CONFIG).read_bytes() == before
    text = (tmp_path / "rep" / f"calib_{SHA}.md").read_text(encoding="utf-8")
    assert text.strip() in capsys.readouterr().out
    assert all(f"## {i}. " in text for i in range(1, 8)) and "PROMPT gives no formula" in text
    assert "| V_SIGN |" in text and "| OK | T-I | 0 | tip_gap_ratio | p95 |" in text

    cfg = load_config(CONFIG)
    res = [E.replay_sample(tree / "real" / f"20261104-0000{n}", cfg)[2] for n in ("01", "03")]  # calib; 04 still is special
    x = [r.debug["max_ext_rate_observed"] for r in res]
    want = math.ceil(max(1.3 * np.percentile(x, 99), max(x)) * 10) / 10
    assert f"| temporal.max_ext_rate | {cfg['temporal']['max_ext_rate']} | {want:.1f} | 2 |" in text
    assert all(r.total is not None and r.scores["planarity"] > 0.6 and r.scores["texture"] > 0.5 for r in res)
    fu = cfg["fusion"]  # current column from config.yaml, so freezing new [calib] values keeps this green
    assert f"| fusion.planarity_floor | {fu['planarity_floor']} | 0.60 | 2 |" in text
    assert f"| fusion.texture_floor | {fu['texture_floor']} | 0.50 | 2 |" in text


def test_no_data_without_real(tmp_path, rounds):
    data = make_tree(tmp_path / "data", rounds, real=False)
    assert run(data, tmp_path / "rep") == 0
    text = (tmp_path / "rep" / f"calib_{SHA}.md").read_text(encoding="utf-8")
    secs = text.split("\n## ")
    assert all("no data" in secs[i] for i in (1, 2, 3)) and "max_ext_rate |" not in text


def test_never_overwrites_and_no_fixture_without_flag(tree, tmp_path):
    assert run(tree, tmp_path, "--fixture", str(tmp_path / "fx.npz")) == 0 and run(tree, tmp_path) == 0
    assert (tmp_path / f"calib_{SHA}.md").is_file() and (tmp_path / f"calib_{SHA}-2.md").is_file()
    assert not (tmp_path / "fx.npz").exists()


def test_export_fixture_keypoints_only(tree, tmp_path, capsys):
    fx = tmp_path / "fixtures" / "g.npz"
    assert run(tree, tmp_path / "rep", "--export-fixture", "--fixture", str(fx)) == 0
    assert str(fx) in capsys.readouterr().out
    with np.load(fx, allow_pickle=False) as z:
        assert set(z.files) == KEYS
        assert list(z["target"]) == list(GESTURES) and z["lm_px"].shape[1:] == (21, 3)
        assert len(z["t"]) == len(z["sample"]) == len(z["present"]) and z["sample"].max() == len(GESTURES) - 1
    assert raw_accuracy(fx, load_config(CONFIG)["gesture"]) == 1.0


def fake(cls, R=0.1, score=0.5, reasons=(), target="V_SIGN"):
    dbg = dict.fromkeys(C.DIST_KEYS, 1.0) | {"R": R}
    return {"cls": cls, "meta": {"variant": "na", "targets": [target]},
            "res": NS(debug=dbg, scores=dict.fromkeys(C.AUC_KEYS, score), reasons=list(reasons), total=None)}


def test_rules():
    assert C.propose_max_ext_rate([10.0, 20.0, 30.0]) == 38.8           # 1.3 x p99 (29.8) = 38.74 -> 38.8
    assert C.propose_max_ext_rate([1.0] * 29 + [100.0]) == 100.0         # 1.3 x p99 = 92.7 < max
    assert C.propose_floor("planarity", [0.456, 0.9]) == 0.45 and C.propose_floor("texture", [0.95]) == 0.5
    cfg = load_config(CONFIG)
    out = "\n".join(C.sec_tau([fake("real", R=0.02478)] * 3, cfg))
    bound = float(out.split("tau_hard to <= ")[1].split()[0])
    assert "| VIOLATED |" in out and bound == 0.0123 and 2 * bound <= 0.02478  # rounded down, never the wrong side


def test_r_overlap_and_auc():
    ss = [fake("real", R=0.1), fake("real", R=0.2), fake("attack_photo", R=0.15),
          fake("real", R=0.1, target="FIST"), fake("attack_photo", R=0.05, target="FIST")]
    rows = {r.split(" |")[0][2:]: r for r in C.sec_r_by_gesture(ss) if r.startswith("| ")}
    assert rows["V_SIGN"].endswith("| OVERLAP |") and not rows["FIST"].endswith("OVERLAP |")
    ss = [fake("real", score=0.9), fake("real", score=0.8), fake("attack_photo", score=0.1),
          fake("attack_replay", score=0.85), fake("attack_replay", score=0.95)]
    text = "\n".join(C.sec_distributions(ss))
    assert "| attack_photo | 1.000 | 1.000 | 1.000 |" in text and "| attack_replay | 0.250 (<0.7) |" in text


def test_unreached_and_still_attacks(tmp_path, rounds):
    """TIMEOUT real + WRONG_GESTURE attack in calib are left out of n; attack_replay still samples are kept."""
    data = tmp_path / "data"
    for name, rd in (("01", "timeout"), ("02", "pass"), ("03", "pass")):     # calib: 01 timeout, 03 pass
        write_sample(data / "real" / f"20261104-0000{name}", "real", *rounds[rd], SHA)
    write_sample(data / "attack_switch" / "20261104-000001", "attack_switch", *rounds["wrong"], SHA)
    for name, v in (("01", "still"), ("02", "moving"), ("03", "still")):    # calib: 01, 03 (both still)
        write_sample(data / "attack_replay" / f"20261104-0000{name}", "attack_replay", *rounds["pass"], SHA, v)
    assert run(data, tmp_path / "rep") == 0
    text = (tmp_path / "rep" / f"calib_{SHA}.md").read_text(encoding="utf-8")
    assert "| temporal.max_ext_rate |" in text and "| 1 |" in text.split("## 1.")[1].split("## 2.")[0]
    assert "| real | 1 |" in text and "| attack_switch | 0 |" in text and "| attack_replay | 2 |" in text


def test_gesture_margins(monkeypatch):
    """Two-valued FIST pose pool: p95 and signed margin per row; frames before t0 + 1 s are ignored."""
    vals = [(0.5, 200.0, 200.0)] + [(1.0 + i / 10, *((100.0, 40.0) if i % 2 else (120.0, 60.0))) for i in range(20)]
    frames = [NS(present=True, t=t, lm_world=(a, b), lm_px=None) for t, a, b in vals]
    monkeypatch.setattr(C, "load_sample", lambda d: (frames, {"targets": ["FIST"], "t0": 0.0}, None, None))
    monkeypatch.setattr(C.gesture, "analyze", lambda w, px, gc: NS(
        label="FIST", pip2d_deg=(w[0],) * 4, pip_deg=(w[1],) * 4, thumb_tip_ratio=0.5, tip_gap_ratio=1.0))
    gc = load_config(CONFIG)["gesture"]   # thresholds and margins from config.yaml, so a [calib] freeze keeps this green
    t2, t3 = gc["pip_extended_deg"], gc["pip3d_min_deg"]
    text = "\n".join(C.sec_gestures([{"dir": None}], load_config(CONFIG)))
    assert "| FIST | 20 | 1.000 |" in text
    assert f"| FIST | I | 0 | pip2d_deg | p95 | 120.000 | {t2} | {E._f(120.0 - t2)} |" in text
    assert f"| FIST | I | 0 | pip_deg | p95 | 60.000 | {t3} | {E._f(60.0 - t3)} |" in text
    m = E._f(min(120.0 - t2, 60.0 - t3))    # p95 of the min on the 120/60 frames
    assert f"| FIST | I | 0 | min(2d,3d) | p95 | {m} | 0 | {m} |" in text


def test_export_fixture_refuses_empty(tmp_path):
    fx = tmp_path / "g.npz"
    fx.write_bytes(b"good")
    assert run(tmp_path / "missing", tmp_path / "rep", "--export-fixture", "--fixture", str(fx)) == 1
    assert fx.read_bytes() == b"good"


def test_bad_config_exits_2(tmp_path, capsys):
    bad = edited_config(tmp_path, "w_nonrigid", ".nan")
    assert C.main(["--data", str(tmp_path / "nothing"), "--config", str(bad), "--out", str(tmp_path / "rep")]) == 2
    assert "error: fusion.w_nonrigid" in capsys.readouterr().err and not (tmp_path / "rep").exists()
