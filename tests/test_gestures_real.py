"""Raw-label accuracy on real calib-half keypoints (PROMPT.md 7.1); skipped until tools.calibrate --export-fixture."""
from pathlib import Path

import numpy as np
import pytest

import gesture
from config import load_config
from tools.evaluate import GESTURE_SETTLE_S

FIXTURE = Path(__file__).parent / "fixtures" / "gestures_real.npz"
ACC_MIN = 0.95  # [brief 7.1]


def raw_accuracy(path, gcfg) -> float:
    """classify() == target over present frames with t >= t0 + 1 s; NaN when there are none."""
    with np.load(path, allow_pickle=False) as z:
        f = {k: z[k] for k in z.files}
    tgt, t0 = f["target"][f["sample"]], f["t0"][f["sample"]]
    keep = np.flatnonzero(f["present"] & (f["t"] >= t0 + GESTURE_SETTLE_S))
    hits = [gesture.classify(f["lm_world"][i], f["lm_px"][i], gcfg) == tgt[i] for i in keep]
    return float(np.mean(hits)) if hits else float("nan")


@pytest.mark.skipif(not FIXTURE.is_file(), reason="no fixture: .venv/bin/python -m tools.calibrate --export-fixture")
def test_raw_label_accuracy():
    assert raw_accuracy(FIXTURE, load_config("config.yaml")["gesture"]) >= ACC_MIN
