"""Replay regression on the user's real-hand smoke samples (dev-5).

data/smoke holds the user's own hand recordings: biometric data, kept on this machine only and
never committed. Samples that are absent are skipped one by one; the whole file is skipped when
none is present.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402
from gesture import classify  # noqa: E402
from liveness.frame import load_sample  # noqa: E402

# sample dir -> gesture -> segments in seconds from the first frame, labelled by eye from frame sheets
TRUTH = {
    # 2026-11-03, right hand, 4 gestures
    "20261103-233858": {
        "THUMBS_UP": ((1.25, 2.55), (7.0, 8.9)),
        "FIST": ((4.15, 5.25), (9.9, 13.2)),
        "OPEN_PALM": ((5.6, 6.65),),
        "V_SIGN": ((3.15, 3.75),),
    },
    # 2026-11-04, 6 gestures x both hands; FIST covers knuckles-to-camera and palm-to-camera.
    # The left-hand V (27.6-28.9 s) had the pinky raised, which is not a V_SIGN by definition, so it is left out.
    "20261104-005418": {
        "OPEN_PALM": ((2.2, 5.2), (20.0, 21.6)),
        "FIST": ((6.3, 11.0), (22.0, 25.4)),
        "THUMBS_UP": ((11.4, 13.3), (25.7, 27.1), (29.2, 29.8)),
        "V_SIGN": ((13.5, 15.1),),
        "POINT_UP": ((15.4, 16.5), (30.1, 31.6)),
        "OK": ((16.9, 18.3), (31.8, 33.0)),
    },
}
PRESENT = [name for name in TRUTH if (ROOT / "data/smoke" / name).exists()]
if not PRESENT:
    pytest.skip("no smoke sample present on this machine", allow_module_level=True)

G = load_config(str(ROOT / "config.yaml"))["gesture"]


@pytest.mark.parametrize("name", PRESENT)
def test_smoke_replay_accuracy(name):
    frames = load_sample(ROOT / "data/smoke" / name)[0]
    t0 = frames[0].t
    for g, segs in TRUTH[name].items():
        hits = [classify(fr.lm_world, fr.lm_px, G) == g for fr in frames
                if fr.present and any(a <= fr.t - t0 <= b for a, b in segs)]
        print(f"{name} {g} {sum(hits)}/{len(hits)}")
        assert hits and sum(hits) / len(hits) >= 0.9, (name, g, sum(hits), len(hits))
