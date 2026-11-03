import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from liveness.frame import empty_frame, load_sample, save_sample  # noqa: E402

T = ("--target", "OPEN_PALM")


def run(*args):
    return subprocess.run([sys.executable, "main.py", *map(str, args)], cwd=ROOT,
                          capture_output=True, text=True, timeout=180)


def write_video(path, frames, size=(1280, 720)):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    for f in frames:
        vw.write(f)
    vw.release()


def gray_video(tmp_path, n=30):
    p = tmp_path / "gray.mp4"
    write_video(p, [np.full((720, 1280, 3), 128, np.uint8)] * n)
    return p


def test_headless_requires_video():
    assert run("--headless").returncode == 2


def test_record_rejects_video():
    assert run("--record", "x", "--video", "y").returncode == 2


def test_record_dir_must_be_a_class():
    r = run("--record", "data/attack_photos/")  # typo of attack_photo; rejected before the camera opens
    assert r.returncode == 2 and "--record" in r.stderr


def test_meta_repeated_flags_extend():
    from main import build_parser
    assert build_parser().parse_args(["--meta", "medium=phone", "--meta", "variant=still", "hand=left"]).meta == \
        ["medium=phone", "variant=still", "hand=left"]


def test_jsonable():
    from main import _jsonable
    got = _jsonable({"s": {3}, "f": np.float32(0.5), "a": np.arange(2), "t": (1, np.int64(2)), "p": Path("x")})
    assert got == {"s": [3], "f": 0.5, "a": [0, 1], "t": [1, 2], "p": "x"} and type(got["f"]) is float


def test_headless_raw_mp4(tmp_path):
    r = run("--video", gray_video(tmp_path), "--headless", "--bench", *T)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert len(lines) == 31
    assert all(l.split()[1] == "0" for l in lines[:-1])
    assert all(l.split()[4:] == ["COUNTDOWN", "-"] for l in lines[:-1])
    j = json.loads(lines[-1])
    assert "proc_ms_p50" in j and "proc_ms_p95" in j and j["frames"] == 30 and j["score_ms"] is None
    assert j["verdict"] is None and j["reasons"] == [] and j["final_state"] == "COUNTDOWN" and j["debug"] is None


def test_raw_mp4_requires_target(tmp_path):
    r = run("--video", gray_video(tmp_path, 3), "--headless")
    assert r.returncode == 2 and "--target is required" in r.stderr


def test_bad_target_and_meta(tmp_path):
    v = gray_video(tmp_path, 3)
    assert run("--video", v, "--headless", "--target", "WAVE").returncode == 2
    r = run("--video", v, "--headless", *T, "--meta", "foo=1")
    assert r.returncode == 2 and "--meta" in r.stderr
    assert run("--video", v, "--headless", *T, "--meta", "medium=paper").returncode == 2


def test_max_frames(tmp_path):
    r = run("--video", gray_video(tmp_path), "--headless", "--max-frames", 5, *T)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert len(lines) == 5 + 1
    assert json.loads(lines[-1])["frames"] == 5


def test_save_frames(tmp_path):
    out = tmp_path / "out"
    r = run("--video", gray_video(tmp_path), "--headless", "--debug", "--max-frames", 3, "--save-frames", out, *T)
    assert r.returncode == 0, r.stderr
    pngs = sorted(out.glob("*.png"))
    assert len(pngs) == 3
    assert cv2.imread(str(pngs[0])).shape[:2] == (720, 1280)


def write_sample(d, ts, meta, size=(320, 240)):
    d.mkdir()
    write_video(d / "video.mp4", [np.full((size[1], size[0], 3), 128, np.uint8)] * len(ts), size)
    save_sample(d, [empty_frame(t, size) for t in ts], {"schema": 1, "frame_wh": list(size), **meta})
    return d


def test_sample_dir_timestamps(tmp_path):
    ts = [0.0, 0.031, 0.07, 0.1, 0.133, 0.2, 0.231, 0.3, 0.341, 0.4]
    d = write_sample(tmp_path / "s", ts, {})
    assert run("--video", d, "--headless").returncode == 2  # no targets/t0 in meta
    r = run("--video", d, "--headless", *T)
    assert r.returncode == 0, r.stderr
    assert [float(l.split()[0]) for l in r.stdout.strip().splitlines()[:-1]] == [round(t, 3) for t in ts]


def test_hold_meta_mismatch_warns(tmp_path):
    # meta says the hold happened, the replay (no hand) times out: None vs a value is a mismatch
    meta = {"targets": ["FIST"], "t0": 1.02, "t_hold_start": 2.5, "t_hold_end": 4.0}
    d = write_sample(tmp_path / "s", [i / 30 for i in range(170)], meta)
    r = run("--video", d, "--headless")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1])["reasons"] == ["TIMEOUT"]
    assert "warning: t_hold_start None != meta 2.5" in r.stderr and "warning: t_hold_end None != meta 4.0" in r.stderr


def test_sample_npz_shorter_than_video(tmp_path):
    from camera import SourceError, VideoSource
    d = write_sample(tmp_path / "s", [i / 30 for i in range(10)], {"targets": ["FIST"], "t0": 0.0})
    save_sample(d, [empty_frame(i / 30, (320, 240)) for i in range(8)], {"schema": 1, "frame_wh": [320, 240]})
    assert run("--video", d, "--headless").returncode == 2  # 8 rows, 10 decoded frames
    # CAP_PROP_FRAME_COUNT can be wrong on real files and pass the open-time check: the decode loop must catch it too
    src = VideoSource(write_sample(tmp_path / "ok", [i / 30 for i in range(10)], {}))
    src._ts = src._ts[:4]
    with pytest.raises(SourceError):
        list(src)
    src.close()


def test_ui_result_has_no_top_left_prompt(monkeypatch):
    import types

    import ui
    calls = []
    monkeypatch.setattr(ui, "_text", lambda img, s, org, *a, **k: calls.append((s, org)))
    out = types.SimpleNamespace(phase="RESULT", prompt="Timeout", target="FIST", verdict="FAIL", reason_text="Timeout",
                                hint=None, quality_ok=None, smooth_label="NONE", t_in_phase=0.0, stable_progress=1.0)
    from config import load_config
    ui.draw(np.zeros((720, 1280, 3), np.uint8), empty_frame(0.0, (1280, 720)), None, out,
            {"fps": 30.0, "proc_ms": 1.0}, False, load_config("config.yaml"))
    assert (("Timeout", (16, 40)) not in calls) and [c for c in calls if c[0] == "Timeout"]
    out.phase, out.prompt = "IDLE", "Press SPACE to start"
    calls.clear()
    ui.draw(np.zeros((720, 1280, 3), np.uint8), empty_frame(0.0, (1280, 720)), None, out,
            {"fps": 30.0, "proc_ms": 1.0}, False, load_config("config.yaml"))
    assert ("Press SPACE to start", (16, 40)) in calls


SMOKE = ROOT / "data" / "smoke" / "20261104-005418"


@pytest.mark.skipif(not SMOKE.is_dir(), reason="local smoke sample absent")
def test_headless_json_has_debug():
    r = run("--video", SMOKE, "--headless", "--target", "OPEN_PALM", "--max-frames", 200)
    assert r.returncode == 0, r.stderr
    j = json.loads(r.stdout.strip().splitlines()[-1])
    assert j["verdict"] is not None and "qualified_ratio" in j["debug"]


def test_sample_dir_meta_targets(tmp_path):
    # no hand at all: REVEAL at t=1.0, first match never happens -> FAIL(TIMEOUT) after meta t0 + 4.0
    d = write_sample(tmp_path / "s", [i / 30 for i in range(170)], {"targets": ["FIST"], "t0": 1.02})
    r = run("--video", d, "--headless")
    assert r.returncode == 0, r.stderr
    j = json.loads(r.stdout.strip().splitlines()[-1])
    assert j["targets"] == ["FIST"] and j["t0"] == 1.02
    assert j["verdict"] == "FAIL" and j["reasons"] == ["TIMEOUT"] and j["final_state"] == "RESULT"
    assert "warning" not in r.stderr
    out = tmp_path / "png"  # also drives ui.draw through COUNTDOWN / CHALLENGE / RESULT
    r = run("--video", d, "--headless", "--target", "V_SIGN", "--debug", "--save-frames", out)
    assert r.returncode == 0, r.stderr
    assert len(list(out.glob("*.png"))) == 170
    assert "warning: --target overrides meta targets" in r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1])["targets"] == ["V_SIGN"]


def test_headless_emoji_hand(tmp_path):
    pil = pytest.importorskip("PIL.Image")
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Apple Color Emoji.ttc", 160)
    except OSError:
        pytest.skip("Apple Color Emoji font unavailable")
    im = pil.new("RGB", (200, 200), (200, 200, 200))
    ImageDraw.Draw(im).text((20, 20), "✋", font=font, embedded_color=True)
    hand = cv2.resize(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR), (480, 480), interpolation=cv2.INTER_CUBIC)
    frames = []
    for i in range(40):
        f = np.full((720, 1280, 3), 200, np.uint8)
        f[120:600, 400 + 2 * i:880 + 2 * i] = hand
        frames.append(f)
    p = tmp_path / "hand.mp4"
    write_video(p, frames)
    r = run("--video", p, "--headless", "--target", "OPEN_PALM")
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    flags = [int(l.split()[1]) for l in lines[:-1]]
    print("present ratio", sum(flags) / len(flags))
    assert len(flags) == 40 and sum(flags) >= 20
    j = json.loads(lines[-1])
    assert j["frames"] == 40 and "verdict" in j


def test_recorder_flow(tmp_path):
    from main import Recorder, build_parser, parse_meta
    args = build_parser().parse_args(["--meta", "medium=phone", "--meta", "hand=left"])
    base = {"camera_index": 0, "model_sha256": "x", **parse_meta(build_parser(), args.meta)}
    rec = Recorder(tmp_path / "cls", 30.0, base)
    img = np.full((720, 1280, 3), 100, np.uint8)
    frames = [empty_frame(i * 0.033, (1280, 720)) for i in range(5)]

    rec.start_round(0.0, (1280, 720))
    for fr in frames[:2]:
        rec.add(fr, img)
    gone = rec.dir
    rec.discard_round()
    assert not gone.exists() and rec.dir is None and rec.last is None

    rec.start_round(0.0, (1280, 720))
    for fr in frames:
        rec.add(fr, img)
    roi, roi_t = np.full((2, 64, 64), 7, np.uint8), np.array([0.0, 0.1])
    d = rec.finish(frames, roi, roi_t, {"targets": ["FIST"], "t0": 0.03, "live": {"verdict": "FAIL",
                   "debug": {"n": np.int64(3), "x": np.float32(0.5)}}})
    rec.discard_round()  # nothing in progress: the finished sample stays
    got, meta, groi, groi_t = load_sample(d)
    assert len(got) == 5 and np.array_equal(groi, roi) and np.array_equal(groi_t, roi_t)
    cap = cv2.VideoCapture(str(d / "video.mp4"))
    n = 0
    while cap.read()[0]:
        n += 1
    assert n == 5
    assert meta["class"] == "cls" and meta["n_frames"] == 5 and meta["discard"] is False
    assert (meta["medium"], meta["hand"], meta["variant"], meta["lighting"], meta["notes"]) == \
        ("phone", "left", "na", "na", "")
    assert meta["targets"] == ["FIST"] and meta["live"]["debug"] == {"n": 3, "x": 0.5}
    for k in ("schema", "frame_wh", "created", "camera_index", "model_sha256", "source", "fps_nominal"):
        assert k in meta
    rec.mark_discard()
    assert json.loads((d / "meta.json").read_text())["discard"] is True
    rec.close()
