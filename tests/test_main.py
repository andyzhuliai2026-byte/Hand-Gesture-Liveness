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


def test_headless_raw_mp4(tmp_path):
    r = run("--video", gray_video(tmp_path), "--headless", "--bench")
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert len(lines) == 31
    assert all(l.split()[1] == "0" for l in lines[:-1])
    j = json.loads(lines[-1])
    assert "proc_ms_p50" in j and "proc_ms_p95" in j and j["frames"] == 30


def test_max_frames(tmp_path):
    r = run("--video", gray_video(tmp_path), "--headless", "--max-frames", 5)
    assert r.returncode == 0, r.stderr
    assert len(r.stdout.strip().splitlines()) == 5


def test_save_frames(tmp_path):
    out = tmp_path / "out"
    r = run("--video", gray_video(tmp_path), "--headless", "--debug", "--max-frames", 3, "--save-frames", out)
    assert r.returncode == 0, r.stderr
    pngs = sorted(out.glob("*.png"))
    assert len(pngs) == 3
    assert cv2.imread(str(pngs[0])).shape[:2] == (720, 1280)


def test_sample_dir_timestamps(tmp_path):
    ts = [0.0, 0.031, 0.07, 0.1, 0.133, 0.2, 0.231, 0.3, 0.341, 0.4]
    d = tmp_path / "s"
    d.mkdir()
    write_video(d / "video.mp4", [np.full((720, 1280, 3), 128, np.uint8)] * 10)
    save_sample(d, [empty_frame(t, (1280, 720)) for t in ts],
                {"schema": 1, "frame_wh": [1280, 720]})
    r = run("--video", d, "--headless")
    assert r.returncode == 0, r.stderr
    assert [float(l.split()[0]) for l in r.stdout.strip().splitlines()] == [round(t, 3) for t in ts]


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
    r = run("--video", p, "--headless")
    assert r.returncode == 0, r.stderr
    flags = [int(l.split()[1]) for l in r.stdout.strip().splitlines()]
    print("present ratio", sum(flags) / len(flags))
    assert len(flags) == 40 and sum(flags) >= 20


def test_recorder_roundtrip(tmp_path):
    from main import Recorder
    rec = Recorder(tmp_path / "cls", 30.0, {"camera_index": 0, "model_sha256": "x"})
    img = np.full((720, 1280, 3), 100, np.uint8)
    for i in range(5):
        rec.add(empty_frame(i * 0.033, (1280, 720)), img)
    d = rec.close()
    frames, meta, _, _ = load_sample(d)
    assert len(frames) == 5
    cap = cv2.VideoCapture(str(d / "video.mp4"))
    n = 0
    while cap.read()[0]:
        n += 1
    assert n == 5
    for k in ("schema", "class", "frame_wh", "n_frames", "created", "discard", "notes", "camera_index", "model_sha256"):
        assert k in meta
    assert meta["class"] == "cls" and meta["n_frames"] == 5
