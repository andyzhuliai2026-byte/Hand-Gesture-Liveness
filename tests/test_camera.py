import json

import cv2
import numpy as np
import pytest

from camera import SourceError, VideoSource


def _write_video(path, n=20):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    img = np.zeros((240, 320, 3), np.uint8)
    img[:, :160] = 255
    for _ in range(n):
        vw.write(img)
    vw.release()


def _sample(tmp_path, rows):
    _write_video(tmp_path / "video.mp4")
    ts = np.cumsum(np.random.default_rng(0).uniform(0.02, 0.06, rows))
    np.savez(tmp_path / "frames.npz", t=ts)
    (tmp_path / "meta.json").write_text(json.dumps({"k": 1}))
    return ts


def test_sample_dir_uses_npz_t_and_no_flip(tmp_path):
    ts = _sample(tmp_path, 20)
    src = VideoSource(tmp_path)
    assert src.meta == {"k": 1} and src.frame_wh == (320, 240)
    out = list(src)
    src.close()
    assert [t for t, _ in out] == pytest.approx(list(ts))
    assert all(f[:, :160].mean() > f[:, 160:].mean() for _, f in out)


def test_npz_row_mismatch(tmp_path):
    _sample(tmp_path, 19)
    with pytest.raises(SourceError, match="19 rows"):
        VideoSource(tmp_path)


def test_bare_mp4(tmp_path):
    _write_video(tmp_path / "a.mp4")
    src = VideoSource(tmp_path / "a.mp4")
    assert src.meta is None
    out = list(src)
    src.close()
    assert len(out) == 20 and [t for t, _ in out] == pytest.approx([i / src.fps_nominal for i in range(20)])


def test_missing_path(tmp_path):
    with pytest.raises(SourceError):
        VideoSource(tmp_path / "nope.mp4")
    with pytest.raises(SourceError):
        VideoSource(tmp_path)  # dir without video.mp4
