import copy
import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from config import load_config
from hand_tracker import HandTracker, ModelError, ensure_model
from liveness.frame import NONE

CFG = load_config("config.yaml")
pytestmark = pytest.mark.skipif(not Path(CFG["tracker"]["model_path"]).exists(), reason="model not downloaded")


def test_black_frame_no_hand():
    with HandTracker(CFG["tracker"], CFG["roi"], lambda lm: "UNKNOWN") as tr:
        f, roi = tr.process(np.zeros((720, 1280, 3), np.uint8), 1.0)
        assert not f.present and f.n_hands == 0 and f.raw_label == NONE and roi is None
        assert f.frame_wh == (1280, 720)
        tr.process(np.zeros((720, 1280, 3), np.uint8), 1.0)  # same t twice: ts forced to increase


def _fail(*a, **k):
    raise AssertionError("network used")


def test_ensure_model_uses_local_file(tmp_path, monkeypatch):
    t = copy.deepcopy(CFG["tracker"])
    t["model_path"] = str(tmp_path / "m.task")
    (tmp_path / "m.task").write_bytes(b"abc")
    monkeypatch.setattr(urllib.request, "urlopen", _fail)
    sha = ensure_model(t)
    assert sha == hashlib.sha256(b"abc").hexdigest()
    assert (tmp_path / "m.task.sha256").read_text().split()[0] == sha


def test_ensure_model_download_failure(tmp_path, monkeypatch, capsys):
    t = copy.deepcopy(CFG["tracker"])
    t["model_path"] = str(tmp_path / "sub" / "m.task")

    def boom(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(ModelError):
        ensure_model(t)
    assert "curl" in capsys.readouterr().err
