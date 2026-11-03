"""Frame sources (PROMPT.md 2): live camera (mirrored once here) and video replay (never flipped)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class SourceError(RuntimeError):
    """Environment / IO failure; main.py prints the message and exits 2."""


class CameraPermissionError(SourceError):
    pass


class CameraSource:
    meta = None

    def __init__(self, index: int, ccfg: dict):
        self._w, self._h = ccfg["width"], ccfg["height"]
        self._max_fail = ccfg["max_read_failures"]
        self.fps_nominal = ccfg["fps_nominal"]
        self.frame_wh: tuple[int, int] | None = None
        self._cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not self._cap.isOpened():
            self._cap.release()
            raise CameraPermissionError(
                f"Cannot open camera {index}. Allow camera for this terminal app in System Settings > "
                "Privacy & Security > Camera, then rerun. If the image comes from an iPhone "
                "(Continuity Camera), try --camera 1.")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._h)

    def __iter__(self) -> Iterator[tuple[float, np.ndarray]]:
        fails = 0
        while True:
            ok, frame = self._cap.read()
            t = time.monotonic()
            if not ok or frame is None:
                fails += 1
                if fails >= self._max_fail:
                    raise SourceError(f"Camera read failed {fails} times in a row")
                continue
            fails = 0
            if self.frame_wh is None:
                h, w = frame.shape[:2]
                self.frame_wh = (w, h)
                if (w, h) != (self._w, self._h):
                    print(f"[camera] requested {self._w}x{self._h}, got {w}x{h}; continuing", file=sys.stderr)
            yield t, cv2.flip(frame, 1)  # the only mirror in the system

    def close(self) -> None:
        self._cap.release()


class VideoSource:
    def __init__(self, path: str | Path):
        p = Path(path)
        self.meta = None
        self._ts = None
        if p.is_dir():
            video = p / "video.mp4"
            if not video.is_file():
                raise SourceError(f"{video} not found")
            if (p / "frames.npz").is_file():
                with np.load(p / "frames.npz", allow_pickle=False) as z:
                    self._ts = z["t"]
            if (p / "meta.json").is_file():
                self.meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        else:
            video = p
        self._cap = cv2.VideoCapture(str(video))
        if not video.is_file() or not self._cap.isOpened():
            self._cap.release()
            raise SourceError(f"Cannot open video {video}")
        self.fps_nominal = float(self._cap.get(cv2.CAP_PROP_FPS))
        self.frame_wh = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if self._ts is None and self.fps_nominal <= 0:
            self._cap.release()
            raise SourceError(f"{video} reports fps {self.fps_nominal} and has no frames.npz timestamps")
        if self._ts is not None:
            n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if n != len(self._ts):
                self._cap.release()
                raise SourceError(f"frames.npz has {len(self._ts)} rows but video has {n} frames")

    def __iter__(self) -> Iterator[tuple[float, np.ndarray]]:
        i = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            if self._ts is not None and i >= len(self._ts):
                raise SourceError(f"frames.npz has {len(self._ts)} rows but video decodes more frames")
            yield (float(self._ts[i]) if self._ts is not None else i / self.fps_nominal), frame
            i += 1
        if self._ts is not None and i != len(self._ts):
            raise SourceError(f"frames.npz has {len(self._ts)} rows but video yielded {i} frames")

    def close(self) -> None:
        self._cap.release()
