"""MediaPipe HandLandmarker wrapper (PROMPT.md 2). Gesture classification is injected by the caller."""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import Callable

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

from liveness.frame import Frame, crop_roi_gray, empty_frame, palm_center_px, roi_stats


class ModelError(RuntimeError):
    pass


def ensure_model(tcfg: dict) -> str:
    """Make sure the model file exists (download if not); return its sha256 hex."""
    path = Path(tcfg["model_path"])
    if not path.exists():
        url = tcfg["model_url"]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_name(path.name + ".part")
            with urllib.request.urlopen(url, timeout=60) as r:
                part.write_bytes(r.read())
            os.replace(part, path)
        except Exception as e:
            print(f"Model download failed: {e}", file=sys.stderr)
            print(f"mkdir -p {path.parent} && curl -L -o {path} {url}", file=sys.stderr)
            raise ModelError(f"Model download failed: {e}") from e
    sha_file = path.with_name(path.name + ".sha256")
    if sha_file.exists():
        return sha_file.read_text().split()[0]
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    sha_file.write_text(sha + "\n")
    return sha


class HandTracker:
    def __init__(self, tcfg: dict, rcfg: dict, classify: Callable[[np.ndarray, np.ndarray], str]):
        self._rcfg, self._classify = rcfg, classify
        opts = vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=tcfg["model_path"]),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=tcfg["num_hands"],
            min_hand_detection_confidence=tcfg["min_hand_detection_confidence"],
            min_hand_presence_confidence=tcfg["min_hand_presence_confidence"],
            min_tracking_confidence=tcfg["min_tracking_confidence"])
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self._last_ts_ms = -1
        self._last_pos = None

    def process(self, img_bgr: np.ndarray, t: float) -> tuple[Frame, np.ndarray | None]:
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ts_ms = max(int(t * 1000), self._last_ts_ms + 1)  # MediaPipe needs strictly increasing ms
        self._last_ts_ms = ts_ms
        res = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
        n = len(res.hand_landmarks)
        if n == 0:
            return empty_frame(t, (w, h)), None
        lms = [np.array([[p.x * w, p.y * h, p.z * w] for p in hand], np.float32) for hand in res.hand_landmarks]
        if self._last_pos is None:
            i = max(range(n), key=lambda k: res.handedness[k][0].score)
        else:
            i = min(range(n), key=lambda k: np.linalg.norm(palm_center_px(lms[k]) - self._last_pos))
        lm_px = lms[i]
        lm_world = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[i]], np.float32)
        center = palm_center_px(lm_px)
        roi = crop_roi_gray(img_bgr, center, self._rcfg["size"])
        roi_mean, roi_sat = roi_stats(roi, self._rcfg["sat_level"])
        self._last_pos = center
        frame = Frame(float(t), n, True, lm_px, lm_world, float(res.handedness[i][0].score),
                      self._classify(lm_world, lm_px), roi_mean, roi_sat, (w, h))
        return frame, roi

    def reset(self) -> None:
        self._last_pos = None

    def close(self) -> None:
        self._lm.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
