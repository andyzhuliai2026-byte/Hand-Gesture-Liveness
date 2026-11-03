"""Per-frame data contract, palm geometry helpers and sample I/O (PROMPT.md 5.0, 7.3).

Portable core: numpy + stdlib only. Every producer (tracker, recorder) and consumer
(session, liveness, evaluate) goes through this module, so keep it the single source of truth.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NONE = "NONE"        # no hand in this frame
UNKNOWN = "UNKNOWN"  # hand present, no gesture rule matched
GESTURES = ("OPEN_PALM", "FIST", "THUMBS_UP", "V_SIGN", "POINT_UP", "OK")
LABELS = GESTURES + (UNKNOWN, NONE)

PALM_CENTER_IDX = (0, 5, 9, 13, 17)
SCHEMA_VERSION = 1
_LABEL_DTYPE = "<U16"


@dataclass
class Frame:
    """One camera/video frame, whether or not a hand was found (5.0)."""

    t: float                    # seconds; time.monotonic() after read() (camera) or recorded t (video)
    n_hands: int                # hands MediaPipe returned for this frame
    present: bool               # the selected hand exists
    lm_px: np.ndarray           # (21, 3) float32 on the mirrored frame: x*W, y*H, z*W; NaN if absent
    lm_world: np.ndarray        # (21, 3) float32 MediaPipe world landmarks, meters; NaN if absent
    hand_score: float           # handedness[i][0].score; NaN if absent
    raw_label: str              # per-frame gesture label: GESTURES | UNKNOWN | NONE
    roi_mean: float             # mean gray of the palm ROI; NaN if absent
    roi_sat_ratio: float        # fraction of palm-ROI pixels >= roi.sat_level; NaN if absent
    frame_wh: tuple[int, int]   # (width, height) of the frame in pixels


@dataclass
class Sequence:
    """One round handed to liveness.score() (PLAN.md 1)."""

    frames: list[Frame]          # COUNTDOWN start -> HOLD end (or the frame that timed out)
    roi: np.ndarray              # (M, size, size) uint8; blocks in [t0, HOLD end], >= roi_interval_s apart; (0, 0, 0) when none
    roi_t: np.ndarray            # (M,) float64
    targets: list[str]
    t0: float
    t_hold_start: float | None
    t_hold_end: float | None
    hold_match_ratio: float | None
    frame_wh: tuple[int, int]


def empty_frame(t: float, frame_wh: tuple[int, int]) -> Frame:
    """Frame with no hand: present=False, NaN geometry, label NONE."""
    nan = np.full((21, 3), np.nan, np.float32)
    return Frame(float(t), 0, False, nan, nan.copy(), float("nan"), NONE,
                 float("nan"), float("nan"), (int(frame_wh[0]), int(frame_wh[1])))


def palm_width_px(lm_px: np.ndarray) -> float:
    """Pixel palm width = max(|P5-P17|, |P0-P9|) on x, y. The max offsets foreshortening when the wrist tilts."""
    xy = np.asarray(lm_px, np.float64)[:, :2]
    return float(max(np.linalg.norm(xy[5] - xy[17]), np.linalg.norm(xy[0] - xy[9])))


def palm_center_px(lm_px: np.ndarray) -> np.ndarray:
    """(2,) mean x, y of P0, P5, P9, P13, P17 in pixels."""
    return np.asarray(lm_px, np.float64)[list(PALM_CENTER_IDX), :2].mean(axis=0)


def crop_roi_gray(img_bgr: np.ndarray, center_xy, size: int) -> np.ndarray:
    """(size, size) uint8 gray patch at native resolution, centered on center_xy.

    At the image border the window is shifted inward, never padded or resized (resizing would
    erase the moire frequencies 5.4 looks for). gray = 0.299 R + 0.587 G + 0.114 B.
    """
    h, w = img_bgr.shape[:2]
    if h < size or w < size:
        raise ValueError(f"frame {w}x{h} smaller than ROI size {size}")
    x0 = int(np.clip(np.rint(center_xy[0] - size / 2), 0, w - size))
    y0 = int(np.clip(np.rint(center_xy[1] - size / 2), 0, h - size))
    patch = img_bgr[y0:y0 + size, x0:x0 + size].astype(np.float32)
    gray = 0.114 * patch[..., 0] + 0.587 * patch[..., 1] + 0.299 * patch[..., 2]
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def roi_stats(roi: np.ndarray, sat_level: int) -> tuple[float, float]:
    """(mean gray, fraction of pixels >= sat_level)."""
    return float(roi.mean()), float((roi >= sat_level).mean())


# ---------------------------------------------------------------- sample I/O (7.3)

def frames_to_arrays(frames: list[Frame]) -> dict[str, np.ndarray]:
    n = len(frames)
    stack = (lambda name: np.stack([getattr(f, name) for f in frames]).astype(np.float32)
             if n else np.zeros((0, 21, 3), np.float32))
    return {
        "t": np.array([f.t for f in frames], np.float64),
        "n_hands": np.array([f.n_hands for f in frames], np.int8),
        "present": np.array([f.present for f in frames], bool),
        "lm_px": stack("lm_px"),
        "lm_world": stack("lm_world"),
        "hand_score": np.array([f.hand_score for f in frames], np.float32),
        "raw_label": np.array([f.raw_label for f in frames], _LABEL_DTYPE),
        "roi_mean": np.array([f.roi_mean for f in frames], np.float32),
        "roi_sat_ratio": np.array([f.roi_sat_ratio for f in frames], np.float32),
    }


def arrays_to_frames(arrs: dict, frame_wh) -> list[Frame]:
    wh = (int(frame_wh[0]), int(frame_wh[1]))
    return [
        Frame(float(arrs["t"][i]), int(arrs["n_hands"][i]), bool(arrs["present"][i]),
              arrs["lm_px"][i].astype(np.float32), arrs["lm_world"][i].astype(np.float32),
              float(arrs["hand_score"][i]), str(arrs["raw_label"][i]),
              float(arrs["roi_mean"][i]), float(arrs["roi_sat_ratio"][i]), wh)
        for i in range(len(arrs["t"]))
    ]


def save_sample(sample_dir, frames: list[Frame], meta: dict,
                roi: np.ndarray | None = None, roi_t: np.ndarray | None = None) -> None:
    """Write frames.npz (+ roi blocks) and meta.json. video.mp4 is written by the recorder."""
    d = Path(sample_dir)
    d.mkdir(parents=True, exist_ok=True)
    arrs = frames_to_arrays(frames)
    arrs["roi"] = np.asarray(roi, np.uint8) if roi is not None else np.zeros((0, 0, 0), np.uint8)
    arrs["roi_t"] = np.asarray(roi_t, np.float64) if roi_t is not None else np.zeros((0,), np.float64)
    np.savez_compressed(d / "frames.npz", **arrs)
    (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_sample(sample_dir) -> tuple[list[Frame], dict, np.ndarray, np.ndarray]:
    """-> (frames, meta, roi, roi_t). Reads only frames.npz + meta.json, never the video."""
    d = Path(sample_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    with np.load(d / "frames.npz", allow_pickle=False) as z:
        arrs = {k: z[k] for k in z.files}
    return arrays_to_frames(arrs, meta["frame_wh"]), meta, arrs["roi"], arrs["roi_t"]
