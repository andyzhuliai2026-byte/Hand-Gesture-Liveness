"""HandLiveness M1: live camera / video replay with gesture overlay, recorder and headless bench."""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

import gesture
import ui
from camera import CameraSource, SourceError, VideoSource
from config import ConfigError, config_sha8, load_config
from hand_tracker import HandTracker, ModelError, ensure_model
from liveness.frame import SCHEMA_VERSION, Frame, save_sample


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HandLiveness M1")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", metavar="PATH")
    p.add_argument("--record", metavar="DIR")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--max-frames", type=int, metavar="N")
    p.add_argument("--save-frames", metavar="DIR")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--config", default="config.yaml")
    return p


class Recorder:
    def __init__(self, root: Path, fps: float, meta_extra: dict):
        self.root, self.fps, self.meta_extra = Path(root), fps, meta_extra
        self.dir: Path | None = None
        self.frames: list[Frame] = []
        self._vw = None
        self._wh = None

    def add(self, fr: Frame, img_bgr: np.ndarray) -> None:
        if self._vw is None:
            h, w = img_bgr.shape[:2]
            self._wh = (w, h)
            base = self.root / time.strftime("%Y%m%d-%H%M%S")
            d, k = base, 0
            while d.exists():
                k += 1
                d = base.with_name(f"{base.name}-{k}")
            d.mkdir(parents=True)
            self.dir = d
            self._vw = cv2.VideoWriter(str(d / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            if not self._vw.isOpened():
                raise SourceError(f"Cannot open video writer for {d / 'video.mp4'}")
        self._vw.write(img_bgr)
        self.frames.append(fr)

    def close(self) -> Path | None:
        if self._vw is None:
            return None
        self._vw.release()
        self._vw = None
        meta = {"schema": SCHEMA_VERSION, "class": self.root.name, "source": "camera",
                "frame_wh": list(self._wh), "n_frames": len(self.frames), "fps_nominal": self.fps,
                "created": datetime.datetime.now().isoformat(timespec="seconds"),
                "discard": False, "notes": "", **self.meta_extra}
        save_sample(self.dir, self.frames, meta)
        print(f"recorded {len(self.frames)} frames -> {self.dir}", file=sys.stderr)
        return self.dir


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.headless and not args.video:
        parser.error("--headless requires --video")
    if args.record and args.video:
        parser.error("--record cannot be used with --video")

    source = tracker = rec = None
    windowed = not args.headless
    try:
        cfg = load_config(args.config)
        gcfg = cfg["gesture"]
        sha = ensure_model(cfg["tracker"])
        source = VideoSource(args.video) if args.video else CameraSource(args.camera, cfg["camera"])
        tracker = HandTracker(cfg["tracker"], cfg["roi"], classify=lambda w, px: gesture.classify(w, px, gcfg))
        smoother = gesture.LabelSmoother(gcfg["smooth_window_s"])
        if args.record:
            rec = Recorder(Path(args.record), source.fps_nominal, {
                "camera_index": args.camera, "mediapipe_version": importlib.metadata.version("mediapipe"),
                "model_sha256": sha, "config_sha8": config_sha8(args.config)})
        if args.save_frames:
            Path(args.save_frames).mkdir(parents=True, exist_ok=True)
        if windowed:
            cv2.namedWindow(ui.WIN, cv2.WINDOW_AUTOSIZE)
        debug = args.debug
        ts, procs, present_n, n = deque(maxlen=31), deque(maxlen=30), 0, 0
        all_proc: list[float] = []
        try:
            for t, img in source:
                t0 = time.perf_counter()
                fr, _roi = tracker.process(img, t)
                pose = gesture.analyze(fr.lm_world, fr.lm_px, gcfg) if fr.present else None
                smooth = smoother.update(fr.t, fr.raw_label)
                proc_ms = (time.perf_counter() - t0) * 1000
                all_proc.append(proc_ms)
                procs.append(proc_ms)
                ts.append(fr.t)
                dt = np.diff(list(ts))
                fps = 1.0 / statistics.median(dt) if len(dt) and statistics.median(dt) > 0 else 0.0
                n += 1
                present_n += int(fr.present)
                if rec:
                    rec.add(fr, img)
                if args.headless:
                    print(f"{fr.t:.3f} {int(fr.present)} {fr.raw_label} {smooth}", flush=True)
                if windowed or args.save_frames:
                    canvas = ui.draw(img, fr, pose, smooth, {"fps": fps, "proc_ms": statistics.median(procs)},
                                     debug, gcfg)
                    if args.save_frames:
                        cv2.imwrite(str(Path(args.save_frames) / f"{n - 1:06d}.png"), canvas)
                if windowed:
                    if n == 1:
                        name = f"cam {args.camera}" if not args.video else Path(args.video).name
                        w, h = source.frame_wh
                        cv2.setWindowTitle(ui.WIN, f"HandLiveness  {name}  {w}x{h}")
                    cv2.imshow(ui.WIN, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("d"):
                        debug = not debug
                    if cv2.getWindowProperty(ui.WIN, cv2.WND_PROP_VISIBLE) < 1:
                        break
                if args.max_frames is not None and n >= args.max_frames:
                    break
        except KeyboardInterrupt:
            pass
        if args.bench:
            p50, p95 = (np.percentile(all_proc, [50, 95]) if all_proc else (0.0, 0.0))
            print(json.dumps({"frames": n, "present_frames": present_n,
                              "proc_ms_p50": round(float(p50), 2), "proc_ms_p95": round(float(p95), 2)}))
        return 0
    except (ConfigError, ModelError, SourceError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    finally:
        try:
            if rec:
                rec.close()
        finally:
            try:
                if tracker:
                    tracker.close()
            finally:
                if source:
                    source.close()
                if windowed:
                    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
