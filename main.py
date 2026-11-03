"""HandLiveness M2: challenge-response session on camera / video replay, per-round recorder, headless bench."""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import os
import shutil
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

import gesture
import liveness
import ui
from camera import CameraSource, SourceError, VideoSource
from config import ConfigError, config_sha8, load_config
from hand_tracker import HandTracker, ModelError, ensure_model
from liveness.frame import GESTURES, SCHEMA_VERSION, Frame, save_sample
from session import RESULT, Session

META_ENUMS = {  # PROMPT.md 7.3; notes is free text
    "medium": ("none", "phone", "tablet", "monitor", "print", "na"),
    "variant": ("still", "moving", "preplay", "handheld", "na"),
    "lighting": ("normal", "dim", "backlit", "na"),
    "hand": ("left", "right", "na"),
}
RECORD_CLASSES = ("smoke", "real", "attack_photo", "attack_switch", "attack_replay", "gestures")  # PROMPT.md 7.3 classes + M1 smoke


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HandLiveness M2")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", metavar="PATH")
    p.add_argument("--record", metavar="DIR")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--max-frames", type=int, metavar="N")
    p.add_argument("--save-frames", metavar="DIR")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--target", choices=GESTURES, metavar="ID")
    p.add_argument("--seed", type=int, metavar="N")
    p.add_argument("--meta", action="extend", nargs="+", default=[], metavar="K=V")
    return p


def parse_meta(parser: argparse.ArgumentParser, items: list[str]) -> dict:
    meta = {k: "na" for k in META_ENUMS} | {"notes": ""}
    for item in items:
        k, sep, v = item.partition("=")
        if not sep or k not in meta or (k in META_ENUMS and v not in META_ENUMS[k]):
            allowed = ", ".join(f"{name}={'|'.join(vals)}" for name, vals in META_ENUMS.items())
            parser.error(f"--meta {item!r}: allowed {allowed}, notes=<text>")
        meta[k] = v
    return meta


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _jsonable(x):
    return json.loads(json.dumps(x, default=_json_default))  # numpy / sets / anything else -> plain JSON


class Recorder:
    """One sample directory per round (PROMPT.md 7.3): video.mp4 while the round runs, frames.npz + meta.json at RESULT."""

    def __init__(self, root: Path, fps: float, meta_base: dict):
        self.root, self.fps, self.meta_base = Path(root), fps, meta_base
        self.dir: Path | None = None    # round in progress
        self.last: Path | None = None   # last finished round
        self.frames: list[Frame] = []
        self._vw = None
        self._wh = None

    def start_round(self, t: float, frame_wh) -> None:
        base = self.root / time.strftime("%Y%m%d-%H%M%S")  # ponytail: wall-clock name only, never a timer
        d, k = base, 0
        while d.exists():
            k += 1
            d = base.with_name(f"{base.name}-{k}")
        d.mkdir(parents=True)
        self.dir, self._wh, self.frames = d, (int(frame_wh[0]), int(frame_wh[1])), []
        self._vw = cv2.VideoWriter(str(d / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self._wh)
        if not self._vw.isOpened():
            raise SourceError(f"Cannot open video writer for {d / 'video.mp4'}")

    def add(self, fr: Frame, img_bgr: np.ndarray) -> None:
        if self._vw is not None:
            self._vw.write(img_bgr)
            self.frames.append(fr)

    def finish(self, frames: list[Frame], roi, roi_t, meta_round: dict) -> Path:
        self._release()
        meta = {"schema": SCHEMA_VERSION, "class": self.root.name, "source": "camera", "frame_wh": list(self._wh),
                "n_frames": len(frames), "fps_nominal": self.fps,
                "created": datetime.datetime.now().isoformat(timespec="seconds"), "discard": False,
                **self.meta_base, **meta_round}
        save_sample(self.dir, frames, _jsonable(meta), roi, roi_t)
        print(f"recorded {len(frames)} frames -> {self.dir}", file=sys.stderr)
        self.last, self.dir = self.dir, None
        return self.last

    def discard_round(self) -> None:
        self._release()
        if self.dir is not None:
            shutil.rmtree(self.dir, ignore_errors=True)
            self.dir = None

    def mark_discard(self) -> None:
        if self.last is None:
            return
        p = self.last / "meta.json"
        meta = json.loads(p.read_text(encoding="utf-8"))
        meta["discard"] = True
        p.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"marked discard -> {self.last}", file=sys.stderr)

    def _release(self) -> None:
        if self._vw is not None:
            self._vw.release()
            self._vw = None

    def close(self) -> None:
        self._release()


def round_summary(ev) -> dict:
    if ev is None:
        return {"verdict": None, "reasons": [], "scores": None, "total": None, "targets": [], "t0": None,
                "t_hold_start": None, "t_hold_end": None, "hold_match_ratio": None, "debug": None}
    res, d = ev.data["result"], ev.data
    return {"verdict": res.verdict, "reasons": res.reasons, "scores": res.scores, "total": res.total,
            "targets": d["targets"], "t0": d["t0"], "t_hold_start": d["t_hold_start"],
            "t_hold_end": d["t_hold_end"], "hold_match_ratio": d["hold_match_ratio"], "debug": res.debug}


def hold_mismatches(smeta: dict, ev) -> list[str]:
    """t_hold_start / t_hold_end of the last replayed RESULT vs the sample meta; None on one side only differs too."""
    out = []
    for key in ("t_hold_start", "t_hold_end"):
        if key not in smeta:
            continue
        a, b = (ev.data[key] if ev is not None else None), smeta[key]
        if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-6):
            out.append(f"warning: {key} {a} != meta {b}")
    return out


def append_log(ev, sample_dir: Path | None) -> None:
    res = ev.data["result"]
    log = Path("logs") / "sessions.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "targets": ev.data["targets"], "verdict": res.verdict, "reasons": res.reasons, "scores": res.scores,
           "total": res.total, "debug": res.debug,
           "sample_dir": os.path.relpath(sample_dir) if sample_dir is not None else None}
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(row)) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.headless and not args.video:
        parser.error("--headless requires --video")
    if args.record and args.video:
        parser.error("--record cannot be used with --video")
    if args.record and Path(args.record).name not in RECORD_CLASSES:
        parser.error(f"--record: directory name must be one of {', '.join(RECORD_CLASSES)}")
    rec_meta = parse_meta(parser, args.meta)

    source = tracker = rec = None
    windowed = not args.headless
    try:
        cfg = load_config(args.config)
        gcfg = cfg["gesture"]
        sha = ensure_model(cfg["tracker"])
        source = VideoSource(args.video) if args.video else CameraSource(args.camera, cfg["camera"])
        smeta = source.meta if args.video and source.meta and "targets" in source.meta and "t0" in source.meta else None
        target = args.target
        if args.video and smeta is None and target is None:
            parser.error("--target is required for a raw video or a sample without targets/t0")
        if smeta is not None:
            if target is not None:
                print("warning: --target overrides meta targets", file=sys.stderr)
            else:
                target = smeta["targets"][0]
        tracker = HandTracker(cfg["tracker"], cfg["roi"], classify=lambda w, px: gesture.classify(w, px, gcfg))
        score_ms: list[float] = []

        def score_fn(seq):
            t_s = time.perf_counter()
            res = liveness.score(seq, cfg)
            score_ms.append((time.perf_counter() - t_s) * 1000)
            return res

        session = Session(cfg, score_fn=score_fn, target=target, seed=args.seed, external_roi=None)
        if args.record:
            rec = Recorder(Path(args.record), source.fps_nominal, {
                "camera_index": args.camera, "mediapipe_version": importlib.metadata.version("mediapipe"),
                "model_sha256": sha, "config_sha8": config_sha8(args.config),
                "challenge_len": cfg["session"]["challenge_len"], **rec_meta})
        if args.save_frames:
            Path(args.save_frames).mkdir(parents=True, exist_ok=True)
        if windowed:
            cv2.namedWindow(ui.WIN, cv2.WINDOW_AUTOSIZE)
        if args.video:
            session.key("space")
        debug = args.debug
        ts, procs, present_n, n = deque(maxlen=31), deque(maxlen=30), 0, 0
        all_proc: list[float] = []
        last_result = None
        try:
            for t, img in source:
                t_s = time.perf_counter()
                fr, roi = tracker.process(img, t)
                out = session.update(fr, roi)
                pose = gesture.analyze(fr.lm_world, fr.lm_px, gcfg) if fr.present else None
                proc_ms = (time.perf_counter() - t_s) * 1000
                all_proc.append(proc_ms)
                procs.append(proc_ms)
                ts.append(fr.t)
                dt = np.diff(list(ts))
                fps = 1.0 / statistics.median(dt) if len(dt) and statistics.median(dt) > 0 else 0.0
                n += 1
                present_n += int(fr.present)
                kinds = [ev.kind for ev in out.events]
                for ev in out.events:
                    if ev.kind == "ROUND_START":
                        tracker.reset()
                        if rec:
                            rec.start_round(fr.t, fr.frame_wh)
                    elif ev.kind == "ROUND_DISCARD" and rec:
                        rec.discard_round()
                if rec and out.in_round:
                    rec.add(fr, img)
                for ev in out.events:
                    if ev.kind != "RESULT":
                        continue
                    last_result = ev
                    sample_dir = None
                    if rec:
                        seq = ev.data["sequence"]
                        res = ev.data["result"]
                        sample_dir = rec.finish(
                            seq.frames if seq else rec.frames, seq.roi if seq else None, seq.roi_t if seq else None,
                            {"targets": ev.data["targets"], "t0": ev.data["t0"],
                             "t_hold_start": ev.data["t_hold_start"], "t_hold_end": ev.data["t_hold_end"],
                             "hold_match_ratio": ev.data["hold_match_ratio"],
                             "live": {"verdict": res.verdict, "total": res.total, "scores": res.scores,
                                      "reasons": res.reasons, "debug": res.debug}})
                    if not args.video:
                        append_log(ev, sample_dir)
                if "REVEAL" in kinds and smeta is not None:
                    session.set_t0(smeta["t0"])
                if args.headless:
                    print(f"{fr.t:.3f} {int(fr.present)} {fr.raw_label} {out.smooth_label} {out.phase} "
                          f"{out.target or '-'}", flush=True)
                if windowed or args.save_frames:
                    canvas = ui.draw(img, fr, pose, out, {"fps": fps, "proc_ms": statistics.median(procs)},
                                     debug, cfg)
                    if args.save_frames:
                        cv2.imwrite(str(Path(args.save_frames) / f"{n - 1:06d}.png"), canvas)
                if windowed:
                    if n == 1:
                        name = f"cam {args.camera}" if not args.video else Path(args.video).name
                        w, h = source.frame_wh
                        cv2.setWindowTitle(ui.WIN, f"HandLiveness  {name}  {w}x{h}")
                    cv2.imshow(ui.WIN, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if "REVEAL" in kinds and not args.video:
                        session.set_t0(time.monotonic())  # the reveal is on screen once waitKey returns (3.1)
                    if key == ord("q"):
                        break
                    if key == ord(" "):
                        session.key("space")
                    elif key == ord("r"):
                        session.key("r")
                    elif key == ord("x") and rec and session.state == RESULT and rec.last is not None:
                        rec.mark_discard()
                    elif key == ord("d"):
                        debug = not debug
                    if cv2.getWindowProperty(ui.WIN, cv2.WND_PROP_VISIBLE) < 1:
                        break
                if args.max_frames is not None and n >= args.max_frames:
                    break
        except KeyboardInterrupt:
            pass
        if smeta is not None:
            for w in hold_mismatches(smeta, last_result):
                print(w, file=sys.stderr)
        if args.headless or args.bench:
            j = round_summary(last_result) | {"frames": n, "present_frames": present_n, "final_state": session.state}
            if args.bench:
                p50, p95 = (np.percentile(all_proc, [50, 95]) if all_proc else (0.0, 0.0))
                j |= {"proc_ms_p50": round(float(p50), 2), "proc_ms_p95": round(float(p95), 2),
                      "score_ms": round(max(score_ms), 2) if score_ms else None}
            print(json.dumps(_jsonable(j)), flush=True)
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
