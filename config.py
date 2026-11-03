"""Strict config loader (PROMPT.md 5.6): every key required, types checked, no hidden defaults.

Code reads cfg["section"]["key"] directly; cfg.get(key, default) is forbidden.
SCHEMA grows with each milestone; a key missing from config.yaml or unknown to SCHEMA is an error.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


class ConfigError(ValueError):
    pass


NUM = (int, float)
PAIR = "pair"  # list of two numbers, lo < hi

SCHEMA = {
    "camera": {"width": int, "height": int, "max_read_failures": int, "fps_nominal": NUM},
    "tracker": {
        "model_path": str, "model_url": str, "running_mode": str, "num_hands": int,
        "min_hand_detection_confidence": NUM, "min_hand_presence_confidence": NUM,
        "min_tracking_confidence": NUM,
    },
    "gesture": {
        "pip_extended_deg": NUM, "thumb_ratio": NUM, "thumb_cross_cos": NUM, "ok_tip_ratio": NUM,
        "ext_pip_deg": PAIR, "ext_thumb_ratio": PAIR, "smooth_window_s": NUM,
    },
    "roi": {"size": int, "sat_level": int},
}


def _check(node, schema: dict, path: str) -> None:
    where = path or "<root>"
    if not isinstance(node, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(node).__name__}")
    missing, extra = schema.keys() - node.keys(), node.keys() - schema.keys()
    if missing:
        raise ConfigError(f"{where}: missing keys {sorted(missing)}")
    if extra:
        raise ConfigError(f"{where}: unknown keys {sorted(extra)}")
    for key, typ in schema.items():
        val, p = node[key], f"{path}.{key}" if path else key
        if isinstance(typ, dict):
            _check(val, typ, p)
        elif typ == PAIR:
            ok = (isinstance(val, list) and len(val) == 2
                  and all(isinstance(v, NUM) and not isinstance(v, bool) for v in val) and val[0] < val[1])
            if not ok:
                raise ConfigError(f"{p}: expected [lo, hi] numbers with lo < hi, got {val!r}")
        elif isinstance(val, bool) or not isinstance(val, typ):
            raise ConfigError(f"{p}: expected {typ}, got {type(val).__name__} {val!r}")


def validate(cfg: dict) -> dict:
    _check(cfg, SCHEMA, "")
    if cfg["tracker"]["running_mode"] != "VIDEO":
        raise ConfigError("tracker.running_mode: only VIDEO is supported (PROMPT.md 2)")
    if cfg["tracker"]["num_hands"] < 1:
        raise ConfigError("tracker.num_hands: must be >= 1")
    return cfg


def load_config(path="config.yaml") -> dict:
    try:
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"cannot read {path}: {e}") from e
    return validate(cfg)


def config_sha8(path="config.yaml") -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:8]
