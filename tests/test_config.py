import copy
import re

import pytest

from config import PAIR, SCHEMA, ConfigError, load_config, validate


def _leaves(schema, path=()):
    for k, v in schema.items():
        if isinstance(v, dict):
            yield from _leaves(v, path + (k,))
        else:
            yield path + (k,), v


def _walk(cfg, path):
    for k in path[:-1]:
        cfg = cfg[k]
    return cfg


@pytest.fixture
def cfg():
    return load_config("config.yaml")


def test_default_passes(cfg):
    validate(cfg)


def test_missing_leaf(cfg):
    for path, _ in _leaves(SCHEMA):
        c = copy.deepcopy(cfg)
        del _walk(c, path)[path[-1]]
        with pytest.raises(ConfigError, match=path[-1]):
            validate(c)


def test_bad_type(cfg):
    for path, typ in _leaves(SCHEMA):
        if typ is PAIR or typ is str:
            continue
        for bad in (True, "1"):
            c = copy.deepcopy(cfg)
            _walk(c, path)[path[-1]] = bad
            with pytest.raises(ConfigError):
                validate(c)


@pytest.mark.parametrize("bad", [[170, 90], [1, 2, 3], [1, 1], "x", [True, 2]])
def test_bad_pair(cfg, bad):
    cfg["gesture"]["ext_pip_deg"] = bad
    with pytest.raises(ConfigError):
        validate(cfg)


def test_unknown_key(cfg):
    cfg["roi"]["extra"] = 1
    with pytest.raises(ConfigError):
        validate(cfg)


def test_running_mode(cfg):
    cfg["tracker"]["running_mode"] = "IMAGE"
    with pytest.raises(ConfigError):
        validate(cfg)


def test_challenge_len(cfg):
    cfg["session"]["challenge_len"] = 2
    with pytest.raises(ConfigError, match="challenge_len"):
        validate(cfg)


def test_weights_sum(cfg):
    cfg["fusion"]["w_texture"] = 0.3
    with pytest.raises(ConfigError, match="weights"):
        validate(cfg)


def test_durations_positive(cfg):
    cfg["session"]["hold_s"] = 0
    with pytest.raises(ConfigError, match="hold_s"):
        validate(cfg)


def test_match_frames_equal(cfg):
    cfg["temporal"]["match_frames"] += 1
    with pytest.raises(ConfigError, match="match_frames"):
        validate(cfg)


@pytest.mark.parametrize("lo, hi", [(0, 10), (11, 10)])
def test_min_max_pairs(cfg, lo, hi):
    cfg["planarity"].update(min_pairs=lo, max_pairs=hi)
    with pytest.raises(ConfigError, match="min_pairs"):
        validate(cfg)


@pytest.mark.parametrize("sec", ["nonrigid", "planarity", "texture"])
def test_sigmoid_scale_positive(cfg, sec):
    cfg[sec]["s"] = 0
    with pytest.raises(ConfigError, match=rf"{sec}\.s"):
        validate(cfg)


@pytest.mark.parametrize("sec, key", [("texture", "min_blocks"), ("planarity", "ransac_iters")])
def test_at_least_one(cfg, sec, key):
    cfg[sec][key] = 0
    with pytest.raises(ConfigError, match=rf"{sec}\.{key}"):
        validate(cfg)


@pytest.mark.parametrize("sec, key, bad", [("fusion", "w_nonrigid", float("nan")), ("planarity", "s", float("nan")),
                                           ("session", "hold_s", float("nan")), ("session", "hold_s", float("inf"))])
def test_non_finite(cfg, sec, key, bad):
    cfg[sec][key] = bad
    with pytest.raises(ConfigError, match=rf"{sec}\.{key}"):
        validate(cfg)


@pytest.mark.parametrize("bad", [[0, float("inf")], [float("-inf"), 1]])
def test_non_finite_pair(cfg, bad):
    cfg["texture"]["band"] = bad
    with pytest.raises(ConfigError, match=r"texture\.band"):
        validate(cfg)


def test_yaml_nan_rejected(tmp_path):
    src = open("config.yaml", encoding="utf-8").read()
    text, n = re.subn(r"(?m)^(\s+hold_s:)\s*[^\s#]+", r"\1 .nan", src)
    assert n == 1
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=r"session\.hold_s"):
        load_config(p)


@pytest.mark.parametrize("sec,key,val", [("session", "hold_s", 10 ** 400), ("fusion", "w_texture", 10 ** 400),
                                         ("texture", "band", [0, 10 ** 400])], ids=["hold_s", "w_texture", "band"])
def test_huge_yaml_int_is_a_config_error(cfg, sec, key, val):
    cfg[sec][key] = val   # beyond float range: isfinite() and the weight sum would raise OverflowError
    with pytest.raises(ConfigError, match=rf"{sec}\.{key}"):
        validate(cfg)


def test_huge_int_weights_sum_is_a_config_error(cfg):
    cfg["fusion"]["w_planarity"] = cfg["fusion"]["w_texture"] = 10 ** 308   # each within float range, the sum is not
    with pytest.raises(ConfigError, match="fusion: weights"):
        validate(cfg)


@pytest.mark.parametrize("raw", [b"a: \xff\n", b"a: " + b"1" * 5000 + b"\n"], ids=["not_utf8", "int_over_4300_digits"])
def test_unreadable_config_text_is_a_config_error(tmp_path, raw):
    p = tmp_path / "c.yaml"
    p.write_bytes(raw)
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(p)
