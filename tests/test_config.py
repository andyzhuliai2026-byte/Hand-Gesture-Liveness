import copy

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
