"""YAML config loading with deep-merge onto defaults and dotted overrides."""
from __future__ import annotations

import copy
import logging
from pathlib import Path
import typing as tp

import yaml

__all__ = ["load_config", "deep_merge", "apply_override"]

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_override(cfg: dict, spec: str) -> None:
    """Apply one ``a.b.c=value`` override in place (YAML-typed value).

    ``'a.b=3'`` sets an int, ``'x=true'`` a bool, ``'y=[1,2]'`` a list;
    intermediate dicts are created as needed.
    """
    if "=" not in spec:
        raise ValueError(f"override {spec!r} must look like key.path=value")
    key, raw = spec.split("=", 1)
    value = yaml.safe_load(raw) if raw != "" else None
    node = cfg
    parts = key.strip().split(".")
    for p in parts[:-1]:
        nxt = node.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            node[p] = nxt
        node = nxt
    node[parts[-1]] = value


def load_config(path: tp.Optional[tp.Union[str, Path]] = None,
                overrides: tp.Optional[tp.Sequence[str]] = None) -> dict:
    """Load a config: defaults <- file <- CLI overrides.

    Parameters
    ----------
    path : path, optional
        YAML file merged onto ``configs/default.yaml``; None loads the
        defaults only (or the file itself when IT IS the defaults file).
    overrides : sequence of str, optional
        ``key.path=value`` specs applied last.
    """
    cfg: dict = {}
    if DEFAULT_CONFIG.is_file():
        cfg = yaml.safe_load(DEFAULT_CONFIG.read_text()) or {}
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        loaded = yaml.safe_load(Path(path).read_text()) or {}
        cfg = deep_merge(cfg, loaded)
    for spec in overrides or []:
        apply_override(cfg, spec)
    return cfg
