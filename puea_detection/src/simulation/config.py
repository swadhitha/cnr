"""
Configuration loading and path helpers for the adaptive-PUEA experiments.

All experimental constants live in ``configs/experiments.json``. This module
only loads that file, fingerprints it (so stale simulation data can be
detected), and resolves the directories used by the new experiments. It does
not touch the original baseline data or models.
"""

from __future__ import annotations

import hashlib
import json
import os
import zlib

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(SRC_DIR)
DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "experiments.json")


def load_config(path: str | None = None) -> dict:
    """Load the experiment configuration (defaults to configs/experiments.json)."""
    with open(path or DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def config_hash(cfg: dict) -> str:
    """Stable short fingerprint of a config, ignoring documentation keys."""

    def strip(obj):
        if isinstance(obj, dict):
            return {k: strip(v) for k, v in obj.items() if not k.startswith("_")}
        if isinstance(obj, list):
            return [strip(v) for v in obj]
        return obj

    payload = json.dumps(strip(cfg), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


#: Config blocks that determine the simulated data. Changing anything else
#: (detector, calibration or evaluation settings) does not require
#: regenerating data/simulation/.
SIMULATION_KEYS = ("seeds", "regimes", "geometry", "channel", "pu", "attacker", "streams", "scenarios")


def simulation_hash(cfg: dict) -> str:
    """Fingerprint of the data-generating part of the config only."""
    return config_hash({k: cfg[k] for k in SIMULATION_KEYS if k in cfg})


def regime_names(cfg: dict) -> list:
    return [k for k in cfg.get("regimes", {"default": {}}) if not k.startswith("_")]


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def regime_config(cfg: dict, regime: str) -> dict:
    """Config with the named regime's overrides deep-merged in."""
    return _deep_merge(cfg, cfg.get("regimes", {}).get(regime, {}))


def stream_key(name: str) -> int:
    """Deterministic integer for a stream name (used to derive RNG seeds)."""
    return zlib.crc32(name.encode("utf-8"))


def experiment_paths() -> dict:
    """
    Directories for the adaptive experiments, kept separate from the
    original pipeline's data/, models/ and reports/ files.
    """
    paths = {
        "root": ROOT_DIR,
        "config": DEFAULT_CONFIG_PATH,
        "sim_data": os.path.join(ROOT_DIR, "data", "simulation"),
        "sim_models": os.path.join(ROOT_DIR, "models", "simulation"),
        "adaptive_reports": os.path.join(ROOT_DIR, "reports", "adaptive"),
        "traces": os.path.join(ROOT_DIR, "reports", "adaptive", "traces"),
        "figures": os.path.join(ROOT_DIR, "reports", "figures"),
    }
    for key, path in paths.items():
        if key not in ("root", "config"):
            os.makedirs(path, exist_ok=True)
    return paths
