"""
Step 1 of the adaptive-PUEA experiments: generate temporal simulation data.

Writes, for every seed in configs/experiments.json:

    data/simulation/<regime>/seed_<k>/geometry.csv        SU positions + static noise offsets
    data/simulation/<regime>/seed_<k>/xgb_train.csv.gz    stable PU + generic random attackers
    data/simulation/<regime>/seed_<k>/calibration.csv.gz  PU-only stable stream
    data/simulation/<regime>/seed_<k>/test_<scenario>.csv.gz   every open-loop scenario
    data/simulation/manifest.json                         simulation-settings hash + file list

Closed-loop scenarios (poisoning) depend on the detector's decisions, so they
are realised during the experiment (run_adaptive_experiment.py) and saved
per system under data/simulation/<regime>/seed_<k>/realized/.

The original dataset under data/*.csv is never read or written.

Usage (from the repository root):
    python puea_detection/src/run_simulation.py [--config path] [--seeds 0 1] [--regimes default]
"""

from __future__ import annotations

import argparse
import json
import os
import time

from simulation import (
    CRNSimulator,
    ScenarioStream,
    simulation_hash,
    experiment_paths,
    generate_calibration_stream,
    generate_xgb_training_stream,
    load_config,
    regime_config,
    regime_names,
)


def seed_dir(paths: dict, regime: str, seed: int) -> str:
    d = os.path.join(paths["sim_data"], regime, f"seed_{seed}")
    os.makedirs(d, exist_ok=True)
    return d


def generate_seed(cfg: dict, regime: str, seed: int, paths: dict) -> list:
    sim = CRNSimulator(cfg, seed)
    out = seed_dir(paths, regime, seed)
    written = []

    sim.geometry.to_frame().to_csv(os.path.join(out, "geometry.csv"), index=False)
    written.append("geometry.csv")

    generate_xgb_training_stream(sim).save(os.path.join(out, "xgb_train.csv.gz"))
    written.append("xgb_train.csv.gz")
    generate_calibration_stream(sim).save(os.path.join(out, "calibration.csv.gz"))
    written.append("calibration.csv.gz")

    n = cfg["streams"]["n_slots"]
    for name, scfg in cfg["scenarios"].items():
        stream = ScenarioStream(sim, name, scfg, n)
        if stream.closed_loop:
            continue
        fname = f"test_{name}.csv.gz"
        stream.generate().save(os.path.join(out, fname))
        written.append(fname)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--regimes", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = experiment_paths()
    seeds = args.seeds if args.seeds else cfg["seeds"]
    regimes = args.regimes if args.regimes else regime_names(cfg)
    manifest = {"simulation_hash": simulation_hash(cfg), "regimes": {}}
    for regime in regimes:
        rcfg = regime_config(cfg, regime)
        manifest["regimes"][regime] = {}
        for seed in seeds:
            t0 = time.time()
            files = generate_seed(rcfg, regime, seed, paths)
            manifest["regimes"][regime][str(seed)] = files
            print(f"[run_simulation] {regime} seed {seed}: {len(files)} files in {time.time() - t0:.1f}s")

    with open(os.path.join(paths["sim_data"], "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[run_simulation] data written to {paths['sim_data']} (simulation hash {manifest['simulation_hash']})")


if __name__ == "__main__":
    main()
