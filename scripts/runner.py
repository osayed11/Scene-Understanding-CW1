"""Launch one pySLAM run and collect its outputs.

TODO (Part 1): fill in once the tagged pySLAM version is known.
  - check `pixi run slam --help` in third_party/pyslam for the exact flags
  - make pySLAM save the trajectory in TUM format (config.yaml / SAVE_TRAJECTORY or CLI flag)
  - copy it to results/trajectories/<run_id>.txt
  - parse wall-clock time per frame from the log into results/logs/<run_id>.log
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from config import PYSLAM, RESULTS, run_id


def run_pyslam(sequence: dict, method: dict, settings: dict, run: int, dry_run: bool = False) -> Path:
    rid = run_id(sequence["name"], method["name"], settings, run)
    traj_out = RESULTS / "trajectories" / f"{rid}.txt"
    log_out = RESULTS / "logs" / f"{rid}.log"

    cmd = ["pixi", "run", "slam"]  # TODO: add dataset/settings/mode flags per `pixi run slam --help`
    cmd += ["--settings", sequence["settings_file"]]
    if method.get("mode") in ("rgbd", "pseudo_rgbd"):
        cmd += ["--depth-dir", sequence[method["depth_source"]]]  # TODO: real flag name
    for k, v in settings.items():
        cmd += [f"--{k}", str(v)]

    print(" ".join(cmd))
    if dry_run:
        return traj_out
    t0 = time.time()
    with open(log_out, "w") as log:
        subprocess.run(cmd, cwd=PYSLAM, stdout=log, stderr=subprocess.STDOUT, check=True)
    elapsed = time.time() - t0
    with open(log_out, "a") as log:
        log.write(f"\n[cw1] wall_seconds={elapsed:.3f}\n")
    # TODO: move pySLAM's trajectory output to traj_out
    return traj_out
