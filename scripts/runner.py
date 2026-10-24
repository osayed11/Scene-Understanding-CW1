"""Launch one pySLAM run and collect its outputs.

How pySLAM (sjulier fork, comp0227-2026) is driven:
  KITTI 06 : pixi run slam --video <mp4> --settings settings/KITTI04-12.yaml --groundtruth groundtruth.txt --timestamps times.txt
  TUM      : pixi run slam --tum <folder> [--sensor mono|rgbd] [--associations associations.txt]
  Replica  : no CLI option: a config file with DATASET.type REPLICA_DATASET (pixi run slam -c <config>)
Every run gets its own config file (-c) = pySLAM's config.yaml with SAVE_TRAJECTORY set to TUM format and
an absolute output folder, so the trajectory lands in results/logs/<run_id>/trajectory_final.txt and is
copied to results/trajectories/<run_id>.txt. The C++ / Python core is chosen with PYSLAM_USE_CPP.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from config import PYSLAM, RESULTS, ROOT, run_id


def _write_run_config(sequence: dict, method: dict, out_dir: Path) -> Path:
    cfg = yaml.safe_load((PYSLAM / "config.yaml").read_text())
    cfg["SAVE_TRAJECTORY"] = {"save_trajectory": True, "format_type": "tum",
                              "output_folder": str(out_dir), "basename": "trajectory"}
    if sequence["name"].startswith("replica"):
        cfg["DATASET"]["type"] = "REPLICA_DATASET"
        cfg["REPLICA_DATASET"] = {"type": "replica", "sensor_type": method.get("sensor", "mono"),
                                  "base_path": str(ROOT / sequence["root"]).rsplit("/", 1)[0],
                                  "name": Path(sequence["root"]).name,
                                  "settings": sequence["settings_file"], "groundtruth_file": "auto"}
    for k, v in (method.get("global_parameters") or {}).items():
        cfg.setdefault("GLOBAL_PARAMETERS", {})
        cfg["GLOBAL_PARAMETERS"] = cfg["GLOBAL_PARAMETERS"] or {}
        cfg["GLOBAL_PARAMETERS"][k] = v
    path = out_dir / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def build_command(sequence: dict, method: dict, settings: dict, cfg_path: Path) -> list[str]:
    cmd = ["pixi", "run", "slam", "-c", str(cfg_path), "--headless", "--no_output_date",
           "--save-map", str(cfg_path.parent / "map")]  # our pySLAM patch: the final map, for the point analyses
    if "speed" not in settings:  # playback speed: 1 = the camera's frame rate (the fair condition), 0 = as fast as possible
        cmd += ["--speed", str(method.get("speed", 1))]
    name = sequence["name"]
    if name.startswith("kitti"):
        cmd += ["--video", str(ROOT / sequence["video"]), "--settings", sequence["settings_file"],
                "--groundtruth", "groundtruth.txt", "--timestamps", "times.txt"]
    elif name.startswith("tum") or name == "own":
        cmd += ["--tum", str(ROOT / sequence["root"]), "--settings", sequence["settings_file"],
                "--sensor", method.get("sensor", "mono")]
        if method.get("associations"):
            cmd += ["--associations", method["associations"]]
    # replica: everything is in the config file
    for k, v in settings.items():
        cmd += [f"--{k}", str(v)]  # method-specific CLI options added to pySLAM (Part 3)
    return cmd


def run_pyslam(sequence: dict, method: dict, settings: dict, run: int, dry_run: bool = False) -> Path:
    rid = run_id(sequence["name"], method["name"], settings, run)
    out_dir = RESULTS / "logs" / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_out = RESULTS / "trajectories" / f"{rid}.txt"
    cfg_path = _write_run_config(sequence, method, out_dir)
    cmd = build_command(sequence, method, settings, cfg_path)
    env = {**os.environ, "PYSLAM_USE_CPP": "true" if method.get("core", "cpp") == "cpp" else "false"}

    print(f"[{rid}] PYSLAM_USE_CPP={env['PYSLAM_USE_CPP']} " + " ".join(cmd))
    if dry_run:
        return traj_out
    t0 = time.time()
    with open(out_dir / "stdout.log", "w") as log:
        proc = subprocess.run(cmd, cwd=PYSLAM, stdout=log, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - t0
    with open(out_dir / "stdout.log", "a") as log:
        log.write(f"\n[cw1] exit={proc.returncode} wall_seconds={elapsed:.3f}\n")
    final = out_dir / "trajectory_final.txt"
    if proc.returncode != 0 or not final.exists():
        raise RuntimeError(f"[{rid}] pySLAM failed (exit {proc.returncode}); see {out_dir/'stdout.log'}")
    shutil.copy(final, traj_out)
    # pySLAM's per-thread logs (tracking.log has the per-frame tracking time) are overwritten by the next run: keep them
    logs_dir = out_dir / "pyslam_logs"
    logs_dir.mkdir(exist_ok=True)
    for f in (PYSLAM / "logs").glob("*.log"):
        shutil.copy(f, logs_dir / f.name)
    # compact the saved map (~300 MB json) into map_points.npz and drop the heavy files
    if (out_dir / "map" / "map.json").exists():
        from extract_map import extract
        extract(out_dir / "map")
    return traj_out
