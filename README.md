# COMP0227 CW1 — Monocular Depth in Monocular SLAM

Question: **when does learned depth help a monocular SLAM system?**
Deadline: Friday 13 November 2026, 16:00 (Moodle). 
## Layout

```
scripts/          shared modules (trajectory.py, metrics.py, config.py, runner.py, depth.py)
                  + one script per task, grouped by part; measure_run.py is THE measuring script
configs/          sequences/*.yaml (paths, settings file, GT) and methods/*.yaml (mode, trust params)
pyslam.diff       our changes to pySLAM against the tagged commit (made by scripts/setup/export_pyslam_diff.sh; submitted)
results/          results.csv (one row per run), trajectories/ (TUM format, per run), tables/
report/           IEEE conference paper (main.tex), 5 pages max
third_party/      pySLAM clone (git-ignored, NOT submitted)
data/ depth_maps/ sequences and predicted depth (git-ignored, NOT submitted)
```

## Setup

```bash
export PYSLAM_TAG=<tag named in the coursework release>
make setup                       # clones pySLAM, installs default + depth pixi envs (~7 GB weights)
pip install numpy scipy pandas pyyaml opencv-python-headless matplotlib tqdm pytest
bash scripts/setup/download_tum.sh
python scripts/setup/prepare_kitti06.py
python scripts/setup/prepare_replica_room0.py
make test
```

## Which command makes which table

| Report item | Command | Output |
|---|---|---|
| Table 1: baseline, C++ vs Python core, determinism, RGB-D reference | `make table1` | `results/tables/table1_*.csv` |
| Depth maps for every frame (reused by all later parts) | `make depth_maps` | `depth_maps/<seq>/` |
| Table 2: depth quality, temporal consistency, keypoint depth + uncertainty + scale | `make table2` | `results/tables/table2_*.csv` |
| Table 3: pseudo RGB-D, method A, method B, trust sweeps | `make table3` | `results/tables/table3_*.csv` |
| Table 4: agreement, frame choice, failure case, own sequence | `make table4` | `results/tables/table4_*.csv` |
| Submission zip (code, results, report) | `make submission` | `submission/` |

Each run appends one row to `results/results.csv` via `scripts/measure_run.py` and saves its trajectory to
`results/trajectories/<run_id>.txt` (TUM format), so every number can be recomputed with:

```bash
python scripts/measure_run.py --sequence tum_fr3 --method baseline --run 1
```

## What is measured (Section 10 of the brief)

| Column | Definition |
|---|---|
| `ate_noscale` | RMSE of position after SE(3) alignment (Umeyama, scale fixed at 1) |
| `ate_scale`, `scale` | RMSE after Sim(3) alignment and the scale that was fitted |
| `rpe_trans`, `rpe_rot_deg` | RMSE of relative translation (m) / rotation (deg) over `rpe_delta` frames, estimate pre-scaled by `scale` |
| `tracked`, `lost_spans` | share of GT frames with a pose; time intervals where tracking was lost |
| `time_per_frame` | wall-clock seconds per frame (machine: TODO) |

Estimates are associated to GT by nearest timestamp within `--max-dt` (default 0.02 s).

## Methods

| Name | Mode | Where depth enters | Starting point or constraint | Which depth wins | Map in metres? |
|---|---|---|---|---|---|
| baseline | mono | — | — | triangulation | no |
| rgbd_gt | RGB-D, sensor depth | pySLAM RGB-D pipeline | constraint | sensor | yes |
| pseudo_rgbd | RGB-D, predicted depth | pySLAM RGB-D pipeline, unchanged | constraint | network | yes (if network metric) |
| method_a | TODO | TODO | TODO | TODO | TODO |
| method_b | TODO | TODO | TODO | TODO | TODO |

## Use of generative AI
Declared in the report (UCL Category 3). TODO: keep a running list here of where AI tools were used.

## Group / contributions
TODO
