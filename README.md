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
bash scripts/setup/clone_pyslam.sh   # clones sjulier/pyslam at comp0227-2026-lab03.1 (PYSLAM_TAG), builds it, installs depth weights (~7 GB)
pip install numpy scipy pandas pyyaml opencv-python-headless matplotlib tqdm pytest
bash scripts/setup/download_data.sh   # about 14 GB download
python scripts/setup/prepare_kitti06.py
python scripts/setup/prepare_replica_room0.py
make test
```

## Which command makes which table

| Report item | Command | Output |
|---|---|---|
| Table 1: baseline, determinism (and C++ vs Python core: methods baseline / baseline_py), RGB-D reference | `make table1` | `results/tables/table1_summary.csv` |
| Table 1b: agreement of the map points with the sequence's depth (TUM, Replica), by parallax and distance | `python scripts/part1/map_depth_agreement.py` | `results/tables/table1_map_depth*.csv` |
| Depth maps for every frame (reused by all later parts) | `make depth_maps` | `depth_maps/<seq>/` |
| Table 2: depth quality, temporal consistency, keypoint depth + uncertainty + scale | `make table2` | `results/tables/table2_*.csv` |
| Table 3: pseudo RGB-D, method A, method B, trust sweeps | `make table3` | `results/tables/table3_*.csv` |
| Table 4: agreement, frame choice, failure case, own sequence | `make table4` | `results/tables/table4_*.csv` |
| Submission zip (code, results, report) | `make submission` | `submission/` |

Each run appends one row to `results/results.csv` via `scripts/measure_run.py` and saves its trajectory to
`results/trajectories/<run_id>.txt` (TUM format); pySLAM's own log, config and plots are in `results/logs/<run_id>/`, and its final map,
compacted by `scripts/extract_map.py` to `results/logs/<run_id>/map/map_points.npz` (keyframes, points, observations, per-keyframe depth, parallax).
Saving the map in a headless run is our one change to pySLAM so far (`--save-map DIR` in main_slam.py, see `pyslam.diff`).
Runs must not be launched in parallel: pySLAM writes its logs to one folder, timings would be shared, and `measure_run.py` read-modify-writes `results.csv`.
Every number can be recomputed with:

```bash
python scripts/measure_run.py --sequence tum_fr3 --method baseline --run 1
```

## What is measured (Section 10 of the brief)

| Column | Definition |
|---|---|
| `ate_noscale` | RMSE of position after SE(3) alignment (Umeyama, scale fixed at 1) |
| `ate_scale`, `scale` | RMSE after Sim(3) alignment and the scale that was fitted |
| `rpe_trans`, `rpe_rot_deg` | RMSE of relative translation (m) / rotation (deg) over `rpe_delta` frames, estimate pre-scaled by `scale` |
| `tracked`, `lost_spans` | poses produced / frames in the sequence; time intervals with no pose for > 3 frame intervals |
| `time_per_frame`, `wall_per_frame` | mean per-frame time of pySLAM's tracking thread (from its tracking.log); wall-clock seconds of the run / frames (with `speed=1` bounded by the playback), headless (machine: MacBook, Apple M4, 10 cores, 24 GB, macOS 26; pySLAM C++ core unless stated) |

Estimates are associated to GT by nearest timestamp within `--max-dt` (default 0.02 s).

## Methods

| Name | Mode | Where depth enters | Starting point or constraint | Which depth wins | Map in metres? |
|---|---|---|---|---|---|
| baseline | mono | — | — | triangulation | no |
| rgbd_gt | RGB-D, sensor depth | pySLAM RGB-D pipeline | constraint | sensor | yes |
| pseudo_rgbd | RGB-D, predicted depth | pySLAM RGB-D pipeline, unchanged | constraint | network | yes (if network metric) |
| method_a | TODO | TODO | TODO | TODO | TODO |
| method_b | TODO | TODO | TODO | TODO | TODO |

## Part 2 workflow

The offline tools read `data/` or, if it is empty, the existing sibling `../data/`.
Use `--data-root /absolute/path/to/data` or `CW1_DATA_ROOT` to override this. They
do not require the native SLAM build. Install the project dependencies and pytest
in a Python environment before running them.

```bash
python scripts/part2/predict_depth.py --all --dry-run
```

This checks all RGB inputs and calibration without importing a depth network.
TUM retains every RGB frame and pairs GT depth by nearest timestamp within 20 ms;
some RGB frames have no depth, and a depth image can be reused. The association
offset is recorded in depth evaluation outputs.

The current inference adapter uses **one fixed Depth Pro checkpoint** on all
three sequences. It follows the [official metric inference API](https://github.com/apple-aiml-research/ml-depth-pro),
passes RGB images and the calibrated focal length, and never receives GT depth
or poses. Start with a CPU timing sample before a full prediction run. From the
course pySLAM checkout, after installing its depth models:

```bash
pixi run -e depth-cpu python ../../scripts/part2/predict_depth.py \
  --pyslam-root . --sequence replica_room0 --max-frames 3
```

Use `-e depth` on a machine with an NVIDIA GPU. The default checkpoint path is
`thirdparty/ml_depth_pro/checkpoints/depth_pro.pt` under `--pyslam-root`; override
it with `--checkpoint`. The adapter does not download weights implicitly. A full
run uses `--all` without `--max-frames`. Actual model inference has not yet been
benchmarked on this WSL machine.

Each frame produces `depth_maps/<sequence>/<rgb_stem>.npy` in float32 metres plus
a PNG with the sequence's configured depth factor. Decimal timestamp stems are
preserved. Predictions outside the PNG range become invalid PNG pixels by the
explicit default `--png-overflow invalid`; their full values remain in NPY and
are used in Part 2 evaluation. `--png-overflow raise` instead rejects overflow.
`manifest.json` records the checkpoint SHA256, calibration, per-frame inference
time, invalid fraction and PNG overflow fraction. RGB content hashes and
timestamps are checked before cached predictions are reused. Matching caches resume;
changing the model or encoding requires a separate `--pred-root`. The generated
`associations.txt` links the selected original RGB frames to predicted PNGs,
with paths relative to the cache's sequence directory (the TUM dataset base).
Part 3 still
needs to connect these to its runner and use matching `DepthMapFactor` settings.

After the three-frame Replica prediction sample:

```bash
python scripts/part2/eval_depth.py --sequence replica_room0 --max-frames 3
python scripts/part2/keypoint_depth.py --sequence replica_room0 --max-frames 3
python scripts/part2/temporal_consistency.py --sequence replica_room0 --max-pairs 2
```

For full caches, `make table2` runs the three analyses. The outputs are:

| Output | Definition |
|---|---|
| `table2_depth.csv`, `table2_depth_frames.csv` | Raw and per-frame median-scaled AbsRel, RMSE and delta < 1.25. Summary errors are means over scored frames; valid pixel coverage and missing-frame counts are retained. KITTI explicitly has no GT depth score. Median scaling uses GT only as an evaluation diagnostic. |
| `table2_temporal.csv` | Depth disagreement at motion-compensated surface correspondences using GT poses. Source GT depth controls correspondence/visibility where available, independently of the predictions. Visibility also uses a source z-buffer and target GT depth; without GT, prediction-derived geometry/visibility is explicitly labelled. Rejection counts and depth timestamp offsets are recorded. |
| `table2_keypoints.csv`, `results/keypoints/<sequence>/*.npz` | Depth and local-patch heuristic uncertainty at OpenCV ORB diagnostic keypoints, plus GT errors where available. These are not claimed to be keypoints from the saved pySLAM runs. |

The keypoint tools also accept `--map-samples samples.npz` for **one sequence**,
containing corresponding `learned_depth_m` and `triangulated_depth_map` arrays,
with optional `sigma_m` and `parallax_deg`. Robust ratios estimate map units per
metre using map observations only. Without these samples, map scale is explicitly
unavailable. Exporting actual pySLAM keypoint/map observations and investigating
spatial error patterns remain Part 2 work; trajectories alone cannot supply them.

The model adapter, sampling helpers and analysis scripts are covered by synthetic
geometry, invalid-data, known-scale and cache-resumption tests. Those checks do
not substitute for a real-network inference run or the full sequence experiments.

## Use of generative AI
Declared in the report (UCL Category 3). Running development log:

- 24 October 2026: Codex read the brief, reviewed the Part 1 scripts and saved
  results, and helped implement Part 2 dataset loading, the Depth Pro inference
  adapter, depth/temporal/keypoint evaluation, map-scale helpers and regression
  tests. Real network predictions and scientific conclusions have not yet been
  produced by this Part 2 implementation.

## Group / contributions
TODO
