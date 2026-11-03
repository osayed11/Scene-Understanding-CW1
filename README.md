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
| Table 4a: triangulated vs learned depth of the map points (all sequences), map scale from the learned depth alone, by parallax / distance / image position | `python scripts/part4/map_vs_learned.py` | `results/tables/table4_agreement*.csv`; per observation `results/logs/<run_id>/map/obs_learned.csv.gz`; Part 2's `--map-samples` input `results/logs/<run_id>/map/map_samples.npz` |
| Table 4b: agreement of a point's learned depth across the keyframes which observe it; fused vs single learned depth; the cost of dropping inconsistent points | `python scripts/part4/keyframe_consistency.py` (after 4a) | `results/tables/table4_consistency*.csv`; per point `results/logs/<run_id>/map/points_consistency.csv.gz` |
| Table 4: frame choice, failure case, own sequence | `make table4` | `results/tables/table4_*.csv` |
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
| `time_per_frame`, `time_per_frame_p90`, `wall_per_frame` | median (and 90th percentile) per-frame time of pySLAM's tracking thread (from its tracking.log; the median so that a sleeping laptop does not count); wall-clock seconds of the run / frames (with `speed=1` bounded by the playback), headless (machine: MacBook, Apple M4, 10 cores, 24 GB, macOS 26; pySLAM C++ core unless stated) |

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

The Depth Pro inference adapter uses **one fixed Depth Pro checkpoint** on all
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
run uses `--all` without `--max-frames`. The first three-frame Replica CPU test
on this WSL machine averaged 72.84 seconds per frame; its manifest records the
individual timings. This is a smoke test rather than a full-sequence benchmark.

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
python scripts/part2/visualize_depth.py --sequence replica_room0 --frame-index 0
```

The visualization writes `results/figures/part2/<sequence>_<frame>.png`: RGB,
raw predicted depth, GT depth and absolute error. Prediction and GT use the same
depth colour scale; no GT-fitted scale is applied. Use `--frame-index 1` or `2`
to inspect the other smoke-test frames.

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

### Three-frame Colab GPU test

`output/jupyter-notebook/depth_pro_colab_smoke.ipynb` provides a private,
three-frame GPU timing/evaluation test when local or UCL GPUs are unavailable.
Open it with Colab's **File → Upload notebook**, then select **Runtime → Change
runtime type → GPU** and **Runtime version 2026.07**. This uses Python 3.12,
compatible with the original Depth Pro dependency requirements. Free GPU
availability is not guaranteed; the notebook checks it before downloading weights.

Run the cells in order and upload `output/jupyter-notebook/cw1_gpu_smoke_inputs.zip`
when prompted. The small ZIP contains only three Replica RGB/depth pairs, poses,
calibration, analysis scripts and the matching CPU timing reference. The notebook
downloads the official checkpoint directly and offers a separate results ZIP for
download. Keep those results before ending the Colab session. GPU execution has
not been validated from this local environment.

Regenerate the portable input bundle after changing the scripts:

```bash
python scripts/part2/prepare_gpu_smoke.py
```

### Choosing between the three depth model families

The controlled comparison uses three spread frames per sequence (nine images),
separate caches, one excluded warmup per case and synchronized inference timing.
Depth Pro and **DA3 Metric Large** use one metric checkpoint across all sequences.
V2 Small, Base and Large each use the declared Hypersim checkpoint indoors and
VKITTI checkpoint outdoors. Relative-depth checkpoints are not interchangeable
with these metric models. GT depth is used only for scoring.

The measured CPU results and provenance are in
[`results/tables/model_comparison/comparison.md`](results/tables/model_comparison/comparison.md),
with CSV and a quality/time figure alongside it. This is a small selection pilot,
not full-sequence accuracy or SLAM evidence. KITTI has no supplied depth GT.
Models retain their official default preprocessing sizes: this compares practical
configurations rather than architectures at an equal image resolution.
The local pilot uses the Intel Core i5-13420H laptop under WSL2, FP32, with six
PyTorch intra-op and six inter-op threads; there is no CUDA GPU on this machine.
These samples favour **DA3 Metric Large** for balance: its runtime projects to
3.87 hours for all 5686 frames, versus 114.85 hours for Depth Pro. V2 Small projects
to 2.95 hours but has substantially higher raw indoor depth errors. These estimates
exclude setup, I/O and scoring; full-sequence and temporal validation remain.

To repeat the CPU comparison after obtaining the official sources and weights
under `weights/`:

```bash
python scripts/part2/compare_depth_models.py --device cpu --dry-run
python scripts/part2/compare_depth_models.py --device cpu
python scripts/part2/summarize_model_benchmarks.py
```

The first command prints the eight sequential cases without loading models.
The others benchmark and summarize them; completed matching predictions resume.
The comparison adapters do not change the default full-cache Depth Pro workflow.

For a GPU comparison, open
[`output/jupyter-notebook/compare_depth_models_colab.ipynb`](output/jupyter-notebook/compare_depth_models_colab.ipynb)
in Colab, select **GPU** and **Runtime version 2026.07**, and upload the locally
generated `output/jupyter-notebook/depth_model_comparison_inputs.zip` when
prompted. Input ZIPs are excluded from Git; create this one with the preparation
command below. It runs the same nine images, with additional separate FP16 cases
for Depth Pro and DA3. Sources and checkpoint checksums are pinned. Downloads are
about 6.9 GB; save the offered results ZIP before ending the runtime. This notebook
has passed local schema, code and input checks; its GPU cases have not been run
from this environment. The earlier three-frame Depth Pro GPU smoke test uses a
different sample and is not mixed into this comparison.

Regenerate its small portable input bundle after changing the comparison scripts:

```bash
python scripts/part2/prepare_model_comparison.py
```

### DA3-only free Colab test

Use [`da3_colab_smoke.ipynb`](output/jupyter-notebook/da3_colab_smoke.ipynb)
for the selected model's GPU test. Open Colab, choose **File > Upload notebook**,
and upload that notebook. Under **Runtime > Change runtime type**, select a GPU
and **Runtime version 2026.07** (Python 3.12), then choose **Runtime > Run all**.
Upload the locally generated `output/jupyter-notebook/depth_model_comparison_inputs.zip`
when prompted; generate it with `prepare_model_comparison.py` above. The notebook runs DA3 Metric Large in FP32 and FP16 on the same
nine selected frames and downloads only DA3's source and approximately 1.34 GB
of model weights. The ZIP also contains adapter code for the other models, but
those models are not loaded or downloaded by this notebook.

Save the offered results ZIP to Downloads before ending the Colab runtime. It
contains hardware details, timings, depth errors and predictions, including
sample-based estimates for the complete 5686-frame cache. This notebook is a
small GPU test, not a full-dataset prediction run. Local validation does not
establish successful Colab execution; the GPU results must be measured there.

The downloaded `da3_gpu_smoke_results.zip` has now been checked: both nine-frame
cases completed on a Tesla T4 with matching input/checkpoint hashes and no failed
frames. FP16 sample means are 0.115 s (Replica), 0.165 s (TUM) and 0.057 s (KITTI),
projecting to 11.99 minutes for all 5686 frames; FP32 projects to 29.68 minutes.
These estimates exclude I/O and transfers and include initialization on the
first TUM frame's new resolution. Indoor raw depth errors are nearly unchanged
between precisions. Rare larger outdoor boundary differences remain; KITTI has
no supplied GT depth. Details are in
[`da3_colab_review.json`](results/tables/model_comparison/da3_colab_review.json).

### Full DA3 FP16 cache on free Colab

[`da3_colab_full.ipynb`](output/jupyter-notebook/da3_colab_full.ipynb) generates
all **5686 frames** using the selected DA3 Metric Large checkpoint and FP16
autocast, with processing resolution 504. It uses the same calibrated inference
as the successful T4 pilot. The input bundle contains about 3.02 GB of original
RGB images, timestamps, calibration and scripts. Reference depth and poses stay
on the local machine; inference never uses them.

1. Upload the full notebook to Colab and select **GPU / T4** with **Runtime
   version 2026.07**.
2. Open Colab's **Files** sidebar and upload the locally generated
   `output/jupyter-notebook/da3_full_inputs.zip` (excluded from Git) to
   `/content`. Wait for the transfer to finish, then choose **Runtime > Run all**.
3. The notebook checks the input checksum and all frame identities, then runs
   KITTI (1101), TUM (2585), and Replica (2000). It reports progress and ETA about
   every 25 frames and exports each sequence separately.
4. Download the three timestamped dataset ZIPs from
   `/content/da3_full_exports` using **Files > right-click > Download**, one at a
   time. Save the small `da3_full_run_metadata.zip` too. Confirm the downloads
   have finished before ending the runtime.

The pilot projects to about 12 minutes of inference. Reading RGB, writing the
native NPY/PNG files, ZIP compression and transfers add time. Canonical NPY
output alone occupies approximately 11.70 GB before compression. The notebook
checks for 45 GiB of free Colab disk space before it starts. It uses local Colab
storage and separate dataset downloads; it does not mount Google Drive.

Each prediction produces a native float32 NPY in metres and a uint16 PNG using
the configured sequence factor (KITTI 256, TUM 5000, Replica 6553.5). Values above
the PNG range become invalid PNG pixels, while NPY retains the full prediction.
Per-frame manifests record input/output hashes, calibration, source/weights,
precision, hardware session, inference time and overflow statistics. Completed
frames resume in a surviving VM. If interrupted, stop the generation cell and
use its rescue cell to export committed frames. A downloaded partial cache can
be restored after reuploading the input bundle in a new VM; resetting a VM alone
does not preserve its files.

Import each downloaded dataset ZIP from the coursework directory:

```bash
python scripts/part2/transfer_da3_cache.py import \
  --archive /path/to/da3_kitti06_TIMESTAMP_results.zip \
  --pred-root depth_maps/da3_metric_large_fp16 --data-root ../data
```

Repeat for TUM and Replica. Import verifies the archive, original RGB content,
full sequence inventory, calibration, output hashes and NPY/PNG agreement before
publishing the cache. It rebuilds associations for local RGB paths. A compatible
partial cache can be extended by importing its full superset. The original
Depth Pro cache remains at its existing location.

After all three caches have been imported, run Part 2 evaluation with the
selected prediction root:

```bash
python scripts/part2/eval_depth.py --all \
  --pred-root depth_maps/da3_metric_large_fp16 \
  --output results/tables/da3/table2_depth.csv
python scripts/part2/temporal_consistency.py --all \
  --pred-root depth_maps/da3_metric_large_fp16 \
  --output results/tables/da3/table2_temporal.csv
python scripts/part2/keypoint_depth.py --all \
  --pred-root depth_maps/da3_metric_large_fp16 \
  --keypoints-root results/keypoints/da3 \
  --output results/tables/da3/table2_keypoints.csv
```

Regenerate the full input bundle after changing its scripts or inputs:

```bash
python scripts/part2/prepare_da3_full.py
```

The runner also supports local `--dry-run` without Torch or weights:
`python scripts/part2/predict_da3.py --all --dry-run`. Full GPU inference and
browser transfers require the user's Colab session; preparing and validating the
notebook locally does not execute that GPU job.

### Full DA3 cache locally in WSL

The existing local environment and checkpoint also support CPU inference. Use
**FP32 on CPU**, with a separate cache from the Colab FP16 predictions:

```bash
cd /mnt/c/Users/adria/University/Awareness/Scene-Understanding-CW1
/home/adria/.venvs/comp0227-model-comparison/bin/python scripts/part2/predict_da3.py \
  --all --device cpu --precision float32 --process-res 504 \
  --pred-root depth_maps/da3_metric_large_cpu_float32
```

The completed six-thread i5-13420H run took **5 h 08 min of timed prediction**
(preprocessing, network and native metric postprocessing), versus the original
nine-frame pilot projection of 3 h 52 min. Observed active sequence processing
took about **5 h 39 min**, including warmup and file handling; initial model
construction and resume validation are additional. An overnight interruption
is excluded from those active times. Keep the laptop powered and awake.
Matching completed frame pairs resume when the same command is run again.

The background run started on 2 November writes progress to
`results/logs/local_da3_cpu_20261102T010342Z/inference.log` and lifecycle state to
`job_status.json` in the same directory. Its cache has the same native NPY/PNG
format as the Colab workflow. Evaluate it with
`--pred-root depth_maps/da3_metric_large_cpu_float32` in the Part 2 commands above.

### Full CPU depth evaluation

The cache completed on 2 November at 18:26 UTC: KITTI 1,101, TUM 2,585 and
Replica 2,000 frames. The integrity audit checked all 5,686 NPY files, 5,686
PNG files and 5,686 source RGB hashes, plus native dimensions, encodings,
frame indices and associations. No missing, corrupt or uncommitted files were
found; recorded invalid and PNG-overflow fractions are zero.

Full-sequence raw depth scores (equal-frame averages):

| Dataset | GT-scored frames | AbsRel | RMSE | delta < 1.25 |
|---|---:|---:|---:|---:|
| TUM fr3 | 2,488 | 7.91% | 0.344 m | 95.06% |
| Replica room0 | 2,000 | 3.96% | 0.140 m | 99.75% |

TUM's other 97 RGB frames have predictions but no GT depth association within
20 ms. Invalid sensor pixels are masked. KITTI has no supplied depth GT.
Per-frame median scaling reduces AbsRel to 5.45% and 2.19%, respectively;
this GT-fitted diagnostic measures remaining shape error and must not supply
online SLAM scale.

The [full analysis](results/tables/da3/full_cpu_float32/analysis.md),
[machine-readable report](results/tables/da3/full_cpu_float32/analysis.json),
[accuracy and scale plot](results/figures/part2/da3_full_cpu_float32/depth_quality.png)
and [selected error maps](results/figures/part2/da3_full_cpu_float32/depth_examples.png)
contain the completed evaluation. CSV tables and integrity/timing/temporal
summaries are under `results/tables/da3/full_cpu_float32/`; figure PDFs are
saved alongside the PNGs. A separate 64-frame sampled spatial diagnostic
finds roughly twice the relative error in defined GT depth-discontinuity bands
as outside them; sample indices, thresholds and pixel populations are recorded
in `spatial_diagnostics.json`. Regenerate the report from the completed, checked
tables with the comparison environment active:

```bash
python scripts/part2/summarize_da3_run.py --data-root ../data
```

The report requires every adjacent temporal pair and every diagnostic ORB
frame, and records input-table hashes. ORB patch sampling uses a batched
implementation verified against the original scalar definition. These ORB
detections are not the saved SLAM observations, and local patch uncertainty is
an uncalibrated heuristic. Real SLAM-keypoint and learned-depth map-scale
analysis still needs `results/logs/<run_id>/map/map_points.npz` from Part 1;
the currently tracked trajectories and aggregate map CSVs cannot reconstruct
those observations. Portable `data/...` association paths also need rebasing
before direct pySLAM use. Part 3 integrations remain separate work.

### Inspect the completed KITTI depth cache

Open the [interactive viewer](results/figures/part2/kitti06_da3/viewer.html) in a
browser to browse 19 selected frames, adjust the colour range and hover for raw
predicted depth values on a two-pixel native grid. The
[overview](results/figures/part2/kitti06_da3/overview.png) uses a common 0-80 m
display scale; the [sampled animation](results/figures/part2/kitti06_da3/preview.gif)
shows every tenth frame at 5 fps, compressing the 114.3 s recording to 22.2 s.
Display clipping leaves cached metric values unchanged.

```bash
python scripts/part2/explore_depth.py \
  --pred-root depth_maps/da3_metric_large_cpu_float32
```

Per-frame cache diagnostics are saved under
`results/tables/da3/kitti06_manifest_diagnostics.*`. The 25-pair motion-compensated
temporal pilot is in `kitti06_temporal_pilot.csv` and its summary JSON alongside
those diagnostics. It uses predicted source geometry and predicted target
visibility; results describe conditional self-consistency, not depth accuracy.
KITTI has no supplied depth GT here. Sky/background values, moving vehicles,
boundary smoothing and camera/pose calibration all limit interpretation.

## Use of generative AI
Declared in the report (UCL Category 3). Running development log:

- 24 October 2026: Codex read the brief, reviewed the Part 1 scripts and saved
  results, and helped implement Part 2 dataset loading, the Depth Pro inference
  adapter, depth/temporal/keypoint evaluation, map-scale helpers and regression
  tests. Real network predictions and scientific conclusions have not yet been
  produced by this Part 2 implementation.
- 24 October 2026: Codex helped prepare a Colab GPU smoke-test notebook and a
  portable three-frame bundle, with notebook/schema and bundle validation. The
  local V2 Small benchmark and CPU Depth Pro predictions are separate from this
  unexecuted Colab test.
- 24 October 2026: Codex implemented metric model comparison adapters and a
  resumable benchmark, ran the official networks on shared spread samples, and
  prepared a separate Colab comparison notebook. The CPU pilot measures raw and
  median-scaled GT errors indoors and runtime on all three sequences. Model
  selection from this small sample still needs full-sequence validation; the
  comparison notebook's GPU cases have not been executed here.
- 24 October 2026: Codex prepared a DA3-only free Colab notebook using the same
  nine-frame bundle, with separate FP32/FP16 cases and full-cache timing
  extrapolation. Notebook code, schema and input checks run locally; actual
  Colab GPU execution requires the user's browser session.
- 24 October 2026: Codex inspected the downloaded DA3 FP32/FP16 T4 pilot and
  helped prepare the complete FP16 Colab workflow, RGB-only input bundle,
  per-frame cache/checkpoint runner and verified export/import helpers. Cache
  tests cover interrupted writes, relocation, identity changes, corruption and
  partial-to-full import. The full GPU run is performed in the user's Colab
  session, with local validation and reproducible scripts recorded here.
- 2 November 2026: Codex integrated the two fetched Part 1 commits while
  preserving local Part 2 files and merging both README versions, reran the
  103-test suite, and launched the full DA3 FP32 CPU cache in WSL with per-frame
  checkpoints and a separate prediction root.
- 2 November 2026: Codex generated KITTI cache diagnostics, static plots, a
  sampled video/animation and an interactive viewer from cached predictions,
  plus a 25-pair conditional temporal-consistency pilot using evaluation poses.
  No new network inference was used for these visualizations; KITTI depth
  accuracy remains unmeasured because supplied depth GT is unavailable.
- 2 November 2026: Codex recovered the interrupted CPU run, audited every saved
  prediction and source-image hash, ran full dense/temporal/diagnostic-keypoint
  evaluation, and generated analysis tables and figures from cached outputs.
  Batched keypoint patch sampling was checked against the original scalar
  method on synthetic and real inputs; the 127-test suite passed. No new
  inference was used for the offline analyses. Missing actual SLAM observations
  and the limits of fitted scale and heuristic uncertainty are recorded.

## Group / contributions
TODO
