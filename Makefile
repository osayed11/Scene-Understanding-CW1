# COMP0227 CW1 — which command makes which table. See README.md.
PY ?= python3
RUNS ?= 5
SEQS = kitti06 tum_fr3 replica_room0

.PHONY: setup test depth_maps table1 table2 table3 table4 report submission

setup:                 ## clone tagged pySLAM + both pixi envs (needs PYSLAM_TAG)
	bash scripts/setup/clone_pyslam.sh

test:
	$(PY) -m pytest -q tests

# ---- Part 1 (Table 1): baseline, cores, determinism, RGB-D reference ----
table1:
	for s in $(SEQS); do for m in baseline baseline_py; do $(PY) scripts/run_config.py --sequence $$s --method $$m --runs $(RUNS); done; done
	for s in tum_fr3 replica_room0; do $(PY) scripts/run_config.py --sequence $$s --method rgbd_gt --runs $(RUNS); done
	$(PY) scripts/part1/determinism.py
	$(PY) scripts/make_tables.py table1

# ---- Part 2 (Table 2): depth maps + depth quality + keypoint depth ----
depth_maps:
	$(PY) scripts/part2/predict_depth.py --all
table2: 
	$(PY) scripts/part2/eval_depth.py
	$(PY) scripts/part2/temporal_consistency.py
	$(PY) scripts/part2/keypoint_depth.py
	$(PY) scripts/make_tables.py table2

# ---- Part 3 (Table 3): pseudo RGB-D reference + methods + trust sweeps ----
table3:
	for s in $(SEQS); do for m in pseudo_rgbd method_a method_b; do \
	  $(PY) scripts/run_config.py --sequence $$s --method $$m --runs $(RUNS); done; done
	$(PY) scripts/make_tables.py table3

# ---- Part 4 (Table 4): agreement, frame choice, failure case, own sequence ----
table4:
	$(PY) scripts/part4/compare_depths.py
	$(PY) scripts/part4/keyframe_agreement.py
	$(PY) scripts/part4/frame_selection.py
	$(PY) scripts/part4/failure_case.py
	$(PY) scripts/run_config.py --sequence own --method baseline --runs $(RUNS)
	$(PY) scripts/run_config.py --sequence own --method method_a --runs $(RUNS)
	$(PY) scripts/make_tables.py table4

report: ; cd report && latexmk -pdf main.tex
submission: ; bash scripts/make_submission.sh
