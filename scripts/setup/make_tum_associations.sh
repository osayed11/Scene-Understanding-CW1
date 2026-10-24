#!/usr/bin/env bash
# TUM RGB-D mode needs an associations.txt (rgb_t rgb_path depth_t depth_path); pySLAM ships the TUM tool.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SEQ="$ROOT/data/tum/rgbd_dataset_freiburg3_long_office_household"
cd "$ROOT/third_party/pyslam"
pixi run python scripts/associate.py "$SEQ/rgb.txt" "$SEQ/depth.txt" > "$SEQ/associations.txt"
wc -l "$SEQ/associations.txt"
