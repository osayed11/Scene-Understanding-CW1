#!/usr/bin/env bash
# Clone the course version of pySLAM (sjulier fork, tag comp0227-2026-lab03.1) into third_party/pyslam and build it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
COMMIT="$(cat "$ROOT/PYSLAM_TAG")"
if [ ! -d "$ROOT/third_party/pyslam" ]; then
  git clone https://github.com/sjulier/pyslam.git "$ROOT/third_party/pyslam"
fi
cd "$ROOT/third_party/pyslam"
git checkout "$COMMIT"
pixi run build                     # environment + native modules + ORB vocabulary (15-60 min)
pixi run check                     # native modules load, optimiser tests pass
pixi run -e depth models-depth     # depth environment + ~7 GB of weights (Part 2)
