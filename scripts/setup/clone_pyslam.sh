#!/usr/bin/env bash
# Clone the tagged course version of pySLAM into third_party/pyslam and install both pixi environments.
set -euo pipefail
TAG="${PYSLAM_TAG:?set PYSLAM_TAG to the tag named in the coursework release}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
git clone https://github.com/luigifreda/pyslam.git "$ROOT/third_party/pyslam"
cd "$ROOT/third_party/pyslam"
git checkout "$TAG"
git submodule update --init --recursive
echo "$TAG" > "$ROOT/PYSLAM_TAG"
pixi install                       # default environment (SLAM runs)
pixi run -e depth models-depth     # depth environment + ~7 GB of weights
