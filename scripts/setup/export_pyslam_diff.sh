#!/usr/bin/env bash
# Write our pySLAM changes as a diff against the tagged commit (pyslam.diff at the repo root, submitted).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TAG="$(cat "$ROOT/PYSLAM_TAG")"  # commit 2c485a7b = tag comp0227-2026-lab03.1 of github.com/sjulier/pyslam
git -C "$ROOT/third_party/pyslam" diff "$TAG" > "$ROOT/pyslam.diff"
echo "wrote pyslam.diff ($(wc -l < "$ROOT/pyslam.diff") lines) against $TAG"
