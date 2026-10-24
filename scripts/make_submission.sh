#!/usr/bin/env bash
# Build the three submission items: report PDF, code ZIP (no pySLAM / data / weights), results CSV + trajectories.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
bash scripts/setup/export_pyslam_diff.sh
mkdir -p submission
zip -r submission/code.zip README.md Makefile pyproject.toml scripts configs tests pyslam.diff PYSLAM_TAG \
    -x '*/__pycache__/*' '*.pyc'
zip -r submission/results.zip results/results.csv results/trajectories results/tables
cp report/main.pdf submission/report.pdf 2>/dev/null || echo "report/main.pdf not built yet"
ls -la submission
