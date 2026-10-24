"""Paths and loaders for the YAML configs in configs/sequences and configs/methods."""
from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]  # repo root (scripts/ is one level down)
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"
PYSLAM = ROOT / "third_party" / "pyslam"


def load_sequence(name: str) -> dict:
    return yaml.safe_load((CONFIGS / "sequences" / f"{name}.yaml").read_text())


def load_method(name: str) -> dict:
    return yaml.safe_load((CONFIGS / "methods" / f"{name}.yaml").read_text())


def run_id(sequence: str, method: str, settings: dict, run: int) -> str:
    """Stable identifier used for trajectory filenames and the results CSV."""
    s = "_".join(f"{k}={v}" for k, v in sorted(settings.items())) or "default"
    return f"{sequence}__{method}__{s}__run{run:02d}"


def count_frames(sequence: dict, run_dir: Path | None = None) -> int:
    """Number of frames the run processed: the denominator of the tracked share and of time per frame
    (not the number of ground-truth poses: TUM's ground truth is at 100 Hz, the camera at 30 Hz).
    Taken from pySLAM's own other_metrics_info.txt (num_total_frames) when the run has one, since the
    frame list depends on the mode (TUM RGB-D reads associations.txt, 2488 frames; mono rgb.txt, 2585);
    otherwise counted from the sequence's frame list."""
    import glob
    import re
    if run_dir is not None and (run_dir / "other_metrics_info.txt").exists():
        m = re.search(r"num_total_frames:\s*(\d+)", (run_dir / "other_metrics_info.txt").read_text())
        if m:
            return int(m.group(1))
    if "frame_list" in sequence:
        with open(ROOT / sequence["frame_list"]) as f:
            return sum(1 for l in f if l.strip() and not l.startswith("#"))
    return len(glob.glob(str(ROOT / sequence["frame_glob"])))
