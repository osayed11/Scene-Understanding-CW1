"""Package identical, spread-out inputs for a three-model GPU comparison.

The default is three frames per sequence at the rounded 10%, 50% and 90%
positions. TUM/Replica samples without an associated depth image are replaced
by the closest unused RGB frame with depth within 20 ms, with earlier positions
breaking ties. Original frame indices, stems and timestamps are retained in an
explicit sample_manifest.json; sparse Replica samples must not be reindexed by
the full-sequence loader. This is a small diagnostic sample, not a final ranking.

Only explicitly named Python/config/calibration files and the selected images
are included. No model weights, source repositories, predictions or full
sequences are copied. --manifest-only can prepare the selection before the model
adapters and benchmark harness are available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PREFIX = "cw1_depth_model_comparison"
SEQUENCES = ("replica_room0", "tum_fr3", "kitti06")
MAX_SAMPLES_PER_SEQUENCE = 3
CALIBRATIONS = {
    "replica_room0": "REPLICA.yaml", "tum_fr3": "TUM3.yaml", "kitti06": "KITTI04-12.yaml",
}
SCRIPT_FILES = (
    "scripts/config.py", "scripts/depth.py", "scripts/part2/common.py",
    "scripts/part2/eval_depth.py", "scripts/part2/predict_depth.py", "scripts/part2/model_adapters.py",
    "scripts/part2/da3_adapter.py", "scripts/part2/benchmark_models.py",
    "scripts/part2/summarize_model_benchmarks.py",
)
CONFIG_FILES = tuple(f"configs/sequences/{name}.yaml" for name in SEQUENCES)


def select_sample_frames(frames, samples_per_sequence=3, *, require_gt=False, max_dt=.02):
    """Select spread frame records; replacement never changes their identity."""
    if (not isinstance(samples_per_sequence, (int, np.integer))
            or not 1 <= samples_per_sequence <= MAX_SAMPLES_PER_SEQUENCE):
        raise ValueError(f"samples_per_sequence must be between 1 and {MAX_SAMPLES_PER_SEQUENCE}")
    if len(frames) < samples_per_sequence:
        raise ValueError("not enough distinct RGB frames for the requested sample")
    targets = np.rint(np.linspace(.1, .9, samples_per_sequence) * (len(frames) - 1)).astype(int)
    if require_gt:
        available = {
            i for i, frame in enumerate(frames)
            if frame.depth_gt is not None and Path(frame.depth_gt).is_file()
            and frame.depth_timestamp is not None
            and math.isfinite(float(frame.depth_timestamp))
            and abs(float(frame.depth_timestamp) - float(frame.timestamp)) <= max_dt
        }
    else:
        available = set(range(len(frames)))
    if len(available) < samples_per_sequence:
        raise ValueError("not enough distinct frames with valid associated ground-truth depth")
    positions = []
    for target in targets:
        chosen = min(available, key=lambda i: (abs(i - int(target)), i))
        positions.append(chosen)
        available.remove(chosen)
    return [frames[i] for i in sorted(positions)]


def _relative_data_path(path: Path, data_root: Path) -> str:
    return "data/" + Path(path).resolve().relative_to(data_root.resolve()).as_posix()


def create_sample_manifest(*, data_root: Path | None = None, samples_per_sequence=3,
                           calibration_root: Path | None = None) -> list[dict]:
    """Resolve full-dataset records once and retain their original identities."""
    from config import load_sequence
    from part2.common import Camera, find_data_root, load_camera, load_frame_records

    data = find_data_root(data_root)
    manifest = []
    for sequence in SEQUENCES:
        cfg = load_sequence(sequence)
        frames = load_frame_records(sequence, data_root=data)
        if calibration_root is None:
            camera = load_camera(sequence, data_root=data)
        else:
            import yaml
            settings = _calibration_file(ROOT, CALIBRATIONS[sequence], calibration_root)
            content = re.sub(r"\A\s*%YAML[^\n]*\n", "", settings.read_text())
            content = re.sub(r"(?m)^([A-Za-z_][\w.]*):(?=[^\s:])", r"\1: ", content)
            settings_cfg = yaml.safe_load(content)
            camera = Camera(*(float(settings_cfg[f"Camera.{key}"]) for key in ("fx", "fy", "cx", "cy")),
                            int(settings_cfg["Camera.width"]), int(settings_cfg["Camera.height"]))
        selected = select_sample_frames(frames, samples_per_sequence,
                                        require_gt=sequence != "kitti06")
        for frame in selected:
            factor = cfg.get("gt_depth_factor", cfg.get("depth_factor")) if frame.depth_gt is not None else None
            if factor is not None and (not math.isfinite(float(factor)) or float(factor) <= 0):
                raise ValueError(f"{sequence}: GT depth factor must be finite and positive")
            manifest.append({
                "sequence": sequence, "index": int(frame.index), "timestamp": float(frame.timestamp),
                "rgb": _relative_data_path(frame.rgb, data),
                "depth_gt": _relative_data_path(frame.depth_gt, data) if frame.depth_gt is not None else None,
                "depth_timestamp": float(frame.depth_timestamp) if frame.depth_timestamp is not None else None,
                "fx": float(camera.fx), "fy": float(camera.fy),
                "cx": float(camera.cx), "cy": float(camera.cy),
                "width": int(camera.width), "height": int(camera.height),
                "gt_depth_factor": float(factor) if factor is not None else None,
            })
    return manifest


def _manifest_bytes(manifest: list[dict]) -> bytes:
    return (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _data_file(data_root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if (path.is_absolute() or not path.parts or path.parts[0] != "data"
            or ".." in path.parts or "\\" in relative):
        raise ValueError(f"manifest image path must stay within data/: {relative}")
    resolved = data_root.joinpath(*path.parts[1:]).resolve()
    if not resolved.is_relative_to(data_root.resolve()):
        raise ValueError(f"manifest image path escapes the dataset root: {relative}")
    return resolved


def _calibration_file(repo_root: Path, name: str, calibration_root: Path | None) -> Path:
    if calibration_root is not None:
        candidates = [Path(calibration_root) / "settings" / name, Path(calibration_root) / name]
    else:
        pyslam = Path(os.environ.get("CW1_PYSLAM_ROOT", str(repo_root / "third_party/pyslam")))
        candidates = [pyslam / "settings" / name,
                      repo_root.parent / "COMP0227_26-27/pyslam/settings" / name]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Calibration {name} unavailable; pass --calibration-root")


def package_bundle(manifest: list[dict], *, repo_root: Path = ROOT, data_root: Path,
                   calibration_root: Path | None = None, output: Path | None = None) -> dict:
    """Create a deterministic ZIP from a strictly bounded explicit manifest."""
    repo = Path(repo_root).resolve()
    data = Path(data_root).resolve()
    if not manifest or len(manifest) > 3 * MAX_SAMPLES_PER_SEQUENCE:
        raise ValueError("manifest must contain between one and nine sample frames")
    counts = {name: 0 for name in SEQUENCES}
    identities = set()
    for row in manifest:
        name, index = row["sequence"], row["index"]
        if name not in counts or not isinstance(index, int) or index < 0:
            raise ValueError("manifest must use supported sequences and nonnegative original indices")
        counts[name] += 1
        if counts[name] > MAX_SAMPLES_PER_SEQUENCE or (name, index) in identities:
            raise ValueError("manifest contains duplicate indices or too many frames per sequence")
        identities.add((name, index))
        if name != "kitti06" and row["depth_gt"] is None:
            raise ValueError(f"{name}: associated ground-truth depth is required")
        if row["depth_gt"] is not None and (
            row["depth_timestamp"] is None
            or abs(float(row["depth_timestamp"]) - float(row["timestamp"])) > .02
            or not math.isfinite(float(row["gt_depth_factor"]))
            or float(row["gt_depth_factor"]) <= 0
        ):
            raise ValueError(f"{name}: invalid depth association or factor")
    encoded_manifest = _manifest_bytes(manifest)
    payloads = {relative: (repo / relative).read_bytes() for relative in (*SCRIPT_FILES, *CONFIG_FILES)}
    for name in CALIBRATIONS.values():
        payloads[f"calibration/settings/{name}"] = _calibration_file(repo, name, calibration_root).read_bytes()
    for row in manifest:
        for relative in (row["rgb"], row["depth_gt"]):
            if relative is not None:
                content = _data_file(data, relative).read_bytes()
                if not content:
                    raise ValueError(f"Empty selected input: {relative}")
                payloads[relative] = content
    payloads["sample_manifest.json"] = encoded_manifest
    output = Path(output) if output is not None else repo / "output/jupyter-notebook/depth_model_comparison_inputs.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=output.stem + "_", suffix=".tmp.zip",
                                         dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for relative, content in sorted(payloads.items()):
                info = zipfile.ZipInfo(f"{PREFIX}/{relative}", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, content)
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    manifest_path = output.with_name("sample_manifest.json")
    manifest_path.write_bytes(encoded_manifest)
    return {
        "output": str(output), "bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "sample_manifest": str(manifest_path), "frames": len(manifest), "frames_per_sequence": counts,
        "files": [f"{PREFIX}/{relative}" for relative in sorted(payloads)],
    }


def main(argv=None) -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    from part2.common import find_data_root

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--calibration-root", type=Path, help="pySLAM root or folder containing the three settings files")
    parser.add_argument("--samples-per-sequence", type=int, default=3,
                        help="spread frames per sequence, 1 to 3; default 3")
    parser.add_argument("--output", type=Path, default=ROOT / "output/jupyter-notebook/depth_model_comparison_inputs.zip")
    parser.add_argument("--manifest-only", action="store_true", help="write selection while waiting for benchmark scripts")
    args = parser.parse_args(argv)
    if not 1 <= args.samples_per_sequence <= MAX_SAMPLES_PER_SEQUENCE:
        parser.error("--samples-per-sequence must be between 1 and 3")
    data = find_data_root(args.data_root)
    manifest = create_sample_manifest(data_root=data, samples_per_sequence=args.samples_per_sequence,
                                      calibration_root=args.calibration_root)
    if args.manifest_only:
        path = args.output.with_name("sample_manifest.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_manifest_bytes(manifest))
        print(json.dumps({"sample_manifest": str(path), "frames": len(manifest)}, indent=2))
    else:
        print(json.dumps(package_bundle(manifest, data_root=data, calibration_root=args.calibration_root,
                                        output=args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
