"""Create a portable, strictly allowlisted Replica GPU smoke-test input archive.

Only the named scripts/config/calibration, the first few RGB/depth pairs and pose
rows, and a sanitized local CPU timing reference are included. Model weights,
predictions, repository metadata and the rest of the datasets are never included.
This packager uses only Python's standard library.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[2]
PREFIX = "cw1_gpu_smoke"
MAX_FRAMES = 10
SCRIPT_FILES = (
    "scripts/config.py",
    "scripts/depth.py",
    "scripts/part2/common.py",
    "scripts/part2/predict_depth.py",
    "scripts/part2/eval_depth.py",
    "scripts/part2/visualize_depth.py",
)
SEQUENCE_CONFIG = "configs/sequences/replica_room0.yaml"


def _data_root(repo_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        candidates = [Path(explicit)]
    elif os.environ.get("CW1_DATA_ROOT"):
        candidates = [Path(os.environ["CW1_DATA_ROOT"])]
    else:
        candidates = [repo_root / "data", repo_root.parent / "data"]
    for candidate in candidates:
        if (candidate / "replica/room0/results/frame000000.jpg").is_file():
            return candidate
    raise FileNotFoundError("Replica room0 inputs unavailable; pass --data-root /path/to/data")


def _calibration(repo_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        candidates = [Path(explicit)]
    else:
        pyslam = Path(os.environ.get("CW1_PYSLAM_ROOT", str(repo_root / "third_party/pyslam")))
        candidates = [pyslam / "settings/REPLICA.yaml",
                      repo_root.parent / "COMP0227_26-27/pyslam/settings/REPLICA.yaml"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("REPLICA.yaml unavailable; pass --calibration /path/to/REPLICA.yaml")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _cpu_reference(manifest_path: Path, rgb_contents: dict[str, bytes], frames: int) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    signature = manifest["signature"]
    if manifest.get("sequence") != "replica_room0" or signature.get("model") != "depth_pro":
        raise ValueError("CPU reference must be the real Replica room0 Depth Pro manifest")
    if signature.get("device") != "cpu":
        raise ValueError("The reference manifest must describe CPU inference")
    checkpoint_hash = signature.get("checkpoint_sha256", "")
    if not isinstance(checkpoint_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", checkpoint_hash):
        raise ValueError("CPU reference has no valid checkpoint SHA256")
    reference_frames, missing = [], []
    for index in range(frames):
        stem = f"frame{index:06d}"
        entry = manifest.get("frames", {}).get(stem)
        if entry is None:
            missing.append(index)
            continue
        if entry.get("index") != index or not math.isclose(float(entry["timestamp"]), index / 25,
                                                           rel_tol=0, abs_tol=1e-8):
            raise ValueError(f"CPU reference has a mismatched frame index/timestamp: {stem}")
        rgb_hash = _sha256(rgb_contents[stem])
        if entry.get("rgb_sha256") != rgb_hash:
            raise ValueError(f"CPU reference RGB checksum does not match bundled input: {stem}")
        seconds = float(entry["inference_seconds"])
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"CPU reference has invalid timing for {stem}")
        reference_frames.append({"stem": stem, "index": index, "timestamp": index / 25,
                                 "rgb_sha256": rgb_hash, "inference_seconds": seconds})
    if not reference_frames:
        raise ValueError("No CPU reference timings match the requested input frames")
    # Explicit fields, rather than recursively copying the manifest, prevent local
    # paths or future private metadata from escaping into the shared archive.
    return {
        "schema_version": 1,
        "benchmark_kind": "adjacent_frame_cpu_smoke_reference",
        "sequence": "replica_room0", "model": "depth_pro",
        "checkpoint_sha256": checkpoint_hash.lower(), "device": "cpu",
        "precision": str(signature["precision"]), "input_color": str(signature["input_color"]),
        "depth_factor": float(signature["depth_factor"]),
        "focal_px": float(signature["focal_px"]),
        "input_size": [int(value) for value in signature["input_size"]],
        "bundle_frame_count": frames, "reference_frame_count": len(reference_frames),
        "missing_reference_indices": missing, "frames": reference_frames,
        "mean_inference_seconds": statistics.mean(row["inference_seconds"] for row in reference_frames),
        "note": "Adjacent-frame smoke test; inference timing excludes model initialization.",
    }


def build_bundle(*, repo_root: Path = ROOT, data_root: Path | None = None,
                 calibration: Path | None = None, cpu_manifest: Path | None = None,
                 output: Path | None = None, frames: int = 3) -> dict:
    """Write a ZIP from the explicit allowlist and return its size/checksum/layout."""
    if not isinstance(frames, int) or not 1 <= frames <= MAX_FRAMES:
        raise ValueError(f"frames must be an integer between 1 and {MAX_FRAMES}")
    repo_root = Path(repo_root).resolve()
    data = _data_root(repo_root, data_root)
    settings = _calibration(repo_root, calibration)
    manifest_path = Path(cpu_manifest) if cpu_manifest is not None else repo_root / "depth_maps/replica_room0/manifest.json"
    output = Path(output) if output is not None else repo_root / "output/jupyter-notebook/cw1_gpu_smoke_inputs.zip"
    payloads = {relative: (repo_root / relative).read_bytes() for relative in (*SCRIPT_FILES, SEQUENCE_CONFIG)}
    payloads["calibration/settings/REPLICA.yaml"] = settings.read_bytes()
    rgb_contents = {}
    sequence_dir = data / "replica/room0"
    for index in range(frames):
        for prefix, extension in (("frame", "jpg"), ("depth", "png")):
            name = f"{prefix}{index:06d}.{extension}"
            content = (sequence_dir / "results" / name).read_bytes()
            if not content:
                raise ValueError(f"Empty input file: {name}")
            payloads[f"data/replica/room0/results/{name}"] = content
            if prefix == "frame":
                rgb_contents[f"frame{index:06d}"] = content
    with (sequence_dir / "traj.txt").open(encoding="utf-8") as handle:
        rows = list(islice((line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")), frames))
    if len(rows) != frames:
        raise ValueError(f"traj.txt has fewer than {frames} pose rows")
    for row in rows:
        values = [float(value) for value in row.split()]
        if len(values) != 16 or not all(math.isfinite(value) for value in values):
            raise ValueError("Replica traj.txt must contain finite flattened 4x4 poses")
    payloads["data/replica/room0/traj.txt"] = ("\n".join(rows) + "\n").encode("utf-8")
    reference = _cpu_reference(manifest_path, rgb_contents, frames)
    payloads["cpu_reference.json"] = (json.dumps(reference, indent=2, allow_nan=False) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=output.stem + "_", suffix=".tmp.zip",
                                         dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for relative, content in sorted(payloads.items()):
                archive.writestr(f"{PREFIX}/{relative}", content)
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"output": str(output), "bytes": output.stat().st_size,
            "sha256": _sha256(output.read_bytes()), "frames": frames,
            "files": [f"{PREFIX}/{relative}" for relative in sorted(payloads)],
            "cpu_mean_inference_seconds": reference["mean_inference_seconds"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--calibration", type=Path, help="source settings/REPLICA.yaml")
    parser.add_argument("--cpu-manifest", type=Path, help="real local CPU Depth Pro manifest.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frames", type=int, default=3, help=f"first N pairs; default 3, maximum {MAX_FRAMES}")
    args = parser.parse_args(argv)
    if not 1 <= args.frames <= MAX_FRAMES:
        parser.error(f"--frames must be between 1 and {MAX_FRAMES}")
    result = build_bundle(data_root=args.data_root, calibration=args.calibration,
                          cpu_manifest=args.cpu_manifest, output=args.output, frames=args.frames)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
