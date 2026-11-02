"""Package full, ordered RGB inputs for portable DA3 inference, without GT or weights.

The archive uses an explicit source allowlist. RGB bytes are hashed while they
are streamed into ZIP_STORED members; metadata is deflated. Dataset originals
are opened read-only. --inventory-only validates inputs and reports sizes without
building the multi-gigabyte ZIP or requiring the new inference/transfer scripts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "cw1_da3_full"
EXPECTED_COUNTS = {"kitti06": 1101, "tum_fr3": 2585, "replica_room0": 2000}
CALIBRATIONS = {"kitti06": "KITTI04-12.yaml", "tum_fr3": "TUM3.yaml", "replica_room0": "REPLICA.yaml"}
SCRIPT_FILES = (
    "scripts/config.py", "scripts/depth.py", "scripts/part2/common.py",
    "scripts/part2/eval_depth.py", "scripts/part2/predict_depth.py",
    "scripts/part2/model_adapters.py", "scripts/part2/da3_adapter.py",
    "scripts/part2/benchmark_models.py", "scripts/part2/summarize_model_benchmarks.py",
    "scripts/part2/predict_da3.py", "scripts/part2/transfer_da3_cache.py",
)


def checked_file(path: Path, root: Path) -> Path:
    """Reject aliases/escapes before reading an explicitly allowed source file."""
    path, root = Path(path).absolute(), Path(root).absolute()
    relative = path.relative_to(root)
    current = root
    if current.is_symlink():
        raise ValueError(f"Symlink source root is not allowed: {root}")
    for part in relative.parts:
        if part in ("..", "."):
            raise ValueError("Input paths must stay within their selected root")
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlink input is not allowed: {current}")
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise ValueError(f"Input file escapes its selected root: {path}")
    return path


def _data_root(repo_root: Path, explicit=None) -> Path:
    if explicit is not None:
        candidates = [Path(explicit)]
    elif os.environ.get("CW1_DATA_ROOT"):
        candidates = [Path(os.environ["CW1_DATA_ROOT"])]
    else:
        candidates = [repo_root / "data", repo_root.parent / "data"]
    for candidate in candidates:
        if all((candidate / name).is_dir() for name in ("kitti06", "tum", "replica")):
            if candidate.is_symlink():
                raise ValueError("Dataset root must not be a symlink")
            return candidate.absolute()
    raise FileNotFoundError("All three datasets must be present under --data-root")


def _calibration_root(repo_root: Path, explicit=None) -> Path:
    if explicit is not None:
        candidates = [Path(explicit)]
    else:
        candidates = [Path(os.environ.get("CW1_PYSLAM_ROOT", str(repo_root / "third_party/pyslam"))),
                      repo_root.parent / "COMP0227_26-27/pyslam"]
    for candidate in candidates:
        if all((candidate / "settings" / name).is_file() for name in CALIBRATIONS.values()):
            return candidate.absolute()
    raise FileNotFoundError("Camera settings unavailable; pass --calibration-root /path/to/pyslam")


def _numeric(text: str, key: str) -> float:
    match = re.search(r"(?m)^\s*" + re.escape(key) + r"\s*:\s*([0-9.eE+\-]+)\s*(?:#.*)?$", text)
    if not match:
        raise ValueError(f"Missing numeric setting {key}")
    value = float(match[1])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite setting {key}")
    return value


def _camera(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    camera = {key: _numeric(text, "Camera." + key) for key in ("fx", "fy", "cx", "cy", "width", "height")}
    for key in ("fx", "fy", "width", "height"):
        if camera[key] <= 0:
            raise ValueError(f"Camera {key} must be positive")
    for key in ("width", "height"):
        if int(camera[key]) != camera[key]:
            raise ValueError("Camera dimensions must be integers")
        camera[key] = int(camera[key])
    return camera


def _rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [line.split() for line in handle if line.strip() and not line.lstrip().startswith("#")]


def inventory_inputs(*, repo_root=ROOT, data_root=None, calibration_root=None) -> dict:
    repo_root = Path(repo_root).absolute()
    data = _data_root(repo_root, data_root)
    calibration = _calibration_root(repo_root, calibration_root)
    manifest = {"schema_version": 1, "total_frames": 0, "sequences": {}}
    sources, metadata, archive_names, source_names = [], {}, set(), set()
    for sequence, expected in EXPECTED_COUNTS.items():
        config_relative = f"configs/sequences/{sequence}.yaml"
        config = checked_file(repo_root / config_relative, repo_root)
        settings = checked_file(calibration / "settings" / CALIBRATIONS[sequence], calibration)
        metadata[config_relative] = config
        metadata["calibration/settings/" + CALIBRATIONS[sequence]] = settings
        depth_factor = _numeric(config.read_text(encoding="utf-8"), "depth_factor")
        if depth_factor <= 0:
            raise ValueError("Depth factor must be positive")
        camera = _camera(settings)
        if sequence == "kitti06":
            time_path = checked_file(data / "kitti06/times.txt", data)
            metadata["data/kitti06/times.txt"] = time_path
            times = _rows(time_path)
            if any(len(row) != 1 for row in times):
                raise ValueError("KITTI times.txt must contain one timestamp per row")
            images = sorted((data / "kitti06/image_2").glob("*.png"))
            if [image.name for image in images] != [f"{i:06d}.png" for i in range(len(images))]:
                raise ValueError("KITTI image numbering must be contiguous from 000000")
            if len(images) != len(times):
                raise ValueError("KITTI image count differs from timestamp count")
            ordered = [(float(row[0]), image) for row, image in zip(times, images)]
        elif sequence == "tum_fr3":
            folder = data / "tum/rgbd_dataset_freiburg3_long_office_household"
            rgb_list = checked_file(folder / "rgb.txt", data)
            metadata["data/tum/rgbd_dataset_freiburg3_long_office_household/rgb.txt"] = rgb_list
            ordered = []
            for row in _rows(rgb_list):
                if len(row) != 2:
                    raise ValueError("TUM rgb.txt must contain timestamp and RGB relative path")
                relative = PurePosixPath(row[1])
                if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2 or relative.parts[0] != "rgb" or relative.suffix != ".png":
                    raise ValueError("TUM rgb.txt may reference only files within its rgb/ folder")
                ordered.append((float(row[0]), folder.joinpath(*relative.parts)))
        else:
            images = sorted((data / "replica/room0/results").glob("frame*.jpg"))
            if [image.name for image in images] != [f"frame{i:06d}.jpg" for i in range(len(images))]:
                raise ValueError("Replica image numbering must be contiguous from frame000000")
            ordered = [(i / 25, image) for i, image in enumerate(images)]
        if len(ordered) != expected:
            raise ValueError(f"{sequence}: expected {expected} frames, found {len(ordered)}")
        times = [t for t, _ in ordered]
        if not all(math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError(f"{sequence}: timestamps must be finite and strictly ordered")
        frames, size = [], 0
        for index, (timestamp, source) in enumerate(ordered):
            source = checked_file(source, data)
            relative = "data/" + source.relative_to(data).as_posix()
            key = relative.casefold()
            source_key = str(source.resolve()).casefold()
            if key in archive_names or source_key in source_names:
                raise ValueError(f"Duplicate input/output identity: {relative}")
            archive_names.add(key)
            source_names.add(source_key)
            stat = source.stat()
            if stat.st_size <= 0:
                raise ValueError(f"Empty RGB input: {relative}")
            frame = {"index": index, "timestamp": timestamp, "rgb": relative}
            frames.append(frame)
            sources.append({"path": source, "relative": relative, "frame": frame,
                            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
            size += stat.st_size
        manifest["sequences"][sequence] = {"count": len(frames), "camera": camera,
                                           "depth_factor": depth_factor, "frames": frames}
        manifest["total_frames"] += len(frames)
    counts = {name: entry["count"] for name, entry in manifest["sequences"].items()}
    pixels = sum(entry["count"] * entry["camera"]["height"] * entry["camera"]["width"] * 4
                 for entry in manifest["sequences"].values())
    return {"manifest": manifest, "sources": sources, "metadata": metadata, "data_root": data,
            "frame_counts": counts, "rgb_bytes": sum(source["size"] for source in sources),
            "uncompressed_npy_pixel_bytes": pixels,
            "uncompressed_npy_bytes": pixels + manifest["total_frames"] * 128}


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _small_member(archive, relative, content):
    info = zipfile.ZipInfo(f"{PREFIX}/{relative}", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def build_bundle(*, repo_root=ROOT, data_root=None, calibration_root=None, output=None) -> dict:
    inventory = inventory_inputs(repo_root=repo_root, data_root=data_root, calibration_root=calibration_root)
    repo_root = Path(repo_root).absolute()
    output = Path(output) if output is not None else repo_root / "output/jupyter-notebook/da3_full_inputs.zip"
    output = output.absolute()
    if output.suffix.lower() != ".zip" or output.resolve().is_relative_to(inventory["data_root"].resolve()):
        raise ValueError("Output must be a .zip outside the read-only data directory")
    metadata = dict(inventory["metadata"])
    for relative in SCRIPT_FILES:
        metadata[relative] = checked_file(repo_root / relative, repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=output.stem + "_", suffix=".tmp.zip", dir=output.parent,
                                         delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            written_names = set()
            for relative, source in sorted(metadata.items()):
                key = relative.casefold()
                if key in written_names:
                    raise ValueError("Duplicate metadata archive member")
                written_names.add(key)
                _small_member(archive, relative, source.read_bytes())
            for source in inventory["sources"]:
                relative = source["relative"]
                if relative.casefold() in written_names:
                    raise ValueError(f"Duplicate RGB archive member: {relative}")
                written_names.add(relative.casefold())
                info = zipfile.ZipInfo(f"{PREFIX}/{relative}", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.file_size = source["size"]
                digest, copied = hashlib.sha256(), 0
                with source["path"].open("rb") as reader, archive.open(info, "w", force_zip64=True) as writer:
                    for block in iter(lambda: reader.read(1024 * 1024), b""):
                        digest.update(block)
                        writer.write(block)
                        copied += len(block)
                after = source["path"].stat()
                if copied != source["size"] or after.st_size != source["size"] or after.st_mtime_ns != source["mtime_ns"]:
                    raise ValueError(f"Source input changed while packaging: {relative}")
                source["frame"]["rgb_sha256"] = digest.hexdigest()
                if source["frame"]["index"] % 500 == 0:
                    print(f"Packaged {relative}", flush=True)
            manifest_bytes = (json.dumps(inventory["manifest"], indent=2, allow_nan=False) + "\n").encode()
            _small_member(archive, "full_input_manifest.json", manifest_bytes)
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    manifest_path = output.parent / "full_input_manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    summary = {"schema_version": 1, "archive": output.name, "prefix": PREFIX,
               "bytes": output.stat().st_size, "sha256": _hash_file(output),
               "frame_counts": inventory["frame_counts"], "total_frames": inventory["manifest"]["total_frames"],
               "rgb_bytes": inventory["rgb_bytes"],
               "input_identity_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "uncompressed_npy_pixel_bytes": inventory["uncompressed_npy_pixel_bytes"],
               "uncompressed_npy_bytes": inventory["uncompressed_npy_bytes"],
               "npy_size_note": "float32 HxW arrays with 128-byte NPY v1 headers; excludes optional PNGs",
               "archive_members": len(metadata) + len(inventory["sources"]) + 1}
    (output.parent / "da3_full_inputs_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args(argv)
    if args.inventory_only:
        inventory = inventory_inputs(data_root=args.data_root, calibration_root=args.calibration_root)
        report = {key: inventory[key] for key in ("frame_counts", "rgb_bytes", "uncompressed_npy_pixel_bytes", "uncompressed_npy_bytes")}
        report["total_frames"] = inventory["manifest"]["total_frames"]
    else:
        report = build_bundle(data_root=args.data_root, calibration_root=args.calibration_root, output=args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
