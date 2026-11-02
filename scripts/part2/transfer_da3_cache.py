"""Export one DA3 sequence cache, or verify/import it against the local RGB data.

The downloaded archive retains lossless metre NPYs and the sequence-specific
16-bit PNGs. Import rebuilds associations for this machine; Colab paths are never
used as local RGB paths. Input archives are read without modifying the original.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ROOT, load_sequence
from part2.common import SEQUENCES, find_data_root, load_camera, load_frame_records

MAX_JSON_BYTES = 32 * 1024**2


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _frames(manifest, sequence):
    if (manifest.get("schema_version") != 2 or manifest.get("sequence") != sequence
            or manifest.get("signature", {}).get("model") != "da3_metric_large"):
        raise ValueError("Expected a DA3 Metric Large cache for the declared sequence")
    rows = manifest.get("frames")
    if not isinstance(rows, dict) or not rows:
        raise ValueError("No committed predictions in this cache")
    indices = []
    for stem, row in rows.items():
        if (not isinstance(stem, str) or not stem or stem in (".", "..")
                or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in stem)
                or any(ord(c) < 32 for c in stem)):
            raise ValueError("Unsafe prediction stem")
        if not isinstance(row.get("index"), int) or row["index"] < 0:
            raise ValueError("Invalid frame index")
        if not math.isfinite(float(row.get("timestamp", float("nan")))):
            raise ValueError("Invalid frame timestamp")
        for key in ("rgb_sha256", "npy_sha256", "png_sha256"):
            value = row.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Missing/invalid {key} for {stem}")
        indices.append(row["index"])
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate frame indices")
    return rows


def export_cache(sequence, pred_root, output):
    """Archive only committed, checksum-verified frames; partial caches are valid."""
    folder = Path(pred_root).resolve() / sequence
    if sequence not in SEQUENCES or folder.is_symlink():
        raise ValueError("Invalid sequence directory")
    manifest_path = folder / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    rows = _frames(manifest, sequence)
    files = {f"{sequence}/manifest.json": manifest_path}
    snapshots = {f"{sequence}/manifest.json": manifest_bytes}
    known_hashes = {}
    print(f"Verifying {sequence}: {len(rows)} committed frames", flush=True)
    for stem, row in rows.items():
        for suffix in ("npy", "png"):
            path = folder / f"{stem}.{suffix}"
            if path.is_symlink() or not path.is_file() or digest_file(path) != row[f"{suffix}_sha256"]:
                raise ValueError(f"Missing or changed cached prediction: {path}")
            files[f"{sequence}/{path.name}"] = path
            known_hashes[f"{sequence}/{path.name}"] = row[f"{suffix}_sha256"]
    associations = folder / "associations.txt"
    if associations.is_file() and not associations.is_symlink():
        files[f"{sequence}/associations.txt"] = associations
        snapshots[f"{sequence}/associations.txt"] = associations.read_bytes()
    records = {name: {"bytes": len(snapshots[name]) if name in snapshots else path.stat().st_size,
                      "sha256": hashlib.sha256(snapshots[name]).hexdigest() if name in snapshots else known_hashes[name]}
               for name, path in files.items()}
    metadata = {"schema_version": 1, "sequence": sequence, "cached_frames": len(rows),
                "full_sequence_frames": manifest.get("full_count"),
                "full_sequence": bool(manifest.get("full_complete")),
                "cache_status": manifest.get("status"), "files": records}
    output = Path(output).resolve()
    if output.is_relative_to(folder):
        raise ValueError("Export archive must be outside its source sequence directory")
    if output.exists():
        raise FileExistsError(f"Archive already exists: {output}; choose a new name")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1,
                             allowZip64=True) as archive:
            for position, (name, path) in enumerate(files.items()):
                if name in snapshots:
                    archive.writestr(name, snapshots[name])
                else:
                    archive.write(path, name)
                if position % 250 == 0:
                    print(f"Packing {sequence}: {position + 1}/{len(files)} files", flush=True)
            archive.writestr("archive_manifest.json", json.dumps(metadata, indent=2) + "\n")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"sequence": sequence, "cached_frames": len(rows), "output": str(output),
            "bytes": output.stat().st_size, "sha256": digest_file(output)}


def _read_json(archive, name):
    if archive.getinfo(name).file_size > MAX_JSON_BYTES:
        raise ValueError(f"Oversized JSON member: {name}")
    return json.loads(archive.read(name))


def _safe_members(archive):
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate archive members")
    for info in infos:
        path = PurePosixPath(info.filename)
        if (path.is_absolute() or ".." in path.parts or "\\" in info.filename or ":" in info.filename
                or stat.S_ISLNK(info.external_attr >> 16) or info.is_dir()
                or any(ord(c) < 32 for c in info.filename)):
            raise ValueError(f"Unsafe archive member: {info.filename}")
    return set(names)


def _verify_local_inputs(sequence, manifest, rows, data_root):
    import numpy as np
    data = find_data_root(data_root)
    frames = load_frame_records(sequence, data)
    if manifest.get("full_count") != len(frames):
        raise ValueError("Local full sequence frame count differs from the prediction cache")
    current = {frame.index: frame for frame in frames}
    camera = load_camera(sequence, data)
    saved = manifest["signature"].get("camera")
    if not isinstance(saved, dict) or any(k not in saved or not np.isclose(saved[k], v, rtol=0, atol=1e-9)
                                          for k, v in asdict(camera).items()):
        raise ValueError("Local calibration differs from the prediction cache")
    if "K" in saved and not np.allclose(saved["K"], camera.K, rtol=0, atol=1e-9):
        raise ValueError("Local camera matrix differs from the prediction cache")
    factor = float(load_sequence(sequence)["depth_factor"])
    if manifest["signature"].get("depth_factor") != factor:
        raise ValueError("Local PNG depth factor differs from the prediction cache")
    def portable_rgb(frame):
        return "data/" + Path(os.path.abspath(frame.rgb)).relative_to(Path(os.path.abspath(data))).as_posix()
    inventory = [{"index": int(f.index), "timestamp": float(f.timestamp), "rgb": portable_rgb(f)} for f in frames]
    fingerprint = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    declared = manifest["signature"].get("full_frame_inventory_sha256")
    if declared is not None and declared != fingerprint:
        raise ValueError("Local full RGB frame inventory differs from the prediction cache")
    for stem, row in rows.items():
        frame = current.get(row["index"])
        if frame is None or frame.stem != stem or abs(frame.timestamp - row["timestamp"]) > 1e-9:
            raise ValueError(f"Local frame identity differs: {stem}")
        expected_rgb = portable_rgb(frame)
        if row.get("rgb") != expected_rgb or digest_file(frame.rgb) != row["rgb_sha256"]:
            raise ValueError(f"Local RGB content/path differs: {stem}")
    return current, camera, factor, len(frames)


def _verify_depth_files(folder, rows, camera, factor, overflow):
    import cv2
    import numpy as np
    cap = 65535 / factor
    for stem in rows:
        depth = np.load(folder / f"{stem}.npy", mmap_mode="r", allow_pickle=False)
        png = cv2.imread(str(folder / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        if depth.dtype != np.float32 or depth.shape != (camera.height, camera.width):
            raise ValueError(f"Invalid canonical prediction format: {stem}")
        if np.any(np.isfinite(depth) & (depth < 0)) or not np.any(np.isfinite(depth) & (depth > 0)):
            raise ValueError(f"Invalid canonical prediction values: {stem}")
        if png is None or png.dtype != np.uint16 or png.shape != depth.shape:
            raise ValueError(f"Invalid PNG prediction format: {stem}")
        valid = np.isfinite(depth) & (depth > 0) & (depth <= cap)
        if overflow == "raise" and np.any(np.isfinite(depth) & (depth > cap)):
            raise ValueError(f"Unexpected PNG overflow: {stem}")
        encoded = np.where(valid, depth * factor, 0).round().astype(np.uint16)
        if not np.array_equal(encoded, png):
            raise ValueError(f"PNG does not encode its canonical prediction: {stem}")


def _associations(folder, rows, current):
    lines = ["# rgb_timestamp rgb_path depth_timestamp predicted_depth_path"]
    for stem, row in sorted(rows.items(), key=lambda item: item[1]["index"]):
        rgb = Path(os.path.relpath(current[row["index"]].rgb.resolve(), folder.resolve())).as_posix()
        if any(c.isspace() for c in rgb):
            raise ValueError("TUM associations require RGB paths without whitespace")
        lines.append(f"{row['timestamp']:.9f} {rgb} {row['timestamp']:.9f} {stem}.png")
    temporary = folder / "associations.txt.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(folder / "associations.txt")


def import_cache(archive_path, pred_root, data_root=None, expected_sha256=None):
    """Verify all data before adding it to a compatible local sequence cache."""
    if expected_sha256 is not None and digest_file(archive_path) != expected_sha256:
        raise ValueError("Downloaded archive SHA256 does not match the expected value")
    root = Path(pred_root).resolve()
    if root == (ROOT / "depth_maps").resolve():
        raise ValueError("Use a separate --pred-root to preserve the Depth Pro cache")
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        names = _safe_members(archive)
        metadata = _read_json(archive, "archive_manifest.json")
        sequence = metadata.get("sequence")
        if sequence not in SEQUENCES:
            raise ValueError("Unknown archive sequence")
        manifest = _read_json(archive, f"{sequence}/manifest.json")
        rows = _frames(manifest, sequence)
        expected = {"archive_manifest.json", f"{sequence}/manifest.json"}
        expected.update(f"{sequence}/{stem}.{suffix}" for stem in rows for suffix in ("npy", "png"))
        if f"{sequence}/associations.txt" in names:
            expected.add(f"{sequence}/associations.txt")
        if names != expected or set(metadata.get("files", {})) != expected - {"archive_manifest.json"}:
            raise ValueError("Archive contains missing or undeclared files")
        required_bytes = sum(archive.getinfo(name).file_size for name in names)
        if required_bytes > 40 * 1024**3:
            raise ValueError("Unexpectedly large single-sequence cache")
        if shutil.disk_usage(root).free < required_bytes + 128 * 1024**2:
            raise OSError("Insufficient disk space to stage and verify the downloaded cache")
        current, camera, factor, total = _verify_local_inputs(sequence, manifest, rows, data_root)
        target = root / sequence
        if target.is_symlink():
            raise ValueError("Prediction sequence directory must not be a symlink")
        old = None
        if target.exists():
            if not (target / "manifest.json").is_file():
                raise ValueError("Existing target has no manifest; choose a separate prediction root")
            old = json.loads((target / "manifest.json").read_text())
            old_rows = _frames(old, sequence)
            if old["signature"] != manifest["signature"]:
                raise ValueError("Existing cache has a different model/calibration/encoding")
            for stem, row in old_rows.items():
                if stem not in rows:
                    raise ValueError("Import must contain the existing cache; use another root for disjoint partials")
                if any(row[k] != rows[stem][k] for k in ("npy_sha256", "png_sha256", "rgb_sha256")):
                    raise ValueError(f"Import conflicts with an existing prediction: {stem}")
        with tempfile.TemporaryDirectory(prefix=f".import-{sequence}-", dir=root) as temporary:
            staged = Path(temporary) / sequence
            staged.mkdir()
            for name, record in metadata["files"].items():
                if archive.getinfo(name).file_size != record["bytes"]:
                    raise ValueError(f"Archive size differs from its manifest: {name}")
                path = staged / PurePosixPath(name).name
                digest = hashlib.sha256()
                with archive.open(name) as source, path.open("wb") as dest:
                    for block in iter(lambda: source.read(8 * 1024**2), b""):
                        digest.update(block)
                        dest.write(block)
                if digest.hexdigest() != record["sha256"]:
                    raise ValueError(f"Archive checksum differs: {name}")
            for stem, row in rows.items():
                for suffix in ("npy", "png"):
                    if digest_file(staged / f"{stem}.{suffix}") != row[f"{suffix}_sha256"]:
                        raise ValueError(f"Prediction checksum differs: {stem}.{suffix}")
            _verify_depth_files(staged, rows, camera, factor, manifest["signature"].get("png_overflow", "invalid"))
            # Rebase before committing; errors here leave the old cache intact.
            _associations(staged, rows, current)
            if not target.exists():
                staged.rename(target)
            else:
                # All identities and all incoming files were checked before any
                # mutation. Publish the manifest last, after the complete pairs.
                for stem in rows:
                    for suffix in ("npy", "png"):
                        source = staged / f"{stem}.{suffix}"
                        destination = target / source.name
                        if destination.exists() and digest_file(destination) != rows[stem][f"{suffix}_sha256"]:
                            raise ValueError(f"Existing output has changed: {destination}")
                for stem in rows:
                    for suffix in ("npy", "png"):
                        os.replace(staged / f"{stem}.{suffix}", target / f"{stem}.{suffix}")
                os.replace(staged / "manifest.json", target / "manifest.json")
            _associations(target, rows, current)
    return {"sequence": sequence, "imported_frames": len(rows), "full_sequence_frames": total,
            "full_sequence": len(rows) == total, "pred_root": str(root),
            "association_file": str(target / "associations.txt")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    export = subs.add_parser("export")
    export.add_argument("--sequence", choices=SEQUENCES, required=True)
    export.add_argument("--pred-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    ingest = subs.add_parser("import")
    ingest.add_argument("--archive", type=Path, required=True)
    ingest.add_argument("--pred-root", type=Path, default=ROOT / "depth_maps/da3_metric_large_fp16")
    ingest.add_argument("--data-root", type=Path)
    ingest.add_argument("--expected-sha256", help="Optional SHA256 printed by the Colab export")
    args = parser.parse_args(argv)
    result = (export_cache(args.sequence, args.pred_root, args.output) if args.action == "export"
              else import_cache(args.archive, args.pred_root, args.data_root, args.expected_sha256))
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
