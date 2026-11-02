"""Generate a resumable DA3METRIC-LARGE cache, without using reference depth.

The canonical maps are native-resolution float32 metres. PNG zero denotes an
invalid or unencodable depth; the per-sequence factor is recorded in the cache.
Run --dry-run first: it checks RGB images and calibration without importing
Torch, opening checkpoints, downloading weights, or creating a cache.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from config import ROOT, load_sequence
from depth import max_depth, save_depth
from part2.common import SEQUENCES, find_data_root, load_camera, load_frame_records

DEFAULT_CACHE = ROOT / "depth_maps" / "da3_metric_large_fp16"
SCHEMA_VERSION = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_manifest(path: Path, manifest: dict) -> None:
    manifest["updated_utc"] = _utc()
    _atomic_text(path, json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def write_portable_associations(manifest: dict, path: Path) -> None:
    """Write all committed frames with portable data/... RGB path tokens.

    These tokens are relative to a bundle root, not the cache directory. The
    transfer importer rebases them to local RGB files before pySLAM consumes it.
    """
    lines = ["# Portable bundle-root RGB paths; rebase before use by pySLAM.",
             "# rgb_timestamp rgb_path depth_timestamp predicted_depth_path"]
    for stem, entry in sorted(manifest["frames"].items(), key=lambda row: row[1]["index"]):
        if any(ch.isspace() for ch in entry["rgb"] + stem):
            raise ValueError("Association paths cannot contain whitespace")
        timestamp = float(entry["timestamp"])
        lines.append(f"{timestamp:.9f} {entry['rgb']} {timestamp:.9f} {stem}.png")
    _atomic_text(Path(path), "\n".join(lines) + "\n")


def _camera_record(camera) -> dict:
    values = asdict(camera)
    K = np.asarray(camera.K, dtype=float)
    if (K.shape != (3, 3) or not np.isfinite(K).all()
            or camera.fx <= 0 or camera.fy <= 0
            or camera.width < 1 or camera.height < 1):
        raise ValueError("Camera calibration must be finite with positive focal lengths and size")
    return {**values, "K": K.tolist()}


def _inventory(sequence: str, data_root, stride: int, max_frames):
    if stride < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError("stride and max_frames must be positive")
    root = find_data_root(data_root)
    frames = load_frame_records(sequence, root)
    camera = load_camera(sequence, root)
    calibration = _camera_record(camera)
    records = {}
    for frame in frames:
        # Keep the dataset's logical layout even when a dataset folder is a symlink.
        relative = Path(os.path.abspath(frame.rgb)).relative_to(Path(os.path.abspath(root)))
        rgb = (Path("data") / relative).as_posix()
        if (frame.stem in records or any(c.isspace() for c in rgb)
                or not np.isfinite(frame.timestamp)):
            raise ValueError("RGB names must be unique, whitespace-free, and have finite timestamps")
        records[frame.stem] = {"index": int(frame.index), "timestamp": float(frame.timestamp), "rgb": rgb}
    selected = frames[::stride][:max_frames]
    return frames, selected, camera, calibration, records


def _read_rgb(frame, camera) -> np.ndarray:
    image = cv2.imread(str(frame.rgb), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode RGB image: {frame.rgb}")
    if image.shape[:2] != (camera.height, camera.width):
        raise ValueError(f"{frame.rgb}: RGB dimensions differ from camera calibration")
    return image


def _validate_depth(depth, shape) -> np.ndarray:
    value = np.asarray(depth, dtype=np.float32)
    if value.shape != tuple(shape):
        raise ValueError(f"Prediction shape {value.shape} differs from native RGB {tuple(shape)}")
    if np.any(np.isfinite(value) & (value < 0)):
        raise ValueError("Network returned negative metric depths")
    if not np.any(np.isfinite(value) & (value > 0)):
        raise ValueError("Network returned no valid positive metric depths")
    return value


def _timed_predict(predictor, image, camera):
    predictor.synchronize()
    start = time.perf_counter()
    depth = predictor.predict(image, camera)
    predictor.synchronize()
    seconds = time.perf_counter() - start
    return _validate_depth(depth, image.shape[:2]), seconds


def _check_entry(folder, stem, entry, expected, frame, shape) -> None:
    if any(entry.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{stem}: cached frame identity differs from the current RGB inventory")
    if entry.get("rgb_sha256") != sha256(frame.rgb):
        raise ValueError(f"{stem}: RGB checksum changed; use --overwrite for this selection")
    for suffix in ("npy", "png"):
        path = folder / f"{stem}.{suffix}"
        if not path.is_file() or entry.get(f"{suffix}_sha256") != sha256(path):
            raise ValueError(f"{stem}: {suffix.upper()} checksum differs or output is missing; use --overwrite")
    depth = np.load(folder / f"{stem}.npy", allow_pickle=False, mmap_mode="r")
    if depth.dtype != np.float32 or depth.shape != shape:
        raise ValueError(f"{stem}: cached NPY must be native-resolution float32")
    png = cv2.imread(str(folder / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    if png is None or png.dtype != np.uint16 or png.shape != shape:
        raise ValueError(f"{stem}: cached PNG must be native-resolution uint16")


def predict_sequence_da3(sequence: str, predictor, identity: dict, *, data_root=None,
                         pred_root=DEFAULT_CACHE, stride=1, max_frames=None,
                         overwrite=False, png_overflow="invalid", progress_every=25,
                         session_metadata=None) -> dict:
    """Generate or verify a selection, preserving every previously committed frame.

    ``predictor`` needs predict(BGR, Camera) and synchronize(), making the cache
    lifecycle testable without Torch. ``identity`` must describe a fixed model,
    checkpoint, source revision, precision, device and preprocessing policy.
    Hardware belongs in session_metadata: compatible caches can span GPUs while
    each old frame retains the hardware session that produced its timing.
    """
    if png_overflow not in ("raise", "invalid") or progress_every < 1:
        raise ValueError("Invalid PNG overflow policy or progress interval")
    all_frames, frames, camera, calibration, inventory = _inventory(sequence, data_root, stride, max_frames)
    factor = float(load_sequence(sequence)["depth_factor"])
    max_depth(factor)
    signature = {**identity, "model": "da3_metric_large", "depth_factor": factor,
                 "png_overflow": png_overflow, "camera": calibration,
                 "input_size": [camera.height, camera.width], "rgb_source_root": "data",
                 "full_frame_inventory_sha256": _json_hash(list(inventory.values()))}
    # JSON round-trip also rejects nonportable objects and non-finite identity values.
    signature = json.loads(json.dumps(signature, allow_nan=False))
    folder = Path(pred_root) / sequence
    manifest_path = folder / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if previous is not None:
        if previous.get("schema_version") != SCHEMA_VERSION or previous.get("sequence") != sequence:
            raise ValueError(f"{folder}: incompatible or unidentified manifest; use a new cache")
        if previous.get("signature") != signature:
            raise ValueError(f"{folder}: model, calibration, encoding, or RGB inventory differs; use a separate --pred-root")
    manifest = previous or {"schema_version": SCHEMA_VERSION, "sequence": sequence,
                            "signature": signature, "frames": {}, "sessions": {},
                            "pending": None, "created_utc": _utc()}
    by_stem = {frame.stem: frame for frame in all_frames}
    selected = {frame.stem for frame in frames}
    pending = manifest.get("pending")
    pending_stem = pending.get("stem") if pending else None
    if pending_stem is not None:
        if pending_stem not in selected or pending_stem not in inventory:
            raise ValueError("An interrupted pending frame must be included in the resumed selection")
        if any(pending.get(key) != value for key, value in inventory[pending_stem].items()):
            raise ValueError("Pending transaction does not match the current RGB inventory")
        if not overwrite and pending.get("rgb_sha256") != sha256(by_stem[pending_stem].rgb):
            raise ValueError("Pending frame RGB checksum changed; use --overwrite")
    # Never adopt output files that have no committed entry or pending transaction.
    if folder.exists():
        known = set(manifest["frames"]) | ({pending_stem} if pending_stem else set())
        for suffix in ("npy", "png"):
            for path in folder.glob(f"*.{suffix}"):
                stem = path.name[:-(len(suffix) + 1)]
                if stem not in known and not (pending_stem and stem == pending_stem + ".tmp"):
                    raise ValueError(f"Unidentified prediction {path}; use a new cache directory")
    for stem, entry in manifest["frames"].items():
        if stem not in inventory:
            raise ValueError(f"Cached frame {stem} is outside the full RGB inventory")
        if overwrite and stem in selected:
            continue
        _check_entry(folder, stem, entry, inventory[stem], by_stem[stem], (camera.height, camera.width))
    needed = [frame for frame in frames if overwrite or frame.stem not in manifest["frames"]]
    reused = len(frames) - len(needed)
    manifest.update(full_count=len(all_frames),
                    selection={"stride": stride, "max_frames": max_frames,
                               "requested_count": len(frames), "indices": [int(f.index) for f in frames]},
                    status="running", full_complete=False)
    manifest.pop("failure", None)
    folder.mkdir(parents=True, exist_ok=True)
    _write_manifest(manifest_path, manifest)
    session_id = uuid.uuid4().hex
    session = {"started_utc": _utc(), "runtime": session_metadata or {},
               "warmup_seconds": None, "new_frames": 0,
               "timing_scope": "synchronized preprocessing + network + native metric postprocessing; excludes RGB IO and cache writes"}
    if needed:
        manifest.setdefault("sessions", {})[session_id] = session
    inference_times = []
    pipeline_start = time.perf_counter()
    current_stem = None
    try:
        if needed:
            current_stem = needed[0].stem
            warm_image = _read_rgb(needed[0], camera)
            _, session["warmup_seconds"] = _timed_predict(predictor, warm_image, camera)
            session["warmup_frame"] = current_stem
            _write_manifest(manifest_path, manifest)
            print(json.dumps({"sequence": sequence, "warmup_seconds": session["warmup_seconds"],
                              "excluded_from_frame_timings": True}), flush=True)
            pipeline_start = time.perf_counter()
        for number, frame in enumerate(needed, 1):
            current_stem = frame.stem
            image = _read_rgb(frame, camera)
            rgb_digest = sha256(frame.rgb)
            depth, seconds = _timed_predict(predictor, image, camera)
            # The pending transaction makes an interruption between the two output
            # renames and manifest commit recoverable, without adopting unknown files.
            manifest["pending"] = {"stem": frame.stem, **inventory[frame.stem],
                                   "rgb_sha256": rgb_digest, "session_id": session_id}
            manifest["frames"].pop(frame.stem, None)
            _write_manifest(manifest_path, manifest)
            save_depth(folder / frame.stem, depth, factor, overflow=png_overflow)
            valid = np.isfinite(depth) & (depth > 0)
            entry = {**inventory[frame.stem], "rgb_sha256": rgb_digest,
                     "npy_sha256": sha256(folder / f"{frame.stem}.npy"),
                     "png_sha256": sha256(folder / f"{frame.stem}.png"),
                     "inference_seconds": seconds, "invalid_fraction": float(1 - valid.mean()),
                     "png_overflow_fraction": float(np.mean(valid & (depth > max_depth(factor)))),
                     "depth_min_m": float(depth[valid].min()), "depth_max_m": float(depth[valid].max()),
                     "session_id": session_id}
            preprocessing = getattr(predictor, "last_preprocessing", None)
            if preprocessing:
                entry["preprocessing"] = preprocessing
            manifest["frames"][frame.stem] = entry
            manifest["pending"] = None
            session["new_frames"] += 1
            inference_times.append(seconds)
            _write_manifest(manifest_path, manifest)
            if number == 1 or number % progress_every == 0 or number == len(needed):
                write_portable_associations(manifest, folder / "associations.txt")
                elapsed = time.perf_counter() - pipeline_start
                remaining = len(needed) - number
                print(json.dumps({"sequence": sequence, "completed_requested": reused + number,
                                  "requested": len(frames), "new_frames": number, "reused_frames": reused,
                                  "remaining": remaining, "last_inference_seconds": seconds,
                                  "eta_seconds": elapsed / number * remaining}), flush=True)
        manifest["status"] = "complete"
        manifest["full_complete"] = len(manifest["frames"]) == len(all_frames)
        if needed:
            session["finished_utc"] = _utc()
            session["mean_inference_seconds"] = float(np.mean(inference_times))
        _write_manifest(manifest_path, manifest)
        write_portable_associations(manifest, folder / "associations.txt")
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["full_complete"] = False
        manifest["failure"] = {"frame": current_stem, "error": f"{type(exc).__name__}: {exc}", "utc": _utc()}
        if needed:
            session["finished_utc"] = _utc()
        _write_manifest(manifest_path, manifest)
        write_portable_associations(manifest, folder / "associations.txt")
        raise
    result = {"sequence": sequence, "status": manifest["status"], "full_count": len(all_frames),
              "requested_frames": len(frames), "new_frames": len(inference_times), "reused_frames": reused,
              "cached_frames": len(manifest["frames"]), "full_complete": manifest["full_complete"],
              "mean_new_inference_seconds": float(np.mean(inference_times)) if inference_times else None}
    print(json.dumps(result), flush=True)
    return result


def _model_identity(checkpoint: Path, source_root: Path, *, device: str, precision: str,
                    process_res: int) -> tuple[Path, dict]:
    # These imports do not import Torch. Model construction remains separate.
    from part2.benchmark_models import checkpoint_fingerprint
    from part2.da3_adapter import _checkpoint_directory
    checkpoint = _checkpoint_directory(checkpoint)
    source_root = source_root.expanduser().resolve()
    if not (source_root / "src/depth_anything_3/api.py").is_file():
        raise FileNotFoundError(f"Official DA3 source missing: {source_root}")
    revision = subprocess.run(["git", "-C", str(source_root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    files = {path.relative_to(source_root).as_posix(): sha256(path)
             for path in sorted((source_root / "src/depth_anything_3").rglob("*.py"))}
    identity = {"model": "da3_metric_large", "model_variant": "da3metric-large",
                "checkpoint": checkpoint_fingerprint(checkpoint), "source_commit": revision,
                "source_files_sha256": _json_hash(files),
                "adapter_sha256": sha256(Path(__file__).with_name("da3_adapter.py")),
                "device": device, "precision": precision, "process_res": process_res,
                "process_res_method": "upper_bound_resize", "input_color": "RGB",
                "metric_conversion": "canonical_depth * mean(processed_fx, processed_fy) / 300",
                "native_output_interpolation": "bilinear_after_metric_conversion"}
    return checkpoint, identity


def _runtime_metadata(predictor) -> dict:
    torch = predictor.torch
    record = {"python": platform.python_version(), "platform": platform.platform(),
              "torch": torch.__version__, "torch_cuda": torch.version.cuda,
              "device": str(predictor.device), "cpu_threads": torch.get_num_threads()}
    if predictor.device.type == "cuda":
        properties = torch.cuda.get_device_properties(predictor.device)
        record.update(gpu_name=properties.name, gpu_memory_bytes=properties.total_memory,
                      cudnn_version=torch.backends.cudnn.version())
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", action="append", choices=SEQUENCES)
    parser.add_argument("--all", action="store_true", help="All common sequences (the default)")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--pred-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/DA3METRIC-LARGE")
    parser.add_argument("--source-root", type=Path, default=ROOT / "weights/Depth-Anything-3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--png-overflow", choices=("invalid", "raise"), default="invalid")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate all RGB inputs and calibration without Torch or weights")
    args = parser.parse_args(argv)
    if args.stride < 1 or (args.max_frames is not None and args.max_frames < 1) or args.process_res < 14:
        parser.error("stride/max-frames must be positive and process-res must be at least 14")
    sequences = list(SEQUENCES) if args.all or not args.sequence else list(dict.fromkeys(args.sequence))
    if args.dry_run:
        for sequence in sequences:
            all_frames, frames, camera, calibration, _ = _inventory(sequence, args.data_root, args.stride, args.max_frames)
            for frame in all_frames:
                _read_rgb(frame, camera)
            print(json.dumps({"sequence": sequence, "full_count": len(all_frames),
                              "requested_frames": len(frames), "camera": calibration,
                              "png_cap_m": max_depth(float(load_sequence(sequence)["depth_factor"])),
                              "dry_run": True}), flush=True)
        return 0
    try:
        checkpoint, identity = _model_identity(args.checkpoint, args.source_root, device=args.device,
                                               precision=args.precision, process_res=args.process_res)
        from part2.da3_adapter import DA3Predictor
        predictor = DA3Predictor(checkpoint, args.device, args.source_root,
                                 precision=args.precision, process_res=args.process_res)
        identity.update(predictor.metadata)
        runtime = _runtime_metadata(predictor)
        for sequence in sequences:
            predict_sequence_da3(sequence, predictor, identity, data_root=args.data_root,
                                 pred_root=args.pred_root, stride=args.stride, max_frames=args.max_frames,
                                 overwrite=args.overwrite, png_overflow=args.png_overflow,
                                 session_metadata=runtime)
    except Exception as exc:
        print(f"DA3 cache generation stopped: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
