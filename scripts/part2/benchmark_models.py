"""Compare fixed metric-depth models on identical deterministic frame samples.

Inference timings include preprocessing, model execution, postprocessing and device
synchronization; exclude image reads, saving and GT evaluation. One warmup frame
is timed separately. CPU/GPU/precision/hardware configurations never share rows.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from config import ROOT, load_sequence
from part2.common import Camera, FrameRecord, SEQUENCES, load_camera, load_frame_records
from part2.eval_depth import depth_metrics

MODELS = ("depth_pro", "dav2_small", "dav2_base", "dav2_large", "da3_metric_large")


def select_indices(n_frames: int, samples: int = 3, indices=None) -> list[int]:
    if n_frames < 1 or samples < 1:
        raise ValueError("frame and sample counts must be positive")
    if indices is not None:
        selected = list(indices)
        if not selected or len(set(selected)) != len(selected) or any(i < 0 or i >= n_frames for i in selected):
            raise ValueError("indices must be distinct valid frame indices")
        return selected
    return sorted(set(np.rint(np.linspace(.1, .9, samples) * (n_frames - 1)).astype(int).tolist()))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_fingerprint(path: Path) -> dict:
    """Fingerprint a weight file or DA3's config-plus-safetensors directory.

    Directory identity covers only the two files actually needed to construct
    the model. README files and download/cache metadata cannot change it.
    """
    path = Path(path)
    if path.is_dir():
        files = {name: {"sha256": sha256(path / name), "bytes": (path / name).stat().st_size}
                 for name in sorted(("config.json", "model.safetensors"))}
        combined = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return {"kind": "directory", "sha256": combined, "files": files}
    digest = sha256(path)
    return {"kind": "file", "sha256": digest,
            "files": {path.name: {"sha256": digest, "bytes": path.stat().st_size}}}


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def timed_prediction(predictor, image, camera, clock=time.perf_counter):
    predictor.synchronize()
    start = clock()
    prediction = predictor.predict(image, camera)
    predictor.synchronize()
    elapsed = clock() - start
    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.ndim != 2 or prediction.shape != image.shape[:2]:
        raise ValueError(f"prediction {prediction.shape} does not match RGB {image.shape[:2]}")
    if np.any(np.isfinite(prediction) & (prediction < 0)) or not np.any(np.isfinite(prediction) & (prediction > 0)):
        raise ValueError("prediction has negative depth or no finite positive depth")
    return prediction, elapsed


def aggregate(rows: list[dict]) -> dict:
    successful = [r for r in rows if r.get("status") == "ok"]
    timed = [r for r in rows if r.get("inference_seconds") is not None]
    result = {"frames_requested": len(rows), "frames_ok": len(successful),
              "frames_failed": sum(r.get("status") == "failed" for r in rows),
              "frames_pending": sum(r.get("status") == "pending" for r in rows),
              "frames_no_gt": sum(r.get("gt_status") == "no_ground_truth" for r in successful),
              "frames_with_gt": sum(r.get("gt_status") == "ok" for r in successful),
              "mean_inference_seconds": (float(np.mean([r["inference_seconds"] for r in timed])) if timed else None),
              "aggregation": "mean_per_frame"}
    for comparison in ("raw", "median_scaled"):
        metrics = [r[comparison] for r in successful if r.get(comparison) is not None]
        result[comparison] = {}
        for key in ("abs_rel", "rmse_m", "delta1", "scale", "prediction_coverage"):
            values = [m[key] for m in metrics if m.get(key) is not None and np.isfinite(m[key])]
            result[comparison][key] = float(np.mean(values)) if values else None
    return result


def load_samples(sequences, samples=3, indices=None, data_root=None, sample_manifest=None):
    selected = []
    if sample_manifest is not None:
        path = Path(sample_manifest).resolve()
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = rows["samples"]
        for row in rows:
            if row["sequence"] not in sequences:
                continue
            def portable(value):
                if value is None:
                    return None
                resolved = (path.parent / value).resolve()
                if not resolved.is_relative_to(path.parent):
                    raise ValueError("sample manifest paths must stay within its bundle directory")
                return resolved
            frame = FrameRecord(int(row["index"]), float(row["timestamp"]), portable(row["rgb"]),
                                portable(row.get("depth_gt")), row.get("depth_timestamp"))
            camera = Camera(*(float(row[k]) for k in ("fx", "fy", "cx", "cy")),
                            int(row["width"]), int(row["height"]))
            selected.append({"sequence": row["sequence"], "frame": frame, "camera": camera,
                             "gt_depth_factor": row.get("gt_depth_factor")})
    else:
        for sequence in sequences:
            frames = load_frame_records(sequence, data_root)
            camera = load_camera(sequence, data_root)
            cfg = load_sequence(sequence)
            for i in select_indices(len(frames), samples, indices):
                selected.append({"sequence": sequence, "frame": frames[i], "camera": camera,
                                 "gt_depth_factor": cfg.get("gt_depth_factor", cfg.get("depth_factor"))})
    if not selected:
        raise ValueError("no samples selected")
    keys = [(s["sequence"], s["frame"].index) for s in selected]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate sequence/frame in sample selection")
    return selected


def _runtime_metadata(device):
    # Import only after CLI parsing and actual model creation, never for --help.
    import torch
    cpu = platform.processor()
    if not cpu and Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), "unknown")
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "opencv": cv2.__version__, "device": device, "cpu": cpu,
            "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(),
            "gpu": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else None,
            "platform": platform.platform()}


def run_benchmark(predictor, samples, *, signature: dict, output: Path, pred_root: Path,
                  setup_seconds: float = 0, overwrite: bool = False) -> dict:
    """Run/checkpoint one configuration; injectable predictor supports lightweight tests."""
    if Path(pred_root).resolve() == (ROOT / "depth_maps").resolve():
        raise ValueError("use a separate prediction root to preserve the baseline depth cache")
    signature = _clean(signature)
    config_hash = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    previous = json.loads(output.read_text()) if output.exists() and not overwrite else None
    if previous and previous.get("config_sha256") != config_hash:
        raise ValueError("existing benchmark configuration differs; use another output/pred-root or --overwrite")
    result = previous or {"schema_version": 1, "benchmark_kind": "spread_frame_model_comparison",
                          "signature": signature, "config_sha256": config_hash,
                          "setup_seconds": setup_seconds, "warmup": None, "frames": []}
    old_rows = {(r["sequence"], r["index"]): r for r in result["frames"]}
    result["status"] = "running"
    # Preserve every completed row while checkpointing later work. Clearing this
    # list first would lose still-unvisited cached rows if a resumed run stopped.
    result["frames"] = [old_rows.get((s["sequence"], s["frame"].index),
        {"sequence": s["sequence"], "index": s["frame"].index, "status": "pending",
         "inference_seconds": None}) for s in samples]
    warmed_up = False
    for position, sample in enumerate(samples):
        frame, camera, sequence = sample["frame"], sample["camera"], sample["sequence"]
        target = Path(pred_root) / sequence / f"{frame.stem}.npy"
        row = {"sequence": sequence, "index": frame.index, "timestamp": frame.timestamp,
               "stem": frame.stem, "rgb": str(frame.rgb), "camera": asdict(camera),
               "prediction": str(target), "status": "pending", "inference_seconds": None,
               "raw": None, "median_scaled": None, "gt_depth_factor": sample["gt_depth_factor"],
               "depth_timestamp": frame.depth_timestamp}
        try:
            row["rgb_sha256"] = sha256(frame.rgb)
            row["depth_gt_sha256"] = sha256(frame.depth_gt) if frame.depth_gt is not None else None
            old = old_rows.get((sequence, frame.index))
            if old and old.get("status") == "ok" and target.is_file():
                if any(old.get(k) != row[k] for k in ("rgb_sha256", "camera", "timestamp",
                                                    "depth_gt_sha256", "gt_depth_factor", "depth_timestamp")):
                    raise ValueError("cached frame input/calibration changed; use --overwrite")
                if old.get("prediction_sha256") != sha256(target):
                    raise ValueError("cached prediction content changed; use --overwrite")
                cached = np.load(target, allow_pickle=False, mmap_mode="r")
                if cached.shape != (camera.height, camera.width):
                    raise ValueError("cached prediction shape differs from calibration")
                continue
            image = cv2.imread(str(frame.rgb), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(frame.rgb)
            if image.shape[:2] != (camera.height, camera.width):
                raise ValueError("RGB dimensions differ from calibration")
            if not warmed_up:
                tick = time.perf_counter()
                try:
                    _, warm_seconds = timed_prediction(predictor, image, camera)
                    warmup = {"status": "ok", "inference_seconds": warm_seconds,
                              "sequence": sequence, "index": frame.index}
                except Exception as exc:
                    warmup = {"status": "failed", "wall_seconds": time.perf_counter() - tick,
                              "sequence": sequence, "index": frame.index, "error": str(exc)}
                result.setdefault("warmup_sessions", []).append(warmup)
                if result["warmup"] is None:
                    result["warmup"] = warmup
                warmed_up = warmup["status"] == "ok"
                if warmup["status"] != "ok":
                    raise RuntimeError("warmup failed: " + warmup["error"])
            prediction, row["inference_seconds"] = timed_prediction(predictor, image, camera)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp.npy")
            np.save(temporary, prediction, allow_pickle=False)
            temporary.replace(target)
            row["prediction_sha256"] = sha256(target)
            if frame.depth_gt is None:
                row["gt_status"] = "no_ground_truth"
            else:
                raw_gt = cv2.imread(str(frame.depth_gt), cv2.IMREAD_UNCHANGED)
                if raw_gt is None:
                    raise FileNotFoundError(frame.depth_gt)
                factor = sample["gt_depth_factor"]
                if factor is None or not np.isfinite(factor) or factor <= 0:
                    raise ValueError("GT depth factor must be finite and positive")
                gt = raw_gt.astype(np.float32) / factor
                gt[raw_gt == 0] = np.nan
                row["raw"] = depth_metrics(prediction, gt)
                row["median_scaled"] = depth_metrics(prediction, gt, median_scale=True)
                row["gt_status"] = "ok" if row["raw"]["valid_pixels"] else "no_valid_pixels"
            row["status"] = "ok"
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
        result["frames"][position] = _clean(row)
        result["summaries"] = {seq: aggregate([r for r in result["frames"] if r["sequence"] == seq])
                               for seq in dict.fromkeys(s["sequence"] for s in samples)}
        result["summaries"]["overall"] = aggregate(result["frames"])
        write_json(output, result)
        print(json.dumps(_clean(row)), flush=True)
    result["status"] = "complete" if all(r["status"] == "ok" for r in result["frames"]) else "completed_with_failures"
    result["summaries"] = {seq: aggregate([r for r in result["frames"] if r["sequence"] == seq])
                           for seq in dict.fromkeys(s["sequence"] for s in samples)}
    result["summaries"]["overall"] = aggregate(result["frames"])
    write_json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--sequence", action="append", choices=(*SEQUENCES, "all"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--domain", choices=("hypersim", "vkitti"), default="hypersim")
    parser.add_argument("--process-res", type=int)
    parser.add_argument("--samples-per-sequence", type=int, default=3)
    parser.add_argument("--indices", nargs="+", help="one sequence only; e.g. 0,1,2 or 0 1 2")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--sample-manifest", type=Path, help="exact portable samples and calibration")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pred-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    sequences = list(SEQUENCES) if not args.sequence or "all" in args.sequence else list(dict.fromkeys(args.sequence))
    indices = [int(i) for group in args.indices for i in group.split(",")] if args.indices else None
    if args.samples_per_sequence < 1 or (args.process_res is not None and args.process_res < 1):
        parser.error("sample count and processing resolution must be positive")
    if indices is not None and (len(sequences) != 1 or args.sample_manifest is not None):
        parser.error("--indices requires one sequence and cannot be combined with --sample-manifest")
    if args.model.startswith("dav2_"):
        if args.domain == "hypersim" and "kitti06" in sequences:
            parser.error("DAV2 Hypersim is indoor: select tum_fr3/replica_room0; benchmark KITTI separately with --domain vkitti")
        if args.domain == "vkitti" and any(s != "kitti06" for s in sequences):
            parser.error("DAV2 VKITTI is outdoor: select kitti06; benchmark indoor sequences separately with --domain hypersim")
    model_case = f"{args.model}_{args.domain}" if args.model.startswith("dav2_") else args.model
    stem = f"{model_case}_{args.device.replace(':', '_')}_{args.precision}"
    output = args.output or ROOT / "results/tables/model_benchmarks" / f"{stem}.json"
    pred_root = args.pred_root or ROOT / "depth_maps/model_benchmarks" / stem
    samples = load_samples(sequences, args.samples_per_sequence, indices, args.data_root, args.sample_manifest)
    fingerprint = checkpoint_fingerprint(args.checkpoint)
    signature = {"model": args.model, "checkpoint_sha256": fingerprint["sha256"],
                 "checkpoint": fingerprint,
                 "device": args.device, "precision": args.precision, "domain": args.domain,
                 "process_res": args.process_res, "selection": [{"sequence": s["sequence"], "index": s["frame"].index}
                                                               for s in samples]}
    try:
        signature["source_git_commit"] = subprocess.check_output(
            ["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        signature["source_git_commit"] = None
    started = time.perf_counter()
    try:
        from part2.model_adapters import create_predictor
        predictor = create_predictor(args.model, args.checkpoint, args.source_root, device=args.device,
                                     precision=args.precision, domain=args.domain, process_res=args.process_res)
        predictor.synchronize()
        setup = time.perf_counter() - started
        signature["adapter"] = predictor.metadata
        signature["runtime"] = _runtime_metadata(args.device)
    except Exception as exc:
        failure = {"schema_version": 1, "status": "model_initialization_failed", "signature": signature,
                   "setup_seconds": time.perf_counter() - started, "error": f"{type(exc).__name__}: {exc}"}
        # A failed attempted initialization must not overwrite an existing valid result.
        write_json(output.with_name(output.stem + "_init_failure.json"), failure)
        print(json.dumps(failure), file=sys.stderr)
        return 1
    result = run_benchmark(predictor, samples, signature=signature, output=output, pred_root=pred_root,
                           setup_seconds=setup, overwrite=args.overwrite)
    print(json.dumps(result["summaries"]), flush=True)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
