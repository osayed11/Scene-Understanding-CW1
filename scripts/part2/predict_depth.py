"""Cache one fixed Depth Pro checkpoint for all sequences (float32 metres + PNG).

The --dry-run inventory needs only the evaluation dependencies. Actual inference
uses the official depth_pro package in the course depth/depth-cpu environment;
it does not import the SLAM core or use ground truth. No weights are downloaded
implicitly. First benchmark e.g. --sequence replica_room0 --max-frames 3.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from config import ROOT, PYSLAM, load_sequence
from depth import max_depth, save_depth
from part2.common import SEQUENCES, load_camera, load_frame_records


class DepthProPredictor:
    """Adapter for Apple's public metric-depth API; RGB input, calibrated focal length."""

    def __init__(self, checkpoint: Path, device: str, source_root: Path | None = None):
        if source_root is not None:
            sys.path.insert(0, str(source_root / "src"))
        try:
            import torch
            import depth_pro
            from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        except ImportError as exc:
            raise RuntimeError(
                "Depth Pro dependencies are unavailable. Run this script in the course "
                "depth-cpu environment (depth on a GPU machine), after models-depth installation."
            ) from exc
        self.torch = torch
        self.device = torch.device(device)
        cfg = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(checkpoint))
        self.model, self.transform = depth_pro.create_model_and_transforms(
            config=cfg, device=self.device, precision=torch.float32)
        self.model.eval()

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            self.torch.mps.synchronize()

    def predict(self, bgr: np.ndarray, focal_px: float) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with self.torch.inference_mode():
            image = self.transform(rgb)
            focal = self.torch.tensor(float(focal_px), device=self.device)
            output = self.model.infer(image, f_px=focal)["depth"]
        return output.detach().float().cpu().numpy().squeeze()


def checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(path: Path, manifest: dict):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def predict_sequence(sequence: str, predictor, identity: dict, *, data_root=None,
                     pred_root=ROOT / "depth_maps", stride=1, max_frames=None,
                     overwrite=False, png_overflow="invalid") -> dict:
    if stride < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError("stride and max_frames must be positive")
    all_frames = load_frame_records(sequence, data_root)
    frames = all_frames[::stride]
    if max_frames is not None:
        frames = frames[:max_frames]
    camera = load_camera(sequence, data_root)
    cfg = load_sequence(sequence)
    factor = float(cfg["depth_factor"])
    folder = Path(pred_root) / sequence
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    signature = {**identity, "depth_factor": factor, "png_overflow": png_overflow,
                 "focal_px": camera.fx, "input_size": [camera.height, camera.width],
                 "rgb_source_root": str(all_frames[0].rgb.parent.resolve())}
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and previous["signature"] != signature:
        raise ValueError(f"{folder}: different model/encoding already cached; use a separate --pred-root")
    manifest = previous or {"schema_version": 1, "sequence": sequence, "signature": signature, "frames": {}}
    elapsed_times = []
    for frame in frames:
        npy, png = folder / f"{frame.stem}.npy", folder / f"{frame.stem}.png"
        rgb_digest = checkpoint_hash(frame.rgb)
        if not overwrite and npy.exists() and png.exists() and frame.stem in manifest["frames"]:
            entry = manifest["frames"][frame.stem]
            if (entry.get("rgb_sha256") != rgb_digest or entry["timestamp"] != frame.timestamp
                    or entry["index"] != frame.index):
                raise ValueError(f"{frame.rgb}: cached RGB content/timestamp differs; use --overwrite or a new cache")
            continue
        if not previous and not overwrite and (npy.exists() or png.exists()):
            raise ValueError(f"Unidentified existing prediction {npy}; use a new cache folder or --overwrite")
        image = cv2.imread(str(frame.rgb), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(frame.rgb)
        if image.shape[:2] != (camera.height, camera.width):
            raise ValueError(f"{frame.rgb}: image size does not match calibration; rescale intrinsics explicitly")
        predictor.synchronize()
        start = time.perf_counter()
        depth = np.asarray(predictor.predict(image, camera.fx), dtype=np.float32)
        predictor.synchronize()
        seconds = time.perf_counter() - start
        if depth.shape != image.shape[:2]:
            raise ValueError(f"{sequence}/{frame.stem}: prediction shape {depth.shape} differs from RGB")
        if np.any(np.isfinite(depth) & (depth < 0)):
            raise ValueError(f"{sequence}/{frame.stem}: network returned negative depths")
        if not np.any(np.isfinite(depth) & (depth > 0)):
            raise ValueError(f"{sequence}/{frame.stem}: network returned no valid depths")
        save_depth(folder / frame.stem, depth, factor, overflow=png_overflow)
        valid = np.isfinite(depth) & (depth > 0)
        manifest["frames"][frame.stem] = {
            "index": frame.index, "timestamp": frame.timestamp, "rgb": str(frame.rgb.resolve()),
            "rgb_sha256": rgb_digest,
            "inference_seconds": seconds, "invalid_fraction": float(1 - valid.mean()),
            "png_overflow_fraction": float(np.mean(valid & (depth > max_depth(factor)))),
            "depth_min_m": float(depth[valid].min()), "depth_max_m": float(depth[valid].max()),
        }
        _write_manifest(manifest_path, manifest)
        elapsed_times.append(seconds)
        print(f"{sequence} {frame.index:05d} {frame.stem}: {seconds:.3f} s", flush=True)
    # The course TUM reader concatenates its sequence base with path tokens.
    # Both tokens must therefore be relative to this cache sequence directory.
    selected_stems = {frame.stem for frame in frames}
    entries = sorted(((stem, entry) for stem, entry in manifest["frames"].items()
                      if stem in selected_stems), key=lambda row: row[1]["index"])
    current_frames = {frame.stem: frame for frame in all_frames}
    with (folder / "associations.txt").open("w") as handle:
        handle.write("# rgb_timestamp rgb_path depth_timestamp predicted_depth_path\n")
        for stem, entry in entries:
            if stem not in current_frames:
                raise ValueError(f"Cached frame {stem} does not belong to the current input sequence")
            rgb_path = Path(os.path.relpath(current_frames[stem].rgb.resolve(), folder.resolve())).as_posix()
            if any(ch.isspace() for ch in rgb_path + stem):
                raise ValueError("TUM associations cannot encode paths containing whitespace; use a path without spaces")
            handle.write(f"{entry['timestamp']:.9f} {rgb_path} {entry['timestamp']:.9f} "
                         f"{stem}.png\n")
    result = {"sequence": sequence, "requested_frames": len(frames), "new_frames": len(elapsed_times),
              "cached_frames": len(manifest["frames"]), "association_frames": len(entries),
              "mean_new_inference_seconds": float(np.mean(elapsed_times)) if elapsed_times else None}
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", action="append", choices=SEQUENCES)
    parser.add_argument("--all", action="store_true", help="All three common sequences (also the default)")
    parser.add_argument("--data-root")
    parser.add_argument("--pred-root", type=Path, default=ROOT / "depth_maps")
    parser.add_argument("--checkpoint", type=Path, help="Depth Pro depth_pro.pt; never downloaded implicitly")
    parser.add_argument("--pyslam-root", type=Path,
                        default=Path(os.environ.get("CW1_PYSLAM_ROOT", str(PYSLAM))))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--png-overflow", choices=("raise", "invalid"), default="invalid")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Inventory inputs without loading the network")
    args = parser.parse_args()
    os.environ["CW1_PYSLAM_ROOT"] = str(args.pyslam_root)
    if args.stride < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("--stride and --max-frames must be positive")
    sequences = SEQUENCES if args.all or not args.sequence else args.sequence
    if args.dry_run:
        for sequence in sequences:
            frames = load_frame_records(sequence, args.data_root)
            camera = load_camera(sequence, args.data_root)
            requested = len(frames[::args.stride][:args.max_frames])
            print(json.dumps({"sequence": sequence, "rgb_frames": len(frames),
                              "depth_gt_matches": sum(f.depth_gt is not None for f in frames),
                              "requested_frames": requested, "size": [camera.width, camera.height],
                              "png_cap_m": max_depth(load_sequence(sequence)["depth_factor"])}))
        return
    source_root = args.pyslam_root / "thirdparty" / "ml_depth_pro"
    checkpoint = args.checkpoint or source_root / "checkpoints" / "depth_pro.pt"
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}. Install models-depth in the course depth-cpu "
                     "environment, or pass --checkpoint. Use --dry-run to check input data first.")
    identity = {"model": "depth_pro", "checkpoint_sha256": checkpoint_hash(checkpoint),
                "device": args.device, "precision": "float32", "input_color": "RGB"}
    predictor = DepthProPredictor(checkpoint.resolve(), args.device, source_root if source_root.is_dir() else None)
    for sequence in sequences:
        predict_sequence(sequence, predictor, identity, data_root=args.data_root, pred_root=args.pred_root,
                         stride=args.stride, max_frames=args.max_frames, overwrite=args.overwrite,
                         png_overflow=args.png_overflow)


if __name__ == "__main__":
    main()
