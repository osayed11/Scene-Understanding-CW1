"""Offline depth at ORB diagnostic keypoints and robust map-scale estimation.

These OpenCV ORB detections sample textured pixels. They are not the keypoints
of a saved pySLAM run, which may use another detector. Local-patch sigma is a
heuristic dispersion, not calibrated network error. Optional --map-samples pairs
learned metric depth with triangulated map depth for the SAME observations.
Ground truth is never used to estimate map scale.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT  # noqa: E402
from depth import depth_at_keypoints  # noqa: E402

DEFAULT_SEQUENCES = ("kitti06", "tum_fr3", "replica_room0")
KEYPOINT_SOURCE = "opencv_ORB_diagnostic"


def estimate_map_scale(
    learned_depth_m: np.ndarray,
    triangulated_depth_map: np.ndarray,
    sigma_m: np.ndarray | None = None,
    parallax_deg: np.ndarray | None = None,
    *,
    min_parallax_deg: float = 1.0,
    max_relative_sigma: float = 0.25,
    mad_multiplier: float = 3.5,
    min_samples: int = 5,
) -> dict:
    """Estimate map units/metre from corresponding camera-z depths without GT.

Positive finite observations pass optional uncertainty/parallax gates, then
median/MAD ratio rejection. Zero MAD keeps only ratios numerically equal to the
median. Insufficient support returns NaN with an explicit status. The input pair
must refer to the same camera/pixel; a global scale cannot repair scale drift.
"""
    if min_samples < 1 or min_parallax_deg < 0 or max_relative_sigma < 0 or mad_multiplier <= 0:
        raise ValueError("invalid map-scale gate or minimum support")
    learned = np.asarray(learned_depth_m, dtype=np.float64)
    triangulated = np.asarray(triangulated_depth_map, dtype=np.float64)
    if learned.shape != triangulated.shape:
        raise ValueError("learned and triangulated depth arrays must have the same shape")
    valid = np.isfinite(learned) & (learned > 0) & np.isfinite(triangulated) & (triangulated > 0)
    if sigma_m is not None:
        sigma = np.asarray(sigma_m, dtype=np.float64)
        if sigma.shape != learned.shape:
            raise ValueError("sigma_m must match the paired depth arrays")
        valid &= np.isfinite(sigma) & (sigma >= 0) & (sigma <= max_relative_sigma * learned)
    if parallax_deg is not None:
        parallax = np.asarray(parallax_deg, dtype=np.float64)
        if parallax.shape != learned.shape:
            raise ValueError("parallax_deg must match the paired depth arrays")
        valid &= np.isfinite(parallax) & (parallax >= min_parallax_deg)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ratios = triangulated[valid] / learned[valid]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    result = {
        "map_scale_status": "unavailable_insufficient_support",
        "scale_map_units_per_metre": float("nan"),
        "scale_metres_per_map_unit": float("nan"),
        "n_scale_input": int(learned.size), "n_scale_candidates": int(ratios.size),
        "n_scale_support": 0, "scale_ratio_mad": float("nan"),
        "scale_parallax_gate_applied": parallax_deg is not None,
        "scale_uncertainty_gate_applied": sigma_m is not None,
        "scale_min_parallax_deg": min_parallax_deg,
        "scale_max_relative_sigma": max_relative_sigma,
    }
    if not ratios.size:
        return result
    median = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios - median)))
    cutoff = max(mad_multiplier * 1.4826 * mad, abs(median) * 1e-6)
    support = ratios[np.abs(ratios - median) <= cutoff]
    result["n_scale_support"] = int(support.size)
    if support.size:
        result["scale_ratio_mad"] = float(np.median(np.abs(support - np.median(support))))
    if support.size < min_samples:
        return result
    scale = float(np.median(support))
    result.update(map_scale_status="estimated_from_paired_map_depths",
                  scale_map_units_per_metre=scale, scale_metres_per_map_unit=1.0 / scale)
    return result


def _map_scale_from_file(path: Path | None, args: argparse.Namespace) -> dict:
    if path is None:
        return {"map_scale_status": "unavailable_no_map_samples",
                "scale_map_units_per_metre": float("nan"),
                "scale_metres_per_map_unit": float("nan"),
                "n_scale_input": 0, "n_scale_candidates": 0, "n_scale_support": 0}
    with np.load(path, allow_pickle=False) as samples:
        required = ("learned_depth_m", "triangulated_depth_map")
        missing = [name for name in required if name not in samples]
        if missing:
            raise ValueError(f"{path}: missing map sample arrays {missing}")
        return estimate_map_scale(
            samples[required[0]], samples[required[1]],
            sigma_m=samples["sigma_m"] if "sigma_m" in samples else None,
            parallax_deg=samples["parallax_deg"] if "parallax_deg" in samples else None,
            min_parallax_deg=args.min_parallax_deg,
            max_relative_sigma=args.max_relative_sigma,
            min_samples=args.min_scale_samples,
        )


def analyze_sequence(sequence: str, args: argparse.Namespace) -> dict:
    from part2.common import load_depth_gt, load_frame_records, load_prediction

    records = load_frame_records(sequence, args.data_root)[::args.stride]
    if args.max_frames is not None:
        records = records[:args.max_frames]
    if not records:
        raise ValueError(f"{sequence}: no selected RGB frames")
    out_dir = args.keypoints_root / sequence
    out_dir.mkdir(parents=True, exist_ok=True)
    orb = cv2.ORB_create(nfeatures=args.nfeatures)
    all_sigma, all_errors, all_gt, all_ratios, paired_sigma = [], [], [], [], []
    n_keypoints = n_valid_depth = n_gt_frames = 0
    for frame in records:
        gray = cv2.imread(str(frame.rgb), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"cannot read RGB image: {frame.rgb}")
        prediction = np.asarray(load_prediction(args.pred_root, sequence, frame))
        if prediction.shape != gray.shape:
            raise ValueError(f"{frame.rgb}: prediction {prediction.shape} differs from RGB {gray.shape}")
        uv = np.asarray([kp.pt for kp in orb.detect(gray, None)], dtype=np.float32).reshape(-1, 2)
        depth_m, sigma_m = depth_at_keypoints(prediction, uv, radius=args.radius)
        np.savez_compressed(out_dir / f"{frame.stem}.npz", uv=uv, depth_m=depth_m,
                            sigma_m=sigma_m, timestamp=frame.timestamp,
                            frame_index=frame.index, keypoint_source=KEYPOINT_SOURCE)
        valid = np.isfinite(depth_m) & (depth_m > 0)
        n_keypoints += len(uv)
        n_valid_depth += int(valid.sum())
        all_sigma.append(sigma_m[valid & np.isfinite(sigma_m)])
        gt = load_depth_gt(sequence, frame, args.data_root)
        if gt is None:
            continue
        n_gt_frames += 1
        if gt.shape != gray.shape:
            raise ValueError(f"{frame.rgb}: GT depth {gt.shape} differs from RGB {gray.shape}")
        gt_depth, _ = depth_at_keypoints(gt, uv, radius=0)
        paired = valid & np.isfinite(gt_depth) & (gt_depth > 0)
        pred_values = depth_m[paired].astype(np.float64)
        gt_values = gt_depth[paired].astype(np.float64)
        all_errors.append(pred_values - gt_values)
        all_gt.append(gt_values)
        all_ratios.append(np.maximum(pred_values / gt_values, gt_values / pred_values))
        paired_sigma.append(sigma_m[paired])
    errors = np.concatenate(all_errors) if all_errors else np.empty(0)
    gt_depths = np.concatenate(all_gt) if all_gt else np.empty(0)
    ratios = np.concatenate(all_ratios) if all_ratios else np.empty(0)
    sigmas = np.concatenate(all_sigma) if all_sigma else np.empty(0)
    error_sigmas = np.concatenate(paired_sigma) if paired_sigma else np.empty(0)
    row = {
        "sequence": sequence, "keypoint_source": KEYPOINT_SOURCE,
        "matches_saved_slam_keypoints": False,
        "uncertainty_kind": "local_patch_dispersion_heuristic",
        "n_frames": len(records), "stride": args.stride,
        "nfeatures_requested": args.nfeatures, "patch_radius": args.radius,
        "n_keypoints": n_keypoints, "n_valid_depth": n_valid_depth,
        "valid_depth_fraction": n_valid_depth / n_keypoints if n_keypoints else float("nan"),
        "median_sigma_m": float(np.median(sigmas)) if sigmas.size else float("nan"),
        "n_gt_frames": n_gt_frames, "n_keypoints_with_gt": int(errors.size),
        "gt_status": "available" if errors.size else "unavailable_or_no_valid_pairs",
        "keypoint_rmse_m": float(np.sqrt(np.mean(errors ** 2))) if errors.size else float("nan"),
        "keypoint_absrel": float(np.mean(np.abs(errors) / gt_depths)) if errors.size else float("nan"),
        "keypoint_delta_1_25": float(np.mean(ratios < 1.25)) if errors.size else float("nan"),
    }
    finite_sigma = np.isfinite(error_sigmas) & (error_sigmas >= 0)
    row["heuristic_sigma_coverage_1x"] = float(np.mean(np.abs(errors[finite_sigma]) <= error_sigmas[finite_sigma])) if finite_sigma.any() else float("nan")
    row["sigma_abs_error_correlation"] = float(np.corrcoef(error_sigmas[finite_sigma], np.abs(errors[finite_sigma]))[0, 1]) if finite_sigma.sum() >= 3 and np.std(error_sigmas[finite_sigma]) > 0 and np.std(np.abs(errors[finite_sigma])) > 0 else float("nan")
    row.update(_map_scale_from_file(args.map_samples, args))
    return row


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    selection = ap.add_mutually_exclusive_group(required=True)
    selection.add_argument("--sequence")
    selection.add_argument("--all", action="store_true", help="the three coursework datasets")
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--pred-root", type=Path, default=ROOT / "depth_maps")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--nfeatures", type=int, default=2000)
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--keypoints-root", type=Path, default=ROOT / "results/keypoints")
    ap.add_argument("--output", type=Path, default=ROOT / "results/tables/table2_keypoints.csv")
    ap.add_argument("--map-samples", type=Path, help="NPZ: corresponding learned_depth_m, triangulated_depth_map; optional sigma_m/parallax_deg")
    ap.add_argument("--min-parallax-deg", type=float, default=1.0)
    ap.add_argument("--max-relative-sigma", type=float, default=0.25)
    ap.add_argument("--min-scale-samples", type=int, default=5)
    args = ap.parse_args(argv)
    if args.stride < 1 or (args.max_frames is not None and args.max_frames < 1) or args.nfeatures < 1 or args.radius < 0:
        ap.error("stride, max-frames and nfeatures must be positive; radius must be nonnegative")
    if args.all and args.map_samples:
        ap.error("--map-samples must belong to one sequence/run; use --sequence")
    sequences = DEFAULT_SEQUENCES if args.all else (args.sequence,)
    rows = [analyze_sequence(sequence, args) for sequence in sequences]
    old_rows = []
    if args.output.exists():
        with args.output.open(newline="") as f:
            old_rows = [r for r in csv.DictReader(f) if r.get("sequence") not in sequences]
    fields = list(dict.fromkeys(k for r in [*old_rows, *rows] for k in r))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([*old_rows, *rows])
    for row in rows:
        print(f"{row['sequence']}: {row['n_valid_depth']}/{row['n_keypoints']} ORB diagnostic keypoints have depth; "
              f"GT pairs={row['n_keypoints_with_gt']}; map scale={row['map_scale_status']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
