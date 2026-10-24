"""Evaluate depth consistency after compensating for camera motion.

Source sensor depth, when available, determines correspondences and visibility
independently of the network. The source prediction is transformed using known
camera-to-world poses and its target-camera z is compared with the prediction
at the corresponding target pixel. Poses and sensor depth are evaluation-only;
they never supply scale to a SLAM method.

Visibility uses a source z-buffer and, when available, target sensor depth.
Without source sensor depth, source predicted geometry determines projection;
without target sensor depth, target predicted depth determines occlusion. Either
fallback makes selection depend on network error and is labelled in the CSV.
Report rejected counts alongside errors. Nearest-pixel sampling avoids mixing
foreground/background depth at edges. Inputs must use undistorted pinhole pixels.

The CSV contains one row per pair. --frame-step k evaluates 0 -> k, k -> 2k,
etc.; it changes both the temporal interval and the sampling of the sequence.
Absolute errors are in metres; relative errors divide by the transformed source
depth. Medians and 90th percentiles describe the retained correspondences, not
all image pixels. Invalid depths, projections and visibility failures are counted.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


SEQUENCES = ("kitti06", "tum_fr3", "replica_room0")
COUNT_FIELDS = (
    "n_sampled", "n_source_valid", "n_source_geometry_valid", "n_source_geometry_invalid",
    "n_projected", "n_zbuffer_occluded", "n_visibility_invalid", "n_visibility_occluded",
    "n_source_prediction_invalid", "n_target_invalid", "n_compared",
)
ERROR_FIELDS = (
    "median_abs_m", "p90_abs_m", "rmse_m", "median_rel", "p90_rel", "mean_signed_m",
)


def _empty_metrics(visibility_mode: str, source_geometry_mode: str = "unavailable") -> dict:
    return {
        "visibility_mode": visibility_mode,
        "source_geometry_mode": source_geometry_mode,
        **{key: 0 for key in COUNT_FIELDS},
        **{key: float("nan") for key in ERROR_FIELDS},
    }


def compare_depth_pair(
    source_depth: np.ndarray,
    target_depth: np.ndarray,
    source_c2w: np.ndarray,
    target_c2w: np.ndarray,
    K: np.ndarray,
    *,
    source_gt_depth: np.ndarray | None = None,
    target_gt_depth: np.ndarray | None = None,
    pixel_stride: int = 8,
    occlusion_abs_tolerance: float = 0.05,
    occlusion_rel_tolerance: float = 0.05,
) -> dict:
    """Return motion-compensated errors and visibility counts for one direction.

    Depths are camera-axis z in metres. Both poses map camera to world. A point
    is considered occluded when its geometry is farther than the visibility depth plus
    ``abs_tolerance + rel_tolerance * visibility_depth``. Invalid sensor depth
    is excluded rather than replaced with a prediction. This permits auditing
    how much of the image the reported conditional error actually represents.
    """
    src = np.asarray(source_depth, dtype=np.float64)
    dst = np.asarray(target_depth, dtype=np.float64)
    if src.ndim != 2 or dst.shape != src.shape:
        raise ValueError("source and target depth must be equally sized 2D arrays")
    if source_gt_depth is not None:
        geometry = np.asarray(source_gt_depth, dtype=np.float64)
        if geometry.shape != src.shape:
            raise ValueError("source ground-truth depth must match source prediction shape")
        geometry_mode = "source_gt"
    else:
        geometry = src
        geometry_mode = "source_prediction"
    if not isinstance(pixel_stride, (int, np.integer)) or pixel_stride < 1:
        raise ValueError("pixel_stride must be a positive integer")
    if (not np.isfinite(occlusion_abs_tolerance) or occlusion_abs_tolerance < 0
            or not np.isfinite(occlusion_rel_tolerance) or occlusion_rel_tolerance < 0):
        raise ValueError("occlusion tolerances must be finite and nonnegative")
    camera = np.asarray(K, dtype=np.float64)
    if (camera.shape != (3, 3) or not np.isfinite(camera).all()
            or camera[0, 0] <= 0 or camera[1, 1] <= 0
            or not np.allclose(camera[2], [0, 0, 1])):
        raise ValueError("K must be a finite pinhole camera matrix")
    transforms = [np.asarray(p, dtype=np.float64) for p in (source_c2w, target_c2w)]
    if any(p.shape != (4, 4) or not np.isfinite(p).all()
           or not np.allclose(p[3], [0, 0, 0, 1]) for p in transforms):
        raise ValueError("poses must be finite 4x4 camera-to-world matrices")
    if target_gt_depth is not None:
        visibility = np.asarray(target_gt_depth, dtype=np.float64)
        if visibility.shape != dst.shape:
            raise ValueError("target ground-truth depth must match target prediction shape")
        mode = "target_gt_and_source_zbuffer"
    else:
        visibility = dst
        mode = "target_prediction_and_source_zbuffer"
    result = _empty_metrics(mode, geometry_mode)

    height, width = src.shape
    v, u = np.mgrid[0:height:pixel_stride, 0:width:pixel_stride]
    u, v = u.ravel(), v.ravel()
    predicted_z = src[v, u]
    geometry_z = geometry[v, u]
    result["n_sampled"] = int(len(geometry_z))
    result["n_source_valid"] = int(np.count_nonzero(np.isfinite(predicted_z) & (predicted_z > 0)))
    valid = np.isfinite(geometry_z) & (geometry_z > 0)
    result["n_source_geometry_invalid"] = int(np.count_nonzero(~valid))
    u, v, geometry_z, predicted_z = u[valid], v[valid], geometry_z[valid], predicted_z[valid]
    result["n_source_geometry_valid"] = int(len(geometry_z))
    if not len(geometry_z):
        return result

    rays = np.linalg.solve(camera, np.vstack((u, v, np.ones(len(geometry_z)))))
    points = rays * geometry_z
    # T_target_source = inverse(T_world_target) @ T_world_source.
    relative = np.linalg.solve(transforms[1], transforms[0])
    points = relative[:3, :3] @ points + relative[:3, 3, None]
    geometry_z = points[2]
    # Invalid network outputs must not remove geometric foreground occluders.
    # Transform predicted z separately; it is used only after visibility selection.
    source_prediction_valid = np.isfinite(predicted_z) & (predicted_z > 0)
    transformed_z = np.full(len(predicted_z), np.nan)
    transformed_z[source_prediction_valid] = (
        (relative[2, :3] @ rays[:, source_prediction_valid]) * predicted_z[source_prediction_valid]
        + relative[2, 3]
    )
    projected = camera @ points
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = projected[:2] / projected[2]
    # Solving K and projecting back may put an exact edge pixel a few ulps
    # outside the image. Keep that pixel without admitting a visible border.
    edge_epsilon = 1e-7
    inside = (np.isfinite(points).all(axis=0) & np.isfinite(uv).all(axis=0)
              & (geometry_z > 0) & (uv[0] >= -edge_epsilon)
              & (uv[0] <= width - 1 + edge_epsilon)
              & (uv[1] >= -edge_epsilon) & (uv[1] <= height - 1 + edge_epsilon))
    geometry_z, transformed_z = geometry_z[inside], transformed_z[inside]
    pixels = np.rint(uv[:, inside]).astype(np.int64)
    result["n_projected"] = int(len(geometry_z))
    if not len(geometry_z):
        return result

    flat = pixels[1] * width + pixels[0]
    zbuffer = np.full(height * width, np.inf)
    np.minimum.at(zbuffer, flat, geometry_z)
    nearest = zbuffer[flat]
    keep = geometry_z <= nearest + occlusion_abs_tolerance + occlusion_rel_tolerance * nearest
    result["n_zbuffer_occluded"] = int(np.count_nonzero(~keep))
    pixels, geometry_z, transformed_z = pixels[:, keep], geometry_z[keep], transformed_z[keep]

    vis_z = visibility[pixels[1], pixels[0]]
    valid_vis = np.isfinite(vis_z) & (vis_z > 0)
    result["n_visibility_invalid"] = int(np.count_nonzero(~valid_vis))
    visible = valid_vis & (
        geometry_z <= vis_z + occlusion_abs_tolerance + occlusion_rel_tolerance * vis_z
    )
    result["n_visibility_occluded"] = int(np.count_nonzero(valid_vis & ~visible))
    pixels, transformed_z = pixels[:, visible], transformed_z[visible]
    valid_prediction = np.isfinite(transformed_z) & (transformed_z > 0)
    result["n_source_prediction_invalid"] = int(np.count_nonzero(~valid_prediction))
    pixels, transformed_z = pixels[:, valid_prediction], transformed_z[valid_prediction]
    target_z = dst[pixels[1], pixels[0]]
    target_valid = np.isfinite(target_z) & (target_z > 0)
    result["n_target_invalid"] = int(np.count_nonzero(~target_valid))
    transformed_z, target_z = transformed_z[target_valid], target_z[target_valid]
    result["n_compared"] = int(len(transformed_z))
    if len(transformed_z):
        difference = target_z - transformed_z
        absolute = np.abs(difference)
        relative_error = absolute / transformed_z
        result.update(
            median_abs_m=float(np.median(absolute)),
            p90_abs_m=float(np.percentile(absolute, 90)),
            rmse_m=float(np.sqrt(np.mean(np.square(difference)))),
            median_rel=float(np.median(relative_error)),
            p90_rel=float(np.percentile(relative_error, 90)),
            mean_signed_m=float(np.mean(difference)),
        )
    return result


def evaluate_sequence(sequence_name, data_root, pred_root, frame_step, pixel_stride,
                      occlusion_abs_tolerance=0.05, occlusion_rel_tolerance=0.05, max_pairs=None):
    """Read cached predictions and return one audit row per sampled frame pair."""
    from part2.common import (
        load_camera, load_depth_gt, load_frame_poses, load_frame_records, load_prediction,
    )

    if not isinstance(frame_step, (int, np.integer)) or frame_step < 1:
        raise ValueError("frame_step must be a positive integer")
    if max_pairs is not None and max_pairs < 1:
        raise ValueError("max_pairs must be positive")
    frames = load_frame_records(sequence_name, data_root=data_root)
    camera = load_camera(sequence_name, data_root=data_root)
    poses = load_frame_poses(sequence_name, frames, data_root=data_root)
    rows = []
    for i in range(0, len(frames) - frame_step, frame_step):
        if max_pairs is not None and len(rows) >= max_pairs:
            break
        j = i + frame_step
        source, target = frames[i], frames[j]
        row = {
            "sequence": sequence_name,
            "source_index": source.index, "target_index": target.index,
            "source_timestamp": source.timestamp, "target_timestamp": target.timestamp,
            "delta_seconds": target.timestamp - source.timestamp,
            "source_depth_timestamp": source.depth_timestamp,
            "source_depth_dt_seconds": (source.depth_timestamp - source.timestamp
                                        if source.depth_timestamp is not None else float("nan")),
            "target_depth_timestamp": target.depth_timestamp,
            "target_depth_dt_seconds": (target.depth_timestamp - target.timestamp
                                        if target.depth_timestamp is not None else float("nan")),
            "frame_step": frame_step, "pixel_stride": pixel_stride,
            "occlusion_abs_tolerance": occlusion_abs_tolerance,
            "occlusion_rel_tolerance": occlusion_rel_tolerance,
            "status": "ok",
        }
        if not np.isfinite(poses[[i, j]]).all():
            row.update(_empty_metrics("unavailable"), status="missing_pose")
        else:
            try:
                source_depth = load_prediction(pred_root, sequence_name, source)
                target_depth = load_prediction(pred_root, sequence_name, target)
            except FileNotFoundError:
                row.update(_empty_metrics("unavailable"), status="missing_prediction")
            else:
                expected_shape = (camera.height, camera.width)
                if source_depth.shape != expected_shape or target_depth.shape != expected_shape:
                    raise ValueError(
                        f"{sequence_name}: predictions must match calibrated image size {expected_shape}"
                    )
                source_gt = load_depth_gt(sequence_name, source, data_root=data_root)
                target_gt = load_depth_gt(sequence_name, target, data_root=data_root)
                row.update(compare_depth_pair(
                    source_depth, target_depth, poses[i], poses[j], camera.K,
                    source_gt_depth=source_gt, target_gt_depth=target_gt, pixel_stride=pixel_stride,
                    occlusion_abs_tolerance=occlusion_abs_tolerance,
                    occlusion_rel_tolerance=occlusion_rel_tolerance,
                ))
                if not row["n_compared"]:
                    row["status"] = "no_visible_correspondences"
        rows.append(row)
    return rows


def main():
    # Support both `python scripts/part2/...py` and module imports from tests.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import RESULTS, ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--sequence", choices=SEQUENCES + ("own",))
    selection.add_argument("--all", action="store_true", help="evaluate the three coursework sequences")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--pred-root", type=Path, default=ROOT / "depth_maps")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--pixel-stride", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, help="Limit pairs for an explicitly partial smoke test")
    parser.add_argument("--occlusion-abs-tolerance", type=float, default=0.05)
    parser.add_argument("--occlusion-rel-tolerance", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=RESULTS / "tables" / "table2_temporal.csv")
    args = parser.parse_args()
    if args.frame_step < 1 or args.pixel_stride < 1:
        parser.error("--frame-step and --pixel-stride must be positive")
    if args.max_pairs is not None and args.max_pairs < 1:
        parser.error("--max-pairs must be positive")
    if (not np.isfinite(args.occlusion_abs_tolerance) or args.occlusion_abs_tolerance < 0
            or not np.isfinite(args.occlusion_rel_tolerance) or args.occlusion_rel_tolerance < 0):
        parser.error("occlusion tolerances must be finite and nonnegative")
    selected = SEQUENCES if args.all else (args.sequence,)
    rows = []
    for name in selected:
        sequence_rows = evaluate_sequence(
            name, args.data_root, args.pred_root, args.frame_step, args.pixel_stride,
            args.occlusion_abs_tolerance, args.occlusion_rel_tolerance,
            args.max_pairs,
        )
        rows.extend(sequence_rows)
        count = sum(row["n_compared"] for row in sequence_rows)
        print(f"{name}: {len(sequence_rows)} pairs, {count} visible depth comparisons")
    if not rows:
        parser.error("no frame pairs: reduce --frame-step or check the sequence")
    previous = []
    if args.output.exists():
        with args.output.open(newline="") as handle:
            previous = [row for row in csv.DictReader(handle) if row["sequence"] not in selected]
    fields = list(dict.fromkeys(key for row in rows + previous for key in row))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(previous + rows)
    print(f"Wrote {args.output}")
    if not any(row["n_compared"] for row in rows):
        parser.exit(1, "No visible depth comparisons; inspect the status and count columns in the CSV.\n")


if __name__ == "__main__":
    main()
