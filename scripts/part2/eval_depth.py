"""Evaluate metric predictions against GT depth, before and after per-frame median scaling.

Summary errors are arithmetic means of per-frame metrics. Median scaling is fitted
on each frame's valid evaluation pixels: it diagnoses shape, not metric scale.
KITTI 06 has no supplied depth ground truth and is reported explicitly.

    python scripts/part2/eval_depth.py --sequence replica_room0 tum_fr3
    python scripts/part2/eval_depth.py --all --stride 10 --skip-missing
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT, load_sequence
from depth import load_depth_png
from part2.common import load_frame_records, load_prediction


SEQUENCES = ("kitti06", "tum_fr3", "replica_room0")
COMPARISONS = ("raw", "median_scaled")
SUMMARY_FIELDS = (
    "sequence", "comparison", "status", "frames_total", "frames_selected",
    "frames_with_gt", "frames_without_gt", "frames_missing_predictions",
    "frames_scored", "frames_no_valid_pixels", "valid_gt_pixels", "valid_pixels",
    "prediction_coverage", "abs_rel", "rmse_m", "delta1", "scale_mean",
    "scale_median", "scale_std", "min_depth_m", "max_depth_m", "aggregation",
)
FRAME_FIELDS = (
    "sequence", "comparison", "frame_index", "timestamp", "rgb", "depth_gt",
    "depth_timestamp", "depth_dt", "status", "valid_gt_pixels", "valid_pixels",
    "prediction_coverage", "abs_rel", "rmse_m", "delta1", "scale",
)


def depth_metrics(prediction: np.ndarray, ground_truth: np.ndarray, *,
                  median_scale: bool = False, min_depth: float = 0.001,
                  max_depth: float | None = None) -> dict:
    """Score aligned HxW maps in metres and expose invalid-prediction coverage.

    GT must be finite, positive and within the requested GT depth range. Prediction
    must be finite and positive. Invalid predictions are excluded from errors but
    remain visible in the coverage denominator. No resizing, clipping or filling
    is performed. delta1 uses a strict ratio threshold of 1.25.
    """
    if not np.isfinite(min_depth) or min_depth < 0:
        raise ValueError("min_depth must be finite and nonnegative")
    if max_depth is not None and (not np.isfinite(max_depth) or max_depth <= min_depth):
        raise ValueError("max_depth must be finite and greater than min_depth")
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape != gt.shape:
        raise ValueError(f"aligned HxW maps required; prediction {pred.shape}, GT {gt.shape}")
    valid_gt = np.isfinite(gt) & (gt > min_depth)
    if max_depth is not None:
        valid_gt &= gt <= max_depth
    valid = valid_gt & np.isfinite(pred) & (pred > 0)
    n_gt, n_valid = int(valid_gt.sum()), int(valid.sum())
    result = {
        "valid_gt_pixels": n_gt, "valid_pixels": n_valid,
        "prediction_coverage": n_valid / n_gt if n_gt else float("nan"),
        "abs_rel": float("nan"), "rmse_m": float("nan"),
        "delta1": float("nan"), "scale": float("nan"),
    }
    if not n_valid:
        return result
    pred_values, gt_values = pred[valid], gt[valid]
    scale = float(np.median(gt_values) / np.median(pred_values)) if median_scale else 1.0
    pred_values = pred_values * scale
    difference = pred_values - gt_values
    result.update(
        abs_rel=float(np.mean(np.abs(difference) / gt_values)),
        rmse_m=float(np.sqrt(np.mean(difference ** 2))),
        delta1=float(np.mean(np.maximum(pred_values / gt_values, gt_values / pred_values) < 1.25)),
        scale=scale,
    )
    return result


def _mean(rows: list[dict], key: str) -> float:
    values = [row[key] for row in rows if np.isfinite(row[key])]
    return float(np.mean(values)) if values else float("nan")


def evaluate_sequence(sequence: str, *, data_root: Path | None = None,
                      pred_root: Path = ROOT / "depth_maps", stride: int = 1,
                      max_frames: int | None = None, min_depth: float = 0.001,
                      max_depth: float | None = None,
                      skip_missing: bool = False) -> tuple[list[dict], list[dict]]:
    """Return summary rows and auditable per-frame rows for a configured sequence."""
    if stride < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError("stride and max_frames must be positive")
    cfg = load_sequence(sequence)
    frames = load_frame_records(sequence, data_root=data_root)
    selected = frames[::stride]
    if max_frames is not None:
        selected = selected[:max_frames]
    has_gt = cfg.get("depth_gt") is not None
    factor = cfg.get("gt_depth_factor", cfg.get("depth_factor"))
    if has_gt and (factor is None or not np.isfinite(factor) or factor <= 0):
        raise ValueError(f"{sequence}: a positive GT depth_factor is required")
    counts = {"frames_with_gt": 0, "frames_without_gt": 0, "frames_missing_predictions": 0}
    per_frame: list[dict] = []
    if has_gt:
        for frame in selected:
            if frame.depth_gt is None:
                counts["frames_without_gt"] += 1
                continue
            counts["frames_with_gt"] += 1
            gt = load_depth_png(frame.depth_gt, factor)
            try:
                pred = load_prediction(pred_root, sequence, frame)
            except FileNotFoundError:
                if not skip_missing:
                    raise
                counts["frames_missing_predictions"] += 1
                pred = np.full_like(gt, np.nan)
                prediction_missing = True
            else:
                prediction_missing = False
            for comparison in COMPARISONS:
                try:
                    metrics = depth_metrics(pred, gt, median_scale=comparison == "median_scaled",
                                            min_depth=min_depth, max_depth=max_depth)
                except ValueError as exc:
                    raise ValueError(f"{sequence} frame {frame.stem}: {exc}") from exc
                per_frame.append({
                    "sequence": sequence, "comparison": comparison,
                    "frame_index": frame.index, "timestamp": frame.timestamp,
                    "rgb": str(frame.rgb), "depth_gt": str(frame.depth_gt),
                    "depth_timestamp": frame.depth_timestamp,
                    "depth_dt": (abs(frame.depth_timestamp - frame.timestamp)
                                 if frame.depth_timestamp is not None else float("nan")),
                    "status": ("missing_prediction" if prediction_missing else
                               "ok" if metrics["valid_pixels"] else "no_valid_pixels"), **metrics,
                })
    else:
        counts["frames_without_gt"] = len(selected)

    summaries = []
    for comparison in COMPARISONS:
        rows = [row for row in per_frame if row["comparison"] == comparison]
        scored = [row for row in rows if row["valid_pixels"]]
        if not has_gt:
            status = "no_ground_truth"
        elif scored:
            status = "ok"
        elif counts["frames_missing_predictions"]:
            status = "no_predictions"
        elif rows:
            status = "no_valid_pixels"
        else:
            status = "no_gt_matches"
        scales = np.asarray([row["scale"] for row in scored])
        n_gt = sum(row["valid_gt_pixels"] for row in rows)
        n_valid = sum(row["valid_pixels"] for row in rows)
        summaries.append({
            "sequence": sequence, "comparison": comparison, "status": status,
            "frames_total": len(frames), "frames_selected": len(selected), **counts,
            "frames_scored": len(scored),
            "frames_no_valid_pixels": sum(row["status"] == "no_valid_pixels" for row in rows),
            "valid_gt_pixels": n_gt, "valid_pixels": n_valid,
            "prediction_coverage": n_valid / n_gt if n_gt else float("nan"),
            "abs_rel": _mean(scored, "abs_rel"), "rmse_m": _mean(scored, "rmse_m"),
            "delta1": _mean(scored, "delta1"), "scale_mean": _mean(scored, "scale"),
            "scale_median": float(np.median(scales)) if scales.size else float("nan"),
            "scale_std": float(np.std(scales)) if scales.size else float("nan"),
            "min_depth_m": min_depth, "max_depth_m": max_depth, "aggregation": "mean_per_frame",
        })
    return summaries, per_frame


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sequence", nargs="+", choices=(*SEQUENCES, "own"))
    selection.add_argument("--all", action="store_true", help="evaluate all three common sequences (default)")
    parser.add_argument("--data-root", type=Path, help="dataset root passed to the shared frame loader")
    parser.add_argument("--pred-root", type=Path, default=ROOT / "depth_maps")
    parser.add_argument("--output", type=Path, default=ROOT / "results/tables/table2_depth.csv")
    parser.add_argument("--per-frame-output", type=Path, help="default: <output stem>_frames.csv")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--min-depth", type=float, default=0.001, help="minimum valid GT depth, metres (exclusive)")
    parser.add_argument("--max-depth", type=float, help="maximum valid GT depth, metres (inclusive); default unlimited")
    parser.add_argument("--skip-missing", action="store_true", help="record missing predictions rather than fail")
    args = parser.parse_args(argv)
    if args.stride < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("--stride and --max-frames must be positive")
    if not np.isfinite(args.min_depth) or args.min_depth < 0:
        parser.error("--min-depth must be finite and nonnegative")
    if args.max_depth is not None and (not np.isfinite(args.max_depth) or args.max_depth <= args.min_depth):
        parser.error("--max-depth must be finite and greater than --min-depth")
    summaries, per_frame = [], []
    for sequence in args.sequence or SEQUENCES:
        seq_summaries, seq_frames = evaluate_sequence(
            sequence, data_root=args.data_root, pred_root=args.pred_root,
            stride=args.stride, max_frames=args.max_frames, min_depth=args.min_depth,
            max_depth=args.max_depth, skip_missing=args.skip_missing,
        )
        summaries.extend(seq_summaries)
        per_frame.extend(seq_frames)
        for row in seq_summaries:
            print(f"{sequence} {row['comparison']}: {row['status']}; "
                  f"frames={row['frames_scored']}/{row['frames_selected']} "
                  f"AbsRel={row['abs_rel']:.5g} RMSE(m)={row['rmse_m']:.5g} delta1={row['delta1']:.5g}")
    frames_output = args.per_frame_output or args.output.with_name(args.output.stem + "_frames.csv")
    _write_csv(args.output, summaries, SUMMARY_FIELDS)
    _write_csv(frames_output, per_frame, FRAME_FIELDS)
    print(f"wrote {args.output} and {frames_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
