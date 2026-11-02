"""Summarize a completed DA3 cache and its offline evaluation tables.

This reads predictions for four explicitly selected qualitative examples only.
All quantitative results come from the existing full-sequence evaluators.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT
from part2.common import load_depth_gt, load_frame_records, load_prediction

NAMES = {"kitti06": "KITTI 06", "tum_fr3": "TUM fr3", "replica_room0": "Replica room0"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def format_number(value, spec):
    return format(float(value), spec) if value is not None and np.isfinite(float(value)) else "unavailable"


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, str) and value.lower() in ("nan", "inf", "-inf", "infinity", "-infinity"):
        return None
    return value


def validate_cohorts(args, summaries, detail, cache):
    """Reject incomplete/stale cohorts before presenting a full-run report."""
    integrity = json.loads((args.table_root / "cache_integrity.json").read_text())
    if not integrity.get("passed"):
        raise ValueError("Cache integrity audit has not passed")
    summary_keys = [(r["sequence"], r["comparison"]) for r in summaries]
    expected_summary = {(seq, comparison) for seq in NAMES for comparison in ("raw", "median_scaled")}
    if len(summary_keys) != 6 or set(summary_keys) != expected_summary:
        raise ValueError("Dense summary does not contain exactly one raw/scaled row per dataset")
    expected_detail = set()
    for seq, values in cache.items():
        records = load_frame_records(seq, args.data_root)
        if not values["full_complete"] or values["status"] != "complete" or len(records) != values["frames"]:
            raise ValueError(f"{seq}: full cache completion/inventory check failed")
        gt = {frame.index for frame in records if frame.depth_gt is not None}
        if seq != "kitti06" and not gt:
            raise ValueError(f"{seq}: no GT cohort is available for full depth analysis")
        expected_detail.update((seq, comparison, index) for comparison in ("raw", "median_scaled") for index in gt)
        for row in (r for r in summaries if r["sequence"] == seq):
            if (int(row["frames_selected"]) != len(records) or int(row["frames_total"]) != len(records)
                    or int(row["frames_scored"]) != len(gt) or int(row["frames_missing_predictions"]) != 0):
                raise ValueError(f"{seq}: incomplete dense evaluation")
    observed = [(r["sequence"], r["comparison"], int(r["frame_index"])) for r in detail]
    if len(observed) != len(expected_detail) or set(observed) != expected_detail or any(r["status"] != "ok" for r in detail):
        raise ValueError("Dense per-frame rows do not cover the complete valid GT cohort")
    temporal = read_csv(args.table_root / "table2_temporal.csv")
    expected_pairs = {(seq, i, i + 1) for seq, values in cache.items() for i in range(values["frames"] - 1)}
    actual_pairs = [(r["sequence"], int(r["source_index"]), int(r["target_index"])) for r in temporal]
    if len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != expected_pairs:
        raise ValueError("Temporal CSV does not cover every adjacent frame pair exactly once")
    if any(int(r["pixel_stride"]) != 8 or int(r["frame_step"]) != 1 for r in temporal):
        raise ValueError("Expected temporal frame_step=1 and pixel_stride=8")
    keypoint_status = json.loads((args.table_root / "keypoints_evaluation_summary.json").read_text())
    if keypoint_status.get("status") != "complete":
        raise ValueError("Full keypoint analysis has not completed")
    keypoints = read_csv(args.table_root / "table2_keypoints.csv")
    if len(keypoints) != 3 or {r["sequence"] for r in keypoints} != set(NAMES):
        raise ValueError("Expected one keypoint summary row per dataset")
    if any(int(r["n_frames"]) != cache[r["sequence"]]["frames"] or int(r["stride"]) != 1 for r in keypoints):
        raise ValueError("Keypoint CSV does not cover the full frame cohort")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-root", type=Path, default=ROOT / "results/tables/da3/full_cpu_float32")
    parser.add_argument("--pred-root", type=Path, default=ROOT / "depth_maps/da3_metric_large_cpu_float32")
    parser.add_argument("--figure-root", type=Path, default=ROOT / "results/figures/part2/da3_full_cpu_float32")
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    args.figure_root.mkdir(parents=True, exist_ok=True)
    summaries = read_csv(args.table_root / "table2_depth.csv")
    detail = read_csv(args.table_root / "table2_depth_frames.csv")
    sequences = ("tum_fr3", "replica_room0")
    selected = {}
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_root": str(args.pred_root),
        "dense_evaluation": summaries,
        "aggregation": "Arithmetic mean of per-frame metrics; valid GT > 0.001 m, no maximum depth or crop.",
        "median_scaling": "A separate GT-fitted scale for every frame; diagnoses depth shape and is not an online SLAM calibration.",
        "cache": {},
        "sources_sha256": {},
    }
    for seq in NAMES:
        manifest = json.loads((args.pred_root / seq / "manifest.json").read_text())
        report["cache"][seq] = {
            "frames": len(manifest["frames"]), "status": manifest["status"],
            "full_complete": manifest["full_complete"], "signature": manifest["signature"],
        }
    validate_cohorts(args, summaries, detail, report["cache"])
    report["input_files_sha256"] = {
        name: hashlib.sha256((args.table_root / name).read_bytes()).hexdigest()
        for name in ("table2_depth.csv", "table2_depth_frames.csv", "table2_temporal.csv", "table2_keypoints.csv",
                     "cache_integrity.json", "cache_diagnostics.json", "depth_eval_summary.json", "temporal_summary.json", "keypoints_evaluation_summary.json")
    }
    for name in ("eval_depth.py", "temporal_consistency.py", "keypoint_depth.py", "common.py", "summarize_da3_run.py"):
        path = Path(__file__).with_name(name)
        report["sources_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["sources_sha256"]["depth.py"] = hashlib.sha256((ROOT / "scripts/depth.py").read_bytes()).hexdigest()
    for name in ("cache_integrity.json", "cache_diagnostics.json", "depth_eval_summary.json", "temporal_summary.json", "keypoints_evaluation_summary.json", "spatial_diagnostics.json"):
        path = args.table_root / name
        if path.exists():
            report[name.removesuffix(".json")] = json.loads(path.read_text())
            report["input_files_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    keypoint_path = args.table_root / "table2_keypoints.csv"
    if keypoint_path.exists():
        report["keypoint_evaluation"] = read_csv(keypoint_path)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.8), layout="constrained", sharex="col")
    for col, seq in enumerate(sequences):
        for comparison, label, color in (("raw", "Raw metric prediction", "#2166ac"),
                                          ("median_scaled", "Per-frame median-scaled", "#c66b16")):
            rows = [r for r in detail if r["sequence"] == seq and r["comparison"] == comparison and r["status"] == "ok"]
            rows.sort(key=lambda r: int(r["frame_index"]))
            indices = [int(r["frame_index"]) for r in rows]
            axes[0, col].plot(indices, [100 * float(r["abs_rel"]) for r in rows], lw=.65, color=color, label=label)
            if comparison == "median_scaled":
                scales = np.array([float(r["scale"]) for r in rows])
                axes[1, col].plot(indices, scales, color="#27836d", lw=.7)
                report.setdefault("scale_diagnostics", {})[seq] = {
                    "p05": float(np.percentile(scales, 5)), "median": float(np.median(scales)),
                    "p95": float(np.percentile(scales, 95)), "std": float(np.std(scales)),
                }
        raw = [r for r in detail if r["sequence"] == seq and r["comparison"] == "raw" and r["status"] == "ok"]
        if not raw:
            raise ValueError(f"{seq}: no valid indoor GT results; cannot plot a full-run analysis")
        ordered = sorted(raw, key=lambda r: float(r["abs_rel"]))
        errors = [float(r["abs_rel"]) for r in ordered]
        report.setdefault("per_frame_error_distribution", {})[seq] = {
            "median_abs_rel": float(np.median(errors)), "p90_abs_rel": float(np.percentile(errors, 90)),
            "max_abs_rel": max(errors), "scored_frames": len(errors),
            "max_depth_timestamp_offset_seconds": max(float(r["depth_dt"]) for r in raw),
            "highest_error_frames": [{"frame_index": int(r["frame_index"]), "abs_rel": float(r["abs_rel"])} for r in ordered[-5:][::-1]],
            "gt_valid_pixel_fraction_in_matched_frames": sum(int(r["valid_gt_pixels"]) for r in raw) / (
                len(raw) * report["cache"][seq]["signature"]["camera"]["width"] * report["cache"][seq]["signature"]["camera"]["height"]),
        }
        chosen = [("median error", ordered[len(ordered) // 2]), ("highest error", ordered[-1])]
        selected[seq] = chosen
        report.setdefault("qualitative_selection", {})[seq] = [
            {"selection": label, "frame_index": int(row["frame_index"]), "abs_rel": float(row["abs_rel"]),
             "rmse_m": float(row["rmse_m"]), "valid_gt_pixels": int(row["valid_gt_pixels"])}
            for label, row in chosen
        ]
        axes[0, col].set_title(NAMES[seq])
        axes[0, col].set_ylabel("AbsRel per frame (%)")
        axes[1, col].axhline(1, color="#555555", ls="--", lw=1, label="No scale correction")
        axes[1, col].set_ylabel("GT-fitted scale multiplier")
        axes[1, col].set_xlabel("Original RGB frame index")
        for ax in axes[:, col]:
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle("DA3 Metric Large: full-sequence depth accuracy and scale")
    fig.supxlabel("Raw outputs are used directly. Median scaling fits ground truth independently for each frame.", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(args.figure_root / f"depth_quality.{ext}", dpi=170)
    plt.close(fig)

    spatial = report.get("spatial_diagnostics")
    if spatial:
        if spatial["status"] != "complete":
            raise ValueError("Sampled spatial analysis is incomplete")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
        x = np.arange(2); width = .35
        for offset, region, label, color in ((-.5, "outside_detected_band", "Outside detected band", "#2166ac"),
                                              (.5, "gt_discontinuity_band", "GT discontinuity band", "#c66b16")):
            values = [100 * spatial["sequences"][seq]["regions"][region]["abs_rel"] for seq in sequences]
            bars = axes[0].bar(x + offset * width, values, width, label=label, color=color)
            axes[0].bar_label(bars, fmt="%.2f%%", fontsize=8)
        axes[0].set_xticks(x, [NAMES[seq] for seq in sequences])
        axes[0].set_title("Error near depth discontinuities")
        bins = spatial["depth_bins"]
        x = np.arange(len(bins))
        for offset, seq, color in ((-.5, "tum_fr3", "#2166ac"), (.5, "replica_room0", "#27836d")):
            values = [spatial["sequences"][seq]["regions"][b["region"]]["abs_rel"] for b in bins]
            axes[1].bar(x + offset * width, [100 * v if v is not None else np.nan for v in values], width, label=NAMES[seq], color=color)
        axes[1].set_xticks(x, ["(0, 1]", "(1, 2]", "(2, 4]", "(4, 8]", ">8"])
        axes[1].set_xlabel("GT depth bin (m); Replica has no sampled pixels >8 m")
        axes[1].set_title("Error by ground-truth depth")
        for ax in axes:
            ax.set_ylabel("Pooled pixel AbsRel (%)")
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle("Sampled spatial diagnostic: 32 evenly spread GT-matched frames per dataset")
        fig.supxlabel("GT neighbor jump >0.1 m +5% of nearer depth; band dilated by 3 pixels. Outside does not imply a flat surface.", fontsize=8)
        for ext in ("png", "pdf"):
            fig.savefig(args.figure_root / f"spatial_errors.{ext}", dpi=170)
        plt.close(fig)

    temporal_path = args.table_root / "table2_temporal.csv"
    if temporal_path.exists():
        temporal = read_csv(temporal_path)
        fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), layout="constrained")
        report["temporal_evaluation"] = {}
        for ax, seq in zip(axes, NAMES):
            rows = [r for r in temporal if r["sequence"] == seq]
            good = [r for r in rows if r["status"] == "ok" and np.isfinite(float(r["median_rel"]))]
            sensor = [r for r in good if r["source_geometry_mode"] == "source_gt" and r["visibility_mode"] == "target_gt_and_source_zbuffer"]
            fallback = [r for r in good if r["source_geometry_mode"] != "source_gt" or r["visibility_mode"] != "target_gt_and_source_zbuffer"]
            primary = sensor or good
            sampled = sum(int(r["n_sampled"]) for r in rows)
            compared = sum(int(r["n_compared"]) for r in rows)
            report["temporal_evaluation"][seq] = {
                "pairs_total": len(rows), "statuses": dict(Counter(r["status"] for r in rows)),
                "sampled_points": sampled, "compared_points": compared,
                "retained_fraction": compared / sampled if sampled else None,
                "primary_mode": "source_gt_and_target_gt_visibility" if sensor else "prediction_dependent",
                "primary_pairs": len(primary),
                "mean_of_pair_median_relative_errors": float(np.mean([float(r["median_rel"]) for r in primary])) if primary else None,
                "mean_of_pair_median_absolute_errors_m": float(np.mean([float(r["median_abs_m"]) for r in primary])) if primary else None,
                "mean_of_pair_p90_relative_errors": float(np.mean([float(r["p90_rel"]) for r in primary])) if primary else None,
            }
            for group, label, color in ((sensor, "GT geometry and visibility", "#2166ac"),
                                         (fallback, "Prediction-dependent geometry or visibility", "#b77832")):
                if group:
                    ax.plot([int(r["source_index"]) for r in group], [100 * float(r["median_rel"]) for r in group],
                            ".", markersize=1.8, color=color, label=label)
            ax.set_title(NAMES[seq], loc="left")
            ax.set_ylabel("Pair median error (%)")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, loc="upper right")
        axes[-1].set_xlabel("Source RGB frame index; every adjacent pair, pixel stride 8")
        fig.suptitle("Motion-compensated depth consistency on retained correspondences")
        fig.supxlabel("KITTI uses approximate reference poses and prediction-dependent visibility. Rejection counts are reported in the tables.", fontsize=8)
        for ext in ("png", "pdf"):
            fig.savefig(args.figure_root / f"temporal_consistency.{ext}", dpi=170)
        plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(14, 10), layout="constrained")
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#c8ccd0")
    error_cmap = plt.get_cmap("inferno").copy(); error_cmap.set_bad("#c8ccd0")
    row_number = 0
    for seq in sequences:
        frames = load_frame_records(seq, args.data_root)
        for selection, row in selected[seq]:
            frame = frames[int(row["frame_index"])]
            bgr = cv2.imread(str(frame.rgb))
            if bgr is None:
                raise FileNotFoundError(frame.rgb)
            gt = load_depth_gt(seq, frame, args.data_root)
            pred = load_prediction(args.pred_root, seq, frame)
            valid = np.isfinite(gt) & (gt > .001) & np.isfinite(pred) & (pred > 0)
            axes[row_number, 0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            axes[row_number, 1].imshow(np.ma.masked_where(~np.isfinite(gt) | (gt <= 0), gt), cmap=cmap, vmin=0, vmax=8)
            depth_image = axes[row_number, 2].imshow(np.ma.masked_invalid(pred), cmap=cmap, vmin=0, vmax=8)
            error_image = axes[row_number, 3].imshow(np.ma.masked_where(~valid, np.abs(pred - gt)), cmap=error_cmap, vmin=0, vmax=1)
            axes[row_number, 0].set_title(f"{NAMES[seq]} frame {frame.index}\n{selection}: AbsRel {100 * float(row['abs_rel']):.2f}%", fontsize=9)
            for col, title in enumerate(("", "Ground truth", "Raw prediction", "Absolute error")):
                if col:
                    axes[row_number, col].set_title(title, fontsize=10)
                axes[row_number, col].axis("off")
            row_number += 1
    fig.colorbar(depth_image, ax=axes[:, 1:3], label="Depth (m); display capped at 8 m", shrink=.65, fraction=.018, extend="max")
    fig.colorbar(error_image, ax=axes[:, 3], label="Absolute error (m); display capped at 1 m", shrink=.65, fraction=.04, extend="max")
    fig.suptitle("DA3: examples selected by median and highest raw per-frame AbsRel")
    fig.supxlabel("Shared depth colour scale; grey = unavailable GT. No scale fit. Stored depths are unchanged.", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(args.figure_root / f"depth_examples.{ext}", dpi=170)
    plt.close(fig)

    report["limitations"] = [
        "KITTI 06 has no supplied GT depth, so no KITTI depth accuracy is claimed.",
        "TUM uses nearest GT depth within 20 ms; unmatched RGB frames are excluded from depth scores and sensor holes are masked.",
        "KITTI temporal geometry/visibility is prediction-dependent and uses the supplied reference-pose stream without an explicit cam0-to-colour-cam2 extrinsic correction.",
        "Temporal consistency does not establish metric accuracy; moving objects and sensor timing can affect it.",
        "ORB keypoints are diagnostic detections, not the saved SLAM observations; patch dispersion is an uncalibrated uncertainty heuristic.",
        "Full inference and these diagnostics do not by themselves finish the actual SLAM-keypoint/map-scale requirements or Part 3 integrations.",
    ]
    path = args.table_root / "analysis.json"
    path.write_text(json.dumps(clean_json(report), indent=2, allow_nan=False) + "\n")
    lines = ["# DA3 full CPU run analysis", "", f"Generated {report['generated_utc']}.", "",
             "All metrics use the saved native float32 metre predictions. No new network inference is performed.", "",
             "## Depth accuracy", "", "| Dataset | Comparison | Scored frames | AbsRel | RMSE (m) | delta < 1.25 |",
             "|---|---|---:|---:|---:|---:|"]
    for row in summaries:
        if row["status"] != "ok":
            continue
        lines.append(f"| {NAMES[row['sequence']]} | {row['comparison']} | {row['frames_scored']} | "
                     f"{float(row['abs_rel']):.2%} | {float(row['rmse_m']):.3f} | {float(row['delta1']):.2%} |")
    lines.extend(["", report["aggregation"], "", report["median_scaling"], "", "## Figures", "",
                  "See `results/figures/part2/da3_full_cpu_float32/depth_quality.png` and `depth_examples.png` (PDF copies alongside).", "",
                  "The examples are selected by raw error over all scored frames: the middle-ranked frame and the highest-error frame in each indoor sequence.", "",
                  "## Scale variation", ""])
    for seq, values in report["scale_diagnostics"].items():
        lines.append(f"- {NAMES[seq]}: per-frame GT-fitted multiplier median {values['median']:.4f}; 5th-95th percentile {values['p05']:.4f}-{values['p95']:.4f}. A multiplier below 1 reduces overestimated depths.")
    lines.append("\nThe median multiplier alone cannot describe frame-to-frame scale variation. Improvements after fitting GT are diagnostic and do not establish that a fixed online correction will transfer to other views.")
    integrity = report.get("cache_integrity")
    if integrity:
        lines.extend(["", "## Integrity", "", f"Audit status: {integrity['status']}; {integrity['error_count']} errors. Verified {integrity['total_committed_frames']} NPY/PNG pairs and their source RGB hashes.",
                      "Associations still use portable data/... tokens; rebase paths before direct pySLAM use."])
    timing = report.get("cache_diagnostics", {}).get("overall")
    if timing:
        lines.extend(["", "## Recorded inference time", "", "| Dataset | Mean seconds/frame | Median | p90 | Inference hours |", "|---|---:|---:|---:|---:|"])
        for seq, values in report["cache_diagnostics"]["sequences"].items():
            t = values["inference_seconds"]
            lines.append(f"| {NAMES[seq]} | {t['mean']:.3f} | {t['median']:.3f} | {t['p90']:.3f} | {t['sum'] / 3600:.3f} |")
        lines.append("\nInference timing excludes cache writes, input I/O, validation and interruption time. Session-level timing limits are in cache_diagnostics.json.")
    if report.get("temporal_evaluation"):
        lines.extend(["", "## Temporal consistency", "", "Every adjacent pair is recorded; images are sampled on an 8-pixel grid. Errors below are equal-pair means of within-pair statistics, conditional on retained correspondences.", "",
                      "| Dataset | Primary pairs | Mean pair-median relative error | Mean pair-median absolute error (m) | All-pair retained/sample |", "|---|---:|---:|---:|---:|"])
        for seq, values in report["temporal_evaluation"].items():
            lines.append(f"| {NAMES[seq]} | {values['primary_pairs']} | {format_number(values['mean_of_pair_median_relative_errors'], '.2%')} | {format_number(values['mean_of_pair_median_absolute_errors_m'], '.4f')} | {format_number(values['retained_fraction'], '.2%')} |")
        lines.append("\nPrimary TUM/Replica errors use source GT geometry and target GT visibility. KITTI uses predictions and approximate reference poses, so its figure has different limitations. TUM fallback modes and missing poses remain explicit in the CSV and temporal_summary.json. Coverage above includes all available modes; missing-pose pairs have no sampled points.")
    if spatial:
        lines.extend(["", "## Sampled spatial patterns", "", spatial["scope"], "",
                      "| Dataset | AbsRel in GT discontinuity band | AbsRel outside detected band | Band fraction of valid pixels |", "|---|---:|---:|---:|"])
        for seq in sequences:
            s = spatial["sequences"][seq]
            edge = s["regions"]["gt_discontinuity_band"]
            outside = s["regions"]["outside_detected_band"]
            lines.append(f"| {NAMES[seq]} | {edge['abs_rel']:.2%} | {outside['abs_rel']:.2%} | {s['gt_discontinuity_band_fraction_of_valid_pixels']:.2%} |")
        lines.append("\nOn these 64 frames, relative error is approximately twice as large in the detected boundary bands. Edges use valid four-neighbor GT jumps >0.1 m +5% of nearer depth, dilated by 3 pixels. These are pooled sampled-pixel metrics, not the full-sequence frame means above; outside the band is not assumed planar. See spatial_diagnostics.json for exact frame indices, bin populations and signed errors, and spatial_errors.png for the plot.")
    if report.get("keypoint_evaluation"):
        lines.extend(["", "## ORB diagnostic keypoints", "", "These are fresh ORB detections, not the saved SLAM observations. Metrics pool all valid GT keypoint pairs and therefore use a different aggregation from dense metrics.", "",
                      "| Dataset | Keypoints | Keypoints with GT | AbsRel | 1x heuristic-sigma coverage | Sigma/absolute-error correlation |", "|---|---:|---:|---:|---:|---:|"])
        for row in report["keypoint_evaluation"]:
            scored = int(row["n_keypoints_with_gt"]) > 0
            accuracy = format_number(row['keypoint_absrel'], '.2%') if scored else "unavailable"
            coverage = format_number(row['heuristic_sigma_coverage_1x'], '.2%') if scored else "unavailable"
            correlation = format_number(row['sigma_abs_error_correlation'], '.3f') if scored else "unavailable"
            lines.append(f"| {NAMES[row['sequence']]} | {row['n_keypoints']} | {row['n_keypoints_with_gt']} | {accuracy} | {coverage} | {correlation} |")
        lines.append("\nPatch dispersion is a heuristic, not a calibrated confidence interval. Map scale is unavailable because no paired triangulated map observations were supplied.")
    lines.extend(["", "## Interpretation limits", ""])
    lines.extend("- " + item for item in report["limitations"])
    (args.table_root / "analysis.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"report": str(path), "figures": str(args.figure_root), "selected": report["qualitative_selection"]}))


if __name__ == "__main__":
    main()
