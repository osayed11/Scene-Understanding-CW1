"""Summarize controlled spread-frame benchmarks without mixing datasets or hardware.

Only spread_*.json harness outputs are read. Partial cases remain visible but are
excluded from Pareto comparisons. CSV/Markdown retain device and precision labels;
the optional figure plots CPU results only, faceted by sequence and hardware.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRAME_COUNTS = {"replica_room0": 2000, "tum_fr3": 2585, "kitti06": 1101}
FIELDS = ("group_id", "model", "device", "precision", "hardware_id", "process_res", "domains",
          "sequence", "status", "frames_expected", "frames_ok", "frames_failed", "frames_pending",
          "gt_frames", "mean_seconds", "raw_abs_rel", "raw_rmse_m", "raw_delta1",
          "scaled_abs_rel", "scaled_rmse_m", "prediction_coverage", "minimum_coverage",
          "estimated_full_hours", "eligible", "comparison_count", "pareto_runtime_metric", "pareto", "sample_id", "flags")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()[:16]


def _mean(values):
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(finite) / len(finite) if finite else None


def _minimum(values):
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return min(finite) if finite else None


def hardware_identity(runtime):
    # Precision deliberately stays outside the hardware key, so labelled
    # precision variants may be compared on the same physical/runtime setup.
    return {key: runtime.get(key) for key in ("device", "cpu", "gpu", "platform", "torch",
                                              "torch_num_threads", "torch_num_interop_threads")}


def sample_identity(frame, signature):
    required = ("rgb_sha256", "camera", "timestamp")
    if any(frame.get(key) is None for key in required):
        return None
    if frame.get("gt_status") == "ok" and not frame.get("depth_gt_sha256"):
        return None
    return {key: frame.get(key) for key in ("sequence", "index", "timestamp", "rgb_sha256", "camera",
              "depth_gt_sha256", "depth_timestamp", "gt_depth_factor")} | {
        "metric_range": frame.get("metric_range", signature.get("metric_range",
                        {"min_depth_m": .001, "max_depth_m": None})),
        "valid_gt_pixels": (frame.get("raw") or {}).get("valid_gt_pixels")}


def load_cases(input_root):
    cases, errors = [], []
    for path in sorted(Path(input_root).glob("spread_*.json")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            if case.get("benchmark_kind") != "spread_frame_model_comparison":
                errors.append(f"{path.name}: not a spread-frame comparison output")
                continue
            case["_filename"] = path.name
            cases.append(case)
        except (OSError, ValueError) as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
    return cases, errors


def summarize_cases(cases):
    groups = defaultdict(list)
    for case in cases:
        signature = case["signature"]
        runtime = signature.get("runtime", {})
        hardware = hardware_identity(runtime)
        adapter = signature.get("adapter", {})
        resolution = adapter.get("process_res", adapter.get("internal_size", signature.get("process_res")))
        key = (signature["model"], signature["device"], signature["precision"],
               _digest(hardware), _canonical(resolution))
        groups[key].append(case)
    rows, details = [], []
    for (model, device, precision, hardware_id, resolution), members in sorted(groups.items()):
        flags, frames, expected, domain_identity = set(), {}, set(), {}
        for case in members:
            signature = case["signature"]
            # Older harness signatures keep the CLI's default 'hypersim' even
            # for general models. Do not mislabel Depth Pro/DA3 as indoor models
            # or alter the saved signature/cache identity to repair the display.
            domain = (signature.get("adapter", {}).get("domain", signature.get("domain", "unspecified"))
                      if model.startswith("dav2_") else "general")
            identity = (signature.get("checkpoint_sha256"), signature.get("source_git_commit"))
            if domain in domain_identity and domain_identity[domain] != identity:
                flags.add("conflicting_checkpoint_or_source_for_domain")
            domain_identity[domain] = identity
            expected.update((s["sequence"], s["index"]) for s in signature.get("selection", []))
            if case.get("status") != "complete":
                flags.add("partial_or_failed_case")
            for frame in case.get("frames", []):
                frame_key = (frame["sequence"], frame["index"])
                value = {**frame, "_identity": sample_identity(frame, signature)}
                if frame_key in frames:
                    # Multiple files must never silently double-weight a sample.
                    if _canonical(frames[frame_key]) != _canonical(value):
                        flags.add("conflicting_duplicate_frame")
                else:
                    frames[frame_key] = value
        if not expected:
            expected = set(frames)
            flags.add("missing_selection_metadata")
        if set(sequence for sequence, _ in expected) != set(FRAME_COUNTS):
            flags.add("missing_sequences")
        if set(frames) != expected:
            flags.add("missing_or_extra_frame_rows")
        if any(f.get("status") != "ok" for f in frames.values()):
            flags.add("incomplete_frames")
        if any(f.get("_identity") is None for f in frames.values()):
            flags.add("missing_input_identity")
        runtime = members[0]["signature"].get("runtime", {})
        if not runtime.get("cpu") and not runtime.get("gpu"):
            flags.add("missing_hardware_identity")
        gt_frames = [f for f in frames.values() if f.get("gt_status") == "ok" and f.get("raw")]
        coverage = [f["raw"].get("prediction_coverage") for f in gt_frames]
        if any(c is None or c < 1 - 1e-12 for c in coverage):
            flags.add("incomplete_prediction_coverage")
        if any(any(f["raw"].get(k) is None or not math.isfinite(f["raw"][k])
                   for k in ("abs_rel", "rmse_m")) for f in gt_frames):
            flags.add("invalid_depth_metrics")
        if any(f.get("sequence") != "kitti06" and (f.get("gt_status") != "ok" or not f.get("raw"))
               for f in frames.values() if f.get("status") == "ok"):
            flags.add("missing_gt_evaluation")
        complete = not flags
        group_id = _digest([model, device, precision, hardware_id, resolution])
        group_rows = []
        for sequence in FRAME_COUNTS:
            subset = [f for (s, _), f in sorted(frames.items()) if s == sequence]
            expected_count = sum(s == sequence for s, _ in expected)
            successful = [f for f in subset if f.get("status") == "ok"]
            scored = [f for f in successful if f.get("gt_status") == "ok" and f.get("raw")]
            identities = [f["_identity"] for f in successful]
            sample_id = _digest(identities) if identities and all(i is not None for i in identities) else ""
            mean_seconds = _mean(f.get("inference_seconds") for f in successful)
            row = {"group_id": group_id, "model": model, "device": device, "precision": precision,
                   "hardware_id": hardware_id, "process_res": resolution,
                   "domains": ",".join(sorted(domain_identity)), "sequence": sequence,
                   "status": "complete" if complete else "partial_or_invalid",
                   "frames_expected": expected_count, "frames_ok": len(successful),
                   "frames_failed": sum(f.get("status") == "failed" for f in subset),
                   "frames_pending": max(0, expected_count - len(successful) - sum(f.get("status") == "failed" for f in subset)),
                   "gt_frames": len(scored), "mean_seconds": mean_seconds,
                   "raw_abs_rel": _mean(f["raw"].get("abs_rel") for f in scored),
                   "raw_rmse_m": _mean(f["raw"].get("rmse_m") for f in scored),
                   "raw_delta1": _mean(f["raw"].get("delta1") for f in scored),
                   "scaled_abs_rel": _mean((f.get("median_scaled") or {}).get("abs_rel") for f in scored),
                   "scaled_rmse_m": _mean((f.get("median_scaled") or {}).get("rmse_m") for f in scored),
                   "prediction_coverage": _mean(f["raw"].get("prediction_coverage") for f in scored),
                   "minimum_coverage": _minimum(f["raw"].get("prediction_coverage") for f in scored),
                   "estimated_full_hours": (mean_seconds * FRAME_COUNTS[sequence] / 3600 if mean_seconds is not None else None),
                   "eligible": complete and bool(scored), "comparison_count": 0, "pareto": None, "sample_id": sample_id,
                   "flags": ";".join(sorted(flags))}
            rows.append(row)
            group_rows.append(row)
        successful = [f for f in frames.values() if f.get("status") == "ok"]
        overall = {**group_rows[0], "sequence": "overall", "frames_expected": len(expected),
                   "frames_ok": len(successful), "frames_failed": sum(r["frames_failed"] for r in group_rows),
                   "frames_pending": sum(r["frames_pending"] for r in group_rows), "gt_frames": len(gt_frames),
                   "mean_seconds": _mean(f.get("inference_seconds") for f in successful),
                   "sample_id": _digest([f.get("_identity") for _, f in sorted(frames.items())]) if complete else "",
                   "estimated_full_hours": sum(r["estimated_full_hours"] for r in group_rows)
                       if complete and all(r["estimated_full_hours"] is not None for r in group_rows) else None,
                   "eligible": complete and bool(gt_frames), "pareto": None}
        # Quality is a macro mean over GT sequences; runtime weights every timed
        # frame equally, including KITTI. Both relevant sample counts are shown.
        for field in ("raw_abs_rel", "raw_rmse_m", "raw_delta1", "scaled_abs_rel", "scaled_rmse_m", "prediction_coverage"):
            overall[field] = _mean(r[field] for r in group_rows if r["gt_frames"])
        overall["minimum_coverage"] = min((r["minimum_coverage"] for r in group_rows if r["minimum_coverage"] is not None), default=None)
        rows.append(overall)
        details.append({"group_id": group_id, "hardware": hardware_identity(members[0]["signature"].get("runtime", {})),
                        "source_files": [m.get("_filename", "synthetic") for m in members], "flags": sorted(flags)})
    mark_pareto(rows)
    return rows, details


def mark_pareto(rows):
    comparisons = defaultdict(list)
    for row in rows:
        runtime_metric = "estimated_full_hours" if row["sequence"] == "overall" else "mean_seconds"
        row["pareto_runtime_metric"] = runtime_metric
        if row["eligible"] and row["sample_id"] and row["raw_abs_rel"] is not None and row[runtime_metric] is not None:
            comparisons[(row["hardware_id"], row["device"], row["sequence"], row["sample_id"])].append(row)
    for candidates in comparisons.values():
        for row in candidates:
            row["comparison_count"] = len(candidates)
        if len(candidates) < 2:
            continue
        for row in candidates:
            runtime_metric = row["pareto_runtime_metric"]
            row["pareto"] = not any(other[runtime_metric] <= row[runtime_metric]
                and other["raw_abs_rel"] <= row["raw_abs_rel"]
                and (other[runtime_metric] < row[runtime_metric] or other["raw_abs_rel"] < row["raw_abs_rel"])
                for other in candidates)


def write_reports(rows, details, errors, output_root, plot=True):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Depth model sample comparison", "",
        "These spread samples are diagnostic evidence, not a conclusive full-dataset model ranking. "
        "Only complete cases with full prediction coverage qualify for Pareto comparisons; samples, "
        "calibration, GT hashes, metric range and hardware must match. No weighted score is used.", "",
        "Inference time excludes setup, warmup, file reads, saving and GT scoring. Per-sequence quality "
        "is a mean over GT frames. Overall quality is a macro mean over GT sequences; overall time "
        "in the sample-mean column averages all successful timed frames, including KITTI. "
        "KITTI supplies no GT depth. Per-sequence Pareto comparisons use mean seconds/frame; "
        "overall Pareto comparisons use estimated full-sequence hours, weighted by actual dataset "
        "frame counts, rather than the equally sampled sequence mix.", "",
        "Full-sequence hours extrapolate the sample's mean time to 2000 Replica, 2585 TUM and 1101 KITTI "
        "frames, excluding overhead. They are estimates, not measured full runs.", "",
        "| Model | Device / precision | Sequence | State | Frames / requested | GT | Sample mean s/frame | Raw AbsRel | Scaled RMSE m | Coverage | Est. hours | Pareto |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    def fmt(value):
        return "—" if value is None else f"{value:.4g}"
    for row in rows:
        lines.append(f"| {row['model']} | {row['device']} / {row['precision']} | {row['sequence']} | {row['status']} | "
            f"{row['frames_ok']} / {row['frames_expected']} | {row['gt_frames']} | {fmt(row['mean_seconds'])} | "
            f"{fmt(row['raw_abs_rel'])} | {fmt(row['scaled_rmse_m'])} | {fmt(row['prediction_coverage'])} | "
            f"{fmt(row['estimated_full_hours'])} | {row['pareto'] if row['pareto'] is not None else 'not compared'} |")
    lines += ["", "GPU comparisons require fresh runs on these exact spread samples. The older three-adjacent-frame "
              "GPU smoke-test bundle is not imported or treated as comparable evidence.", "", "## Case provenance", ""]
    for item in details:
        lines.append(f"- `{item['group_id']}`: hardware `{_canonical(item['hardware'])}`; files "
                     f"{', '.join(item['source_files'])}; flags: {', '.join(item['flags']) or 'none'}.")
    if errors:
        lines += ["", "## Inputs not summarized", ""] + [f"- {error}" for error in errors]
    (output_root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if plot:
        plot_cpu(rows, output_root / "comparison.png")


def plot_cpu(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hardware = sorted({r["hardware_id"] for r in rows if r["device"] == "cpu"}) or ["unmeasured"]
    colors = {model: plt.get_cmap("tab10")(i % 10) for i, model in enumerate(sorted({r["model"] for r in rows}))}
    names = {"depth_pro": "Depth Pro", "da3_metric_large": "DA3 Metric Large",
             "dav2_small": "V2 Small", "dav2_base": "V2 Base", "dav2_large": "V2 Large"}
    fig, axes = plt.subplots(len(hardware), 2, figsize=(12, 4.5 * len(hardware)), squeeze=False, layout="constrained")
    for i, identity in enumerate(hardware):
        for j, sequence in enumerate(("replica_room0", "tum_fr3")):
            ax = axes[i, j]
            candidates = [r for r in rows if r["device"] == "cpu" and r["hardware_id"] == identity
                          and r["sequence"] == sequence and r["mean_seconds"] is not None and r["raw_abs_rel"] is not None]
            for row in candidates:
                ax.scatter(row["mean_seconds"], row["raw_abs_rel"], marker="o" if row["eligible"] else "x",
                           color=colors[row["model"]], s=50)
                rightmost = row["mean_seconds"] == max(r["mean_seconds"] for r in candidates)
                label = names.get(row["model"], row["model"]) + " / " + row["precision"].replace("float", "FP")
                ax.annotate(label + (" (partial)" if not row["eligible"] else ""),
                            (row["mean_seconds"], row["raw_abs_rel"]), xytext=(-5 if rightmost else 5, 5),
                            ha="right" if rightmost else "left", textcoords="offset points", fontsize=8, clip_on=True)
            if not candidates:
                ax.text(.5, .5, "No CPU quality measurements yet", ha="center", transform=ax.transAxes)
            ax.set_title(f"{sequence} · CPU hardware {identity}")
            use_log = candidates and min(r["mean_seconds"] for r in candidates) > 0 and max(
                r["mean_seconds"] for r in candidates) / min(r["mean_seconds"] for r in candidates) > 5
            if use_log:
                ax.set_xscale("log")
            ax.set_xlabel("Mean inference seconds/frame" + (" (log scale)" if use_log else ""))
            ax.set_ylabel("Raw AbsRel (lower is better)")
            ax.grid(alpha=.2)
    fig.suptitle("Spread-frame CPU diagnostics; × = partial/ineligible; three samples are not a final ranking")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "results/tables/model_benchmarks")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/tables/model_comparison")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    cases, errors = load_cases(args.input_root)
    rows, details = summarize_cases(cases)
    write_reports(rows, details, errors, args.output_root, plot=not args.no_plot)
    print(f"Summarized {len(cases)} case files into {len(details)} configurations: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
