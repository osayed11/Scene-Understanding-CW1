from copy import deepcopy
import csv
import json

from part2 import summarize_model_benchmarks as summary


def case(model="dav2_small", sequences=("replica_room0", "tum_fr3", "kitti06"), *, seconds=2., error=.1,
         device="cpu", status="complete", domain="hypersim", coverage=1.):
    frames = []
    for seq in sequences:
        has_gt = seq != "kitti06"
        frames.append({"sequence": seq, "index": 10, "timestamp": 1., "status": "ok",
            "inference_seconds": seconds, "rgb_sha256": "rgb-" + seq, "camera": {"fx": 600},
            "depth_gt_sha256": "gt-" + seq if has_gt else None, "depth_timestamp": 1. if has_gt else None,
            "gt_depth_factor": 5000, "gt_status": "ok" if has_gt else "no_ground_truth",
            "raw": {"abs_rel": error, "rmse_m": error * 2, "delta1": .9, "prediction_coverage": coverage,
                    "valid_gt_pixels": 100} if has_gt else None,
            "median_scaled": {"abs_rel": error / 2, "rmse_m": error, "prediction_coverage": coverage} if has_gt else None})
    return {"status": status, "benchmark_kind": "spread_frame_model_comparison", "signature": {
        "model": model, "device": device, "precision": "float32", "domain": domain,
        "checkpoint_sha256": domain, "source_git_commit": "source", "adapter": {"process_res": 518},
        "runtime": {"device": device, "cpu": "test CPU", "gpu": "GPU" if device == "cuda" else None,
                    "torch_num_threads": 6}, "selection": [{"sequence": s, "index": 10} for s in sequences]},
        "frames": frames}


def test_combines_v2_domains_without_double_weighting_and_separates_gpu():
    indoor = case(sequences=("replica_room0", "tum_fr3"))
    outdoor = case(sequences=("kitti06",), domain="vkitti", seconds=5.)
    gpu = case(device="cuda", seconds=.1)
    rows, groups = summary.summarize_cases([indoor, outdoor, gpu])
    assert len(groups) == 2
    overall = next(r for r in rows if r["sequence"] == "overall" and r["device"] == "cpu")
    assert overall["frames_ok"] == 3 and overall["gt_frames"] == 2
    assert overall["mean_seconds"] == 3  # all three frames, including KITTI
    assert overall["raw_abs_rel"] == .1 and overall["eligible"]
    assert overall["domains"] == "hypersim,vkitti"
    assert overall["pareto"] is None and overall["comparison_count"] == 1  # GPU is a separate comparison


def test_pareto_requires_matching_inputs_and_full_coverage():
    fast = case(model="fast", seconds=1, error=.1)
    slow = case(model="slow", seconds=2, error=.2)
    other_inputs = case(model="different", seconds=.1, error=.01)
    other_inputs["frames"][0]["rgb_sha256"] = "different-input"
    incomplete = case(model="incomplete", seconds=.01, error=.001, coverage=.99)
    rows, _ = summary.summarize_cases([fast, slow, other_inputs, incomplete])
    totals = {r["model"]: r for r in rows if r["sequence"] == "overall"}
    assert totals["fast"]["pareto"] and totals["slow"]["pareto"] is False
    assert totals["different"]["pareto"] is None  # different input has no comparable peer
    assert not totals["incomplete"]["eligible"] and totals["incomplete"]["pareto"] is None


def test_partial_and_failed_cases_are_visible_but_ineligible(tmp_path):
    partial = case(status="running")
    partial["frames"][1]["status"] = "failed"
    partial["frames"][1]["inference_seconds"] = None
    rows, details = summary.summarize_cases([partial])
    overall = next(r for r in rows if r["sequence"] == "overall")
    assert overall["frames_failed"] == 1 and overall["frames_ok"] == 2
    assert not overall["eligible"] and overall["status"] == "partial_or_invalid"
    summary.write_reports(rows, details, [], tmp_path, plot=False)
    assert (tmp_path / "comparison.csv").is_file() and (tmp_path / "comparison.md").is_file()
    assert not (tmp_path / "comparison.png").exists()
    assert "older three-adjacent-frame" in (tmp_path / "comparison.md").read_text()


def test_reader_ignores_original_adjacent_benchmarks(tmp_path):
    (tmp_path / "part2_v2_small_benchmark.json").write_text(json.dumps(case()))
    (tmp_path / "spread_valid.json").write_text(json.dumps(case()))
    cases, errors = summary.load_cases(tmp_path)
    assert len(cases) == 1 and not errors


def test_calibration_or_metric_range_changes_prevent_comparison():
    baseline = case(model="baseline", seconds=2., error=.2)
    changed = case(model="changed", seconds=1., error=.1)
    changed["frames"][0]["camera"]["fx"] = 700
    changed["signature"]["metric_range"] = {"min_depth_m": .1, "max_depth_m": 10}
    rows, _ = summary.summarize_cases([baseline, changed])
    totals = [r for r in rows if r["sequence"] == "overall"]
    assert all(r["pareto"] is None for r in totals)
    assert len({r["sample_id"] for r in totals}) == 2


def test_overall_pareto_weights_full_sequence_counts_instead_of_equal_samples():
    small = case(model="small", error=.2)
    da3 = case(model="da3", error=.1)
    timings = {
        "small": {"replica_room0": 1.573, "tum_fr3": 1.036, "kitti06": 4.363},
        "da3": {"replica_room0": 2.255, "tum_fr3": 3.060, "kitti06": 1.361},
    }
    for model_case in (small, da3):
        for frame in model_case["frames"]:
            frame["inference_seconds"] = timings[model_case["signature"]["model"]][frame["sequence"]]
    rows, _ = summary.summarize_cases([small, da3])
    totals = {r["model"]: r for r in rows if r["sequence"] == "overall"}
    assert totals["small"]["mean_seconds"] > totals["da3"]["mean_seconds"]
    assert totals["small"]["estimated_full_hours"] < totals["da3"]["estimated_full_hours"]
    assert totals["small"]["pareto"] and totals["da3"]["pareto"]
    assert all(r["pareto_runtime_metric"] == "estimated_full_hours" for r in totals.values())


def test_general_models_ignore_old_hypersim_default_without_changing_signatures():
    cases = [case(model="depth_pro"), case(model="da3_metric_large")]
    originals = deepcopy(cases)
    rows, _ = summary.summarize_cases(cases)
    assert all(row["domains"] == "general" for row in rows)
    assert cases == originals
