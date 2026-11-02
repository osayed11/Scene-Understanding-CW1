"""Run the controlled comparison across all three coursework model families.

This orchestrates benchmark_models.py; weights and public source checkouts must
already exist under --weights-root. GPU commands require CUDA-enabled PyTorch.
Models run sequentially, each with isolated outputs. No weights are downloaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("depth_pro", "dav2_small", "dav2_base", "dav2_large", "da3_metric_large")


def cases(weights: Path, selected: list[str], include_fp16: bool):
    result = []
    for model in selected:
        if model == "depth_pro":
            source = weights / "ml_depth_pro"
            result.append((model, "general", "float32", source / "checkpoints/depth_pro.pt", source, ["all"]))
            if include_fp16:
                result.append((model, "general", "float16", source / "checkpoints/depth_pro.pt", source, ["all"]))
        elif model == "da3_metric_large":
            source = weights / "Depth-Anything-3"
            result.append((model, "general", "float32", weights / "DA3METRIC-LARGE", source, ["all"]))
            if include_fp16:
                result.append((model, "general", "float16", weights / "DA3METRIC-LARGE", source, ["all"]))
        else:
            source = weights / "Depth-Anything-V2"
            encoder = {"dav2_small": "vits", "dav2_base": "vitb", "dav2_large": "vitl"}[model]
            for domain, sequences in (("hypersim", ["replica_room0", "tum_fr3"]), ("vkitti", ["kitti06"])):
                checkpoint = source / f"metric_depth/checkpoints/depth_anything_v2_metric_{domain}_{encoder}.pth"
                result.append((model, domain, "float32", checkpoint, source, sequences))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--weights-root", type=Path, default=ROOT / "weights")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--samples-per-sequence", type=int, default=3)
    parser.add_argument("--include-fp16", action="store_true", help="Also benchmark Depth Pro and DA3 FP16 on CUDA")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/tables/model_benchmarks")
    parser.add_argument("--pred-root", type=Path, default=ROOT / "depth_maps/model_benchmarks")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.samples_per_sequence < 1:
        parser.error("--samples-per-sequence must be positive")
    if args.include_fp16 and not args.device.startswith("cuda"):
        parser.error("--include-fp16 requires a CUDA device")
    log_root = ROOT / "results/logs/model_comparison"
    outcomes = []
    for model, domain, precision, checkpoint, source, sequences in cases(args.weights_root, args.models, args.include_fp16):
        tag = f"spread_{model}_{domain}_{args.device.replace(':', '_')}_{precision}"
        output = args.output_root / f"{tag}.json"
        command = [sys.executable, str(ROOT / "scripts/part2/benchmark_models.py"),
                   "--model", model, "--checkpoint", str(checkpoint), "--source-root", str(source),
                   "--device", args.device, "--precision", precision,
                   "--samples-per-sequence", str(args.samples_per_sequence), "--output", str(output),
                   "--pred-root", str(args.pred_root / tag)]
        if domain != "general":
            command += ["--domain", domain]
        for sequence in sequences:
            command += ["--sequence", sequence]
        for name in ("data_root", "sample_manifest"):
            if getattr(args, name) is not None:
                command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        if args.overwrite:
            command += ["--overwrite"]
        print(json.dumps({"case": tag, "command": command}), flush=True)
        if args.dry_run:
            continue
        log_root.mkdir(parents=True, exist_ok=True)
        log_path = log_root / f"{tag}.log"
        with log_path.open("w") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        row = {"case": tag, "exit_code": completed.returncode, "output": str(output), "log": str(log_path)}
        outcomes.append(row)
        print(json.dumps(row), flush=True)
    if args.dry_run:
        return 0
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "run_status.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    return int(any(row["exit_code"] for row in outcomes))


if __name__ == "__main__":
    raise SystemExit(main())
