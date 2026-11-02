"""Plot RGB, raw predicted metric depth, GT depth and absolute error for one frame."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from config import ROOT
from part2.common import SEQUENCES, load_depth_gt, load_frame_records, load_prediction
from part2.eval_depth import depth_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", choices=SEQUENCES, default="replica_room0")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--pred-root", type=Path, default=ROOT / "depth_maps")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--depth-max", type=float, help="Shared depth colour-scale maximum in metres")
    parser.add_argument("--error-max", type=float, default=.5, help="Absolute-error colour-scale maximum in metres")
    args = parser.parse_args()
    if args.frame_index < 0:
        parser.error("--frame-index must be nonnegative")
    if (args.depth_max is not None and (not np.isfinite(args.depth_max) or args.depth_max <= 0)
            or not np.isfinite(args.error_max) or args.error_max <= 0):
        parser.error("colour-scale maxima must be positive and finite")
    frames = load_frame_records(args.sequence, args.data_root)
    if args.frame_index >= len(frames):
        parser.error(f"frame index must be below {len(frames)}")
    frame = frames[args.frame_index]
    bgr = cv2.imread(str(frame.rgb), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(frame.rgb)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    prediction = load_prediction(args.pred_root, args.sequence, frame)
    truth = load_depth_gt(args.sequence, frame, args.data_root)
    if prediction.shape != rgb.shape[:2] or (truth is not None and truth.shape != prediction.shape):
        raise ValueError("RGB, predicted depth and GT depth must have matching image dimensions")
    values = [prediction[np.isfinite(prediction) & (prediction > 0)]]
    if truth is not None:
        values.append(truth[np.isfinite(truth) & (truth > 0)])
    values = np.concatenate(values)
    if not values.size:
        raise ValueError("No valid depth pixels to visualize")
    depth_max = args.depth_max or max(.5, np.ceil(float(values.max()) * 2) / 2)
    depth_norm = Normalize(0, depth_max)
    depth_cmap = plt.get_cmap("viridis").copy()
    error_cmap = plt.get_cmap("inferno").copy()
    depth_cmap.set_bad("#c8ccd0")
    error_cmap.set_bad("#c8ccd0")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig = plt.figure(figsize=(15.5, 9.8), layout="constrained")
    grid = fig.add_gridspec(2, 4, width_ratios=[1, .035, 1, .035])
    axes = np.array([[fig.add_subplot(grid[row, col]) for col in (0, 2)] for row in (0, 1)])
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB image", loc="left", fontweight="bold")
    masked_prediction = np.ma.masked_where(~np.isfinite(prediction) | (prediction <= 0), prediction)
    depth_artist = axes[0, 1].imshow(masked_prediction, cmap=depth_cmap, norm=depth_norm)
    axes[0, 1].set_title("Predicted depth — raw metres", loc="left", fontweight="bold")
    footnote = "Prediction and ground truth use the same colour scale. Grey = unavailable depth. No fitted scale applied."
    if truth is not None:
        masked_truth = np.ma.masked_where(~np.isfinite(truth) | (truth <= 0), truth)
        truth_artist = axes[1, 0].imshow(masked_truth, cmap=depth_cmap, norm=depth_norm)
        axes[1, 0].set_title("Ground-truth depth", loc="left", fontweight="bold")
        valid = np.isfinite(prediction) & (prediction > 0) & np.isfinite(truth) & (truth > 0)
        error = np.ma.masked_where(~valid, np.abs(prediction - truth))
        error_artist = axes[1, 1].imshow(error, cmap=error_cmap, vmin=0, vmax=args.error_max)
        axes[1, 1].set_title("Absolute prediction error", loc="left", fontweight="bold")
        error_bar = fig.colorbar(error_artist, cax=fig.add_subplot(grid[1, 3]),
                                extend="max" if np.any(error.compressed() > args.error_max) else "neither")
        error_bar.set_label("Absolute error (m)")
        fig.colorbar(truth_artist, cax=fig.add_subplot(grid[1, 1]),
                     extend="max" if values.max() > depth_max else "neither").set_label("Depth (m)")
        metrics = depth_metrics(prediction, truth)
        footnote += f"\nAbsRel {metrics['abs_rel']:.2%}  |  RMSE {metrics['rmse_m']:.3f} m  |  delta < 1.25: {metrics['delta1']:.2%}"
    else:
        for axis in axes[1]:
            axis.text(.5, .5, "No supplied depth ground truth", ha="center", va="center", transform=axis.transAxes)
        footnote = "Raw metric prediction. No ground-truth depth is supplied for this sequence."
    for axis in axes.ravel():
        axis.set_axis_off()
    fig.colorbar(depth_artist, cax=fig.add_subplot(grid[0, 3]),
                 extend="max" if values.max() > depth_max else "neither").set_label("Depth (m)")
    model = "learned depth"
    manifest_path = args.pred_root / args.sequence / "manifest.json"
    if manifest_path.exists():
        model = json.loads(manifest_path.read_text())["signature"]["model"].replace("_", " ").title()
    fig.suptitle(f"{args.sequence} · {frame.stem} · {model}", fontsize=17, fontweight="bold")
    fig.supxlabel(footnote, fontsize=10)
    output = args.output or ROOT / "results/figures/part2" / f"{args.sequence}_{frame.stem}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
