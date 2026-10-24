"""Part 1 (Section 10): how well the map points agree with the sequence's depth, where it has one.

For every observation in a run's map_points.npz (scripts/extract_map.py): the triangulated depth of the point
in that keyframe against the ground-truth depth image at the keypoint (TUM: depth camera; Replica: exact).
Monocular maps have an arbitrary scale, so one scale is fitted per run (median of gt / map depth) before the
errors; RGB-D maps should fit a scale of ~1. Reported per run, and bucketed by parallax and by distance
(reused in Part 4 against the learned depth).

usage: python scripts/part1/map_depth_agreement.py [--runs tum_fr3__baseline__speed=1__run01 ...] (default: every run with a map)
output: results/tables/table1_map_depth.csv (one row per run), results/tables/table1_map_depth_buckets.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from part2.common import load_depth_gt, load_frame_records  # noqa: E402

PARALLAX_BINS = [0, 1, 2, 5, 10, 20, 180]
DISTANCE_BINS = [0, 1, 2, 3, 5, 10, 1000]


def samples_for_run(run_dir: Path, sequence: str, frames) -> pd.DataFrame | None:
    npz = run_dir / "map" / "map_points.npz"
    if not npz.exists():
        return None
    z = np.load(npz)
    frame_t = np.array([f.timestamp for f in frames])
    kf_frame = np.array([int(np.argmin(np.abs(frame_t - t))) for t in z["kf_timestamp"]])
    if np.abs(frame_t[kf_frame] - z["kf_timestamp"]).max() > 0.02:
        raise RuntimeError(f"{run_dir}: keyframe timestamps do not match the sequence's frames")
    rows = []
    depth_cache = {}
    for kf_i in np.unique(z["obs_kf"]):
        fr = frames[kf_frame[kf_i]]
        if fr.depth_gt is None:
            continue
        if kf_i not in depth_cache:
            depth_cache[kf_i] = load_depth_gt(sequence, fr)
        d = depth_cache[kf_i]
        sel = np.where(z["obs_kf"] == kf_i)[0]
        u = np.clip(np.rint(z["obs_u"][sel]).astype(int), 0, d.shape[1] - 1)
        v = np.clip(np.rint(z["obs_v"][sel]).astype(int), 0, d.shape[0] - 1)
        rows.append(pd.DataFrame({
            "obs": sel, "kf_img_id": z["kf_img_id"][kf_i], "frame_index": fr.index,
            "u": z["obs_u"][sel], "v": z["obs_v"][sel],
            "depth_map": z["obs_depth_map"][sel], "depth_gt": d[v, u],
            "parallax_deg": z["point_parallax_deg"][z["obs_point"][sel]],
        }))
    if not rows:
        return None
    df = pd.concat(rows, ignore_index=True)
    return df[np.isfinite(df.depth_gt) & (df.depth_gt > 0) & (df.depth_map > 0)]


def summarise(df: pd.DataFrame) -> dict:
    scale = float(np.median(df.depth_gt / df.depth_map))  # map units -> metres
    rel = (df.depth_map * scale - df.depth_gt) / df.depth_gt
    return {"n_obs": len(df), "scale_fit": scale,
            "abs_rel_median": float(np.median(np.abs(rel))), "abs_rel_mean": float(np.mean(np.abs(rel))),
            "within_10pct": float(np.mean(np.abs(rel) < 0.10)), "within_25pct": float(np.mean(np.abs(rel) < 0.25)),
            "signed_rel_median": float(np.median(rel))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", help="run ids (default: every run under results/logs with a map)")
    a = ap.parse_args()
    run_dirs = [ROOT / "results/logs" / r for r in a.runs] if a.runs else sorted((ROOT / "results/logs").glob("*__run*"))
    frames_cache, per_run, buckets = {}, [], []
    for run_dir in run_dirs:
        sequence = run_dir.name.split("__")[0]
        if sequence not in ("tum_fr3", "replica_room0"):
            continue  # no depth ground truth
        if sequence not in frames_cache:
            frames_cache[sequence] = load_frame_records(sequence)
        df = samples_for_run(run_dir, sequence, frames_cache[sequence])
        if df is None:
            continue
        s = summarise(df)
        per_run.append({"run_id": run_dir.name, "sequence": sequence, **s})
        rel = np.abs(df.depth_map * s["scale_fit"] - df.depth_gt) / df.depth_gt
        for kind, bins, col in (("parallax_deg", PARALLAX_BINS, df.parallax_deg), ("distance_m", DISTANCE_BINS, df.depth_gt)):
            cat = pd.cut(col, bins)
            for b, g in rel.groupby(cat, observed=True):
                buckets.append({"run_id": run_dir.name, "sequence": sequence, "bucket_by": kind, "bucket": str(b),
                                "n_obs": len(g), "abs_rel_median": float(g.median()), "within_10pct": float((g < 0.1).mean())})
        print(f"{run_dir.name}: {s}")
    out = ROOT / "results/tables"
    pd.DataFrame(per_run).to_csv(out / "table1_map_depth.csv", index=False)
    pd.DataFrame(buckets).to_csv(out / "table1_map_depth_buckets.csv", index=False)
    print(f"wrote {out/'table1_map_depth.csv'} ({len(per_run)} runs) and table1_map_depth_buckets.csv")


if __name__ == "__main__":
    main()
