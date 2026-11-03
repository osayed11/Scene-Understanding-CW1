"""Part 4, job 1: the triangulated depth of the map points against their learned depth, per observation.

For every observation in a run's map (results/logs/<run_id>/map/map_points.npz: a point seen by a keyframe at a
keypoint) three depths are paired at that keypoint:
  depth_map   the point's depth in the keyframe's camera, from triangulation + bundle adjustment (map units)
  learned_m   the Part 2 network's depth at the keypoint (median of a 7x7 patch; sigma = its spread), metres
  depth_gt    the sequence's depth image, same sampler (TUM: depth camera, Replica: exact; KITTI: none)
Monocular maps have an arbitrary scale. It is estimated from the learned depth alone (Part 2's
estimate_map_scale: median ratio with parallax/sigma gates and MAD rejection), never from ground truth, and the
ground-truth scale is only reported beside it as a check. Errors are relative to the reference depth:
  tri_vs_learned   (map * scale_learned - learned) / learned     the disagreement, available on all sequences
  tri_vs_gt        (map * scale_learned - gt) / gt               what the map is worth in metres with that scale
  tri_vs_gt_gtscale(map * scale_gt - gt) / gt                    the map's own geometry (Part 1's number)
  learned_vs_gt    (learned - gt) / gt
and 'learned_closer' is the share of observations where learned is nearer the truth than the scaled map point.
Everything is bucketed by parallax, by distance and by position in the image (centre / middle / border, by the
larger of the normalised offsets from the principal point).

usage: python scripts/part4/map_vs_learned.py [--runs <run_id> ...] [--pred-root depth_maps/da3_metric_large_cpu_float32]
outputs: results/tables/table4_agreement.csv          one row per run
         results/tables/table4_agreement_buckets.csv  one row per run x bucket
         results/tables/table4_agreement_summary.csv  per sequence x method, mean/std over runs
         results/logs/<run_id>/map/obs_learned.csv.gz one row per observation (for job 2, cross-keyframe agreement)
         results/logs/<run_id>/map/map_samples.npz    Part 2's --map-samples input (learned/triangulated pairs)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from depth import depth_at_keypoints  # noqa: E402
from part2.common import load_depth_gt, load_frame_records, load_prediction  # noqa: E402
from part2.keypoint_depth import estimate_map_scale  # noqa: E402

PARALLAX_BINS = [0, 1, 2, 5, 10, 20, 180]
DISTANCE_BINS = {"indoor": [0, 1, 2, 3, 5, 10, 1000], "kitti": [0, 5, 10, 20, 40, 80, 1000]}
POSITION_BINS = [0, 1 / 3, 2 / 3, 1.01]
POSITION_LABELS = ["centre", "middle", "border"]
PATCH_RADIUS = 3  # Part 2's keypoint sampler (keypoint_depth.py --radius)


def observations(run_dir: Path, sequence: str, frames, pred_root: Path) -> pd.DataFrame | None:
    npz = run_dir / "map" / "map_points.npz"
    if not npz.exists():
        return None
    z = np.load(npz)
    frame_t = np.array([f.timestamp for f in frames])
    kf_frame = np.array([int(np.argmin(np.abs(frame_t - t))) for t in z["kf_timestamp"]])
    if np.abs(frame_t[kf_frame] - z["kf_timestamp"]).max() > 0.02:
        raise RuntimeError(f"{run_dir.name}: keyframe timestamps do not match the sequence's frames")
    n_obs_point = np.bincount(z["obs_point"], minlength=len(z["point_id"]))
    rows = []
    for kf_i in np.unique(z["obs_kf"]):
        fr = frames[kf_frame[kf_i]]
        pred = load_prediction(pred_root, sequence, fr)
        h, w = pred.shape
        sel = np.where(z["obs_kf"] == kf_i)[0]
        kps = np.column_stack([z["obs_u"][sel], z["obs_v"][sel]])
        learned, sigma = depth_at_keypoints(pred, kps, PATCH_RADIUS)
        gt = load_depth_gt(sequence, fr)
        depth_gt = depth_at_keypoints(gt, kps, PATCH_RADIUS)[0] if gt is not None else np.full(len(sel), np.nan)
        pos = np.maximum(np.abs(z["obs_u"][sel] - float(z["cx"])) / (w / 2), np.abs(z["obs_v"][sel] - float(z["cy"])) / (h / 2))
        rows.append(pd.DataFrame({
            "obs": sel, "point": z["obs_point"][sel], "kf": kf_i, "kf_img_id": z["kf_img_id"][kf_i],
            "frame_index": fr.index, "u": z["obs_u"][sel], "v": z["obs_v"][sel], "position": pos,
            "depth_map": z["obs_depth_map"][sel], "learned_m": learned, "sigma_m": sigma, "depth_gt": depth_gt,
            "parallax_deg": z["point_parallax_deg"][z["obs_point"][sel]], "n_obs_point": n_obs_point[z["obs_point"][sel]],
            "reproj_px": z["obs_reproj_px"][sel],
        }))
    df = pd.concat(rows, ignore_index=True)
    return df[np.isfinite(df.learned_m) & (df.learned_m > 0) & (df.depth_map > 0)].reset_index(drop=True)


def rel_stats(rel: pd.Series, prefix: str) -> dict:
    a = np.abs(rel)
    return {f"{prefix}_abs_rel_median": float(a.median()), f"{prefix}_abs_rel_mean": float(a.mean()),
            f"{prefix}_within_10pct": float((a < .10).mean()), f"{prefix}_within_25pct": float((a < .25).mean()),
            f"{prefix}_signed_median": float(rel.median())}


def score(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Per-run scalars and the per-observation relative errors (columns added to df)."""
    sc = estimate_map_scale(df.learned_m.values, df.depth_map.values, df.sigma_m.values, df.parallax_deg.values)
    s_learned = sc["scale_metres_per_map_unit"]  # metres per map unit, from the network alone
    has_gt = np.isfinite(df.depth_gt) & (df.depth_gt > 0)
    s_gt = float(np.median(df.depth_gt[has_gt] / df.depth_map[has_gt])) if has_gt.any() else np.nan
    df["tri_m"] = df.depth_map * s_learned
    df["tri_vs_learned"] = (df.tri_m - df.learned_m) / df.learned_m
    df["agrees_within_sigma"] = np.abs(df.tri_m - df.learned_m) <= df.sigma_m
    df["tri_vs_gt"] = (df.tri_m - df.depth_gt) / df.depth_gt
    df["tri_vs_gt_gtscale"] = (df.depth_map * s_gt - df.depth_gt) / df.depth_gt
    df["learned_vs_gt"] = (df.learned_m - df.depth_gt) / df.depth_gt
    df["learned_closer"] = np.abs(df.learned_vs_gt) < np.abs(df.tri_vs_gt)
    out = {"n_obs": len(df), "n_points": int(df.point.nunique()), "n_keyframes": int(df.kf.nunique()),
           "scale_m_per_unit_learned": s_learned, "scale_m_per_unit_gt": s_gt,
           "scale_learned_over_gt": s_learned / s_gt if np.isfinite(s_gt) else np.nan,
           "scale_status": sc["map_scale_status"], "scale_support": sc["n_scale_support"],
           "scale_ratio_mad_rel": sc["scale_ratio_mad"] * s_learned if np.isfinite(sc["scale_ratio_mad"]) else np.nan,
           "agrees_within_sigma": float(df.agrees_within_sigma.mean()),
           **rel_stats(df.tri_vs_learned, "tri_vs_learned")}
    if has_gt.any():
        g = df[has_gt]
        out.update(n_obs_gt=len(g), **rel_stats(g.tri_vs_gt, "tri_vs_gt"), **rel_stats(g.tri_vs_gt_gtscale, "tri_vs_gt_gtscale"),
                   **rel_stats(g.learned_vs_gt, "learned_vs_gt"), learned_closer=float(g.learned_closer.mean()))
    return out, df


def buckets(df: pd.DataFrame, sequence: str) -> list[dict]:
    dist_bins = DISTANCE_BINS["kitti" if sequence.startswith("kitti") else "indoor"]
    dist = df.depth_gt.where(np.isfinite(df.depth_gt), df.learned_m)
    cuts = {"parallax_deg": pd.cut(df.parallax_deg, PARALLAX_BINS),
            "distance_m": pd.cut(dist, dist_bins),
            "position": pd.cut(df.position, POSITION_BINS, labels=POSITION_LABELS)}
    rows = []
    for kind, cat in cuts.items():
        for b, g in df.groupby(cat, observed=True):
            row = {"bucket_by": kind, "bucket": str(b), "n_obs": len(g), "share": len(g) / len(df),
                   "tri_vs_learned_abs_rel_median": float(np.abs(g.tri_vs_learned).median()),
                   "agrees_within_sigma": float(g.agrees_within_sigma.mean())}
            gg = g[np.isfinite(g.depth_gt)]
            if len(gg):
                row.update(tri_vs_gt_abs_rel_median=float(np.abs(gg.tri_vs_gt).median()),
                           tri_vs_gt_gtscale_abs_rel_median=float(np.abs(gg.tri_vs_gt_gtscale).median()),
                           learned_vs_gt_abs_rel_median=float(np.abs(gg.learned_vs_gt).median()),
                           learned_closer=float(gg.learned_closer.mean()))
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", help="run ids (default: every run under results/logs with a map)")
    ap.add_argument("--pred-root", type=Path, default=ROOT / "depth_maps/da3_metric_large_cpu_float32")
    a = ap.parse_args()
    run_dirs = [ROOT / "results/logs" / r for r in a.runs] if a.runs else sorted((ROOT / "results/logs").glob("*__run*"))
    frames_cache, per_run, per_bucket = {}, [], []
    for run_dir in run_dirs:
        sequence, method, settings, run = run_dir.name.split("__")
        if sequence not in frames_cache:
            frames_cache[sequence] = load_frame_records(sequence)
        df = observations(run_dir, sequence, frames_cache[sequence], a.pred_root)
        if df is None:
            continue
        s, df = score(df)
        ids = {"run_id": run_dir.name, "sequence": sequence, "method": method, "settings": settings}
        per_run.append({**ids, **s})
        per_bucket += [{**ids, **b} for b in buckets(df, sequence)]
        df.to_csv(run_dir / "map" / "obs_learned.csv.gz", index=False, float_format="%.6g")
        np.savez(run_dir / "map" / "map_samples.npz", learned_depth_m=df.learned_m.values,
                 triangulated_depth_map=df.depth_map.values, sigma_m=df.sigma_m.values, parallax_deg=df.parallax_deg.values)
        print(f"{run_dir.name}: n={s['n_obs']} scale learned={s['scale_m_per_unit_learned']:.3f} gt={s['scale_m_per_unit_gt']:.3f} "
              f"tri_vs_learned={s['tri_vs_learned_abs_rel_median']:.3f} tri_vs_gt={s.get('tri_vs_gt_abs_rel_median', np.nan):.3f} "
              f"learned_vs_gt={s.get('learned_vs_gt_abs_rel_median', np.nan):.3f} learned_closer={s.get('learned_closer', np.nan):.2f}")
    out = ROOT / "results/tables"
    runs = pd.DataFrame(per_run)
    runs.to_csv(out / "table4_agreement.csv", index=False)
    pd.DataFrame(per_bucket).to_csv(out / "table4_agreement_buckets.csv", index=False)
    num = runs.select_dtypes("number").columns
    summary = runs.groupby(["sequence", "method", "settings"])[num].agg(["mean", "std"])
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary.insert(0, "n_runs", runs.groupby(["sequence", "method", "settings"]).size())
    summary.reset_index().to_csv(out / "table4_agreement_summary.csv", index=False)
    print(f"wrote table4_agreement.csv ({len(runs)} runs), table4_agreement_buckets.csv, table4_agreement_summary.csv in {out}")


if __name__ == "__main__":
    main()
