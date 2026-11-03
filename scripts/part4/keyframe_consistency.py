"""Part 4, job 2: does a map point's learned depth agree across the keyframes which observe it?

Input: results/logs/<run_id>/map/obs_learned.csv.gz from scripts/part4/map_vs_learned.py (one row per observation
of a map point by a keyframe: the point's triangulated depth in that keyframe, scaled to metres with the scale
estimated from the learned depth, and the learned depth at the keypoint).
Per point with at least two observations, the ratio r_k = learned_k / triangulated_k is one number per keyframe. If
the network were consistent, every keyframe would give the same ratio (the point is one 3D point, so the ratios of
the two depths are equal up to the network's error). Its spread is
  spread = 1.4826 * MAD(r_k) / median(r_k)      robust relative spread of the learned depth across keyframes
  range  = (max r_k - min r_k) / median(r_k)
Where the sequence has depth ground truth, the spread is compared with the learned depth's error, and two ways of
using the learned depth are scored per point: single (the error at each observation, as a live method sees it) and
fused (the median ratio over the keyframes, applied to each keyframe: what a method which waits for several
keyframes could use). 'drop above' rows score the policy of ignoring the learned depth of points whose spread is
above a threshold: the share of points dropped and the learned error of the points kept.

usage: python scripts/part4/keyframe_consistency.py [--runs <run_id> ...]
outputs: results/tables/table4_consistency.csv, table4_consistency_buckets.csv, table4_consistency_summary.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SPREAD_BINS = [0, .02, .05, .10, .20, np.inf]
NOBS_BINS = [1, 2, 3, 5, 8, 12, 20, 1000]
SPAN_BINS = [-1, 5, 15, 40, 100, 300, 100000]  # frames between the first and the last observing keyframe
PARALLAX_BINS = [0, 1, 2, 5, 10, 20, 180]
DROP_THRESHOLDS = [.02, .05, .10, .20]


def per_point(obs: pd.DataFrame) -> pd.DataFrame:
    obs = obs[obs.n_obs_point >= 2].copy()
    obs["ratio"] = obs.learned_m / obs.tri_m
    has_gt = np.isfinite(obs.depth_gt) & (obs.depth_gt > 0)
    g = obs.groupby("point")
    pts = pd.DataFrame({
        "n_obs": g.size(),
        "ratio_med": g.ratio.median(),
        "mad": g.ratio.apply(lambda r: np.median(np.abs(r - r.median()))),
        "rng": g.ratio.apply(lambda r: r.max() - r.min()),
        "span_frames": g.kf_img_id.max() - g.kf_img_id.min(),
        "parallax_deg": g.parallax_deg.first(),
        "sigma_rel": (obs.sigma_m / obs.learned_m).groupby(obs.point).median(),
    })
    pts["spread"] = 1.4826 * pts.mad / pts.ratio_med
    pts["range"] = pts.rng / pts.ratio_med
    # errors against ground truth, per point (median over its observations)
    if has_gt.any():
        o = obs[has_gt].copy()
        o["fused_m"] = o.tri_m * o.point.map(pts.ratio_med)
        o["fused_vs_gt"] = (o.fused_m - o.depth_gt) / o.depth_gt
        gg = o.groupby("point")
        pts["learned_err"] = gg.learned_vs_gt.apply(lambda e: np.median(np.abs(e)))
        pts["fused_err"] = gg.fused_vs_gt.apply(lambda e: np.median(np.abs(e)))
        pts["tri_err"] = gg.tri_vs_gt.apply(lambda e: np.median(np.abs(e)))
        pts["tri_err_gtscale"] = gg.tri_vs_gt_gtscale.apply(lambda e: np.median(np.abs(e)))
    return pts.reset_index()


def summarise(pts: pd.DataFrame) -> dict:
    out = {"n_points": len(pts), "obs_per_point_median": float(pts.n_obs.median()),
           "spread_median": float(pts.spread.median()), "spread_p90": float(pts.spread.quantile(.9)),
           "range_median": float(pts.range.median()),
           "spread_below_2pct": float((pts.spread < .02).mean()), "spread_below_5pct": float((pts.spread < .05).mean()),
           "spread_below_10pct": float((pts.spread < .10).mean()),
           "spread_below_sigma": float((pts.spread < pts.sigma_rel).mean())}
    if "learned_err" in pts:
        ok = pts.learned_err.notna()
        p = pts[ok]
        out.update(n_points_gt=len(p), learned_err_median=float(p.learned_err.median()),
                   fused_err_median=float(p.fused_err.median()), tri_err_median=float(p.tri_err.median()),
                   tri_err_gtscale_median=float(p.tri_err_gtscale.median()),
                   fused_better_than_single=float((p.fused_err < p.learned_err).mean()),
                   spread_vs_learned_err_spearman=float(p.spread.corr(p.learned_err, method="spearman")))
        for thr in DROP_THRESHOLDS:
            keep = p.spread <= thr
            out[f"drop_above_{int(thr*100):02d}pct_share_dropped"] = float(1 - keep.mean())
            out[f"drop_above_{int(thr*100):02d}pct_learned_err_kept"] = float(p.learned_err[keep].median())
            out[f"drop_above_{int(thr*100):02d}pct_learned_err_dropped"] = float(p.learned_err[~keep].median()) if (~keep).any() else np.nan
    return out


def buckets(pts: pd.DataFrame) -> list[dict]:
    cuts = {"spread": pd.cut(pts.spread, SPREAD_BINS, right=False),
            "n_obs": pd.cut(pts.n_obs, NOBS_BINS, right=False),
            "span_frames": pd.cut(pts.span_frames, SPAN_BINS),
            "parallax_deg": pd.cut(pts.parallax_deg, PARALLAX_BINS)}
    rows = []
    for kind, cat in cuts.items():
        for b, g in pts.groupby(cat, observed=True):
            row = {"bucket_by": kind, "bucket": str(b), "n_points": len(g), "share": len(g) / len(pts),
                   "spread_median": float(g.spread.median()), "range_median": float(g.range.median())}
            if "learned_err" in g and g.learned_err.notna().any():
                gg = g[g.learned_err.notna()]
                row.update(learned_err_median=float(gg.learned_err.median()), fused_err_median=float(gg.fused_err.median()),
                           tri_err_median=float(gg.tri_err.median()), tri_err_gtscale_median=float(gg.tri_err_gtscale.median()),
                           learned_closer=float((gg.learned_err < gg.tri_err).mean()))
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*")
    a = ap.parse_args()
    run_dirs = [ROOT / "results/logs" / r for r in a.runs] if a.runs else sorted((ROOT / "results/logs").glob("*__run*"))
    per_run, per_bucket = [], []
    for run_dir in run_dirs:
        f = run_dir / "map" / "obs_learned.csv.gz"
        if not f.exists():
            continue
        sequence, method, settings, run = run_dir.name.split("__")
        pts = per_point(pd.read_csv(f))
        s = summarise(pts)
        ids = {"run_id": run_dir.name, "sequence": sequence, "method": method, "settings": settings}
        per_run.append({**ids, **s})
        per_bucket += [{**ids, **b} for b in buckets(pts)]
        pts.to_csv(run_dir / "map" / "points_consistency.csv.gz", index=False, float_format="%.6g")
        print(f"{run_dir.name}: points={s['n_points']} spread med={s['spread_median']:.3f} p90={s['spread_p90']:.3f} "
              f"<5%={s['spread_below_5pct']:.2f} learned_err={s.get('learned_err_median', np.nan):.3f} "
              f"fused_err={s.get('fused_err_median', np.nan):.3f} tri_err={s.get('tri_err_median', np.nan):.3f} "
              f"rho(spread,err)={s.get('spread_vs_learned_err_spearman', np.nan):.2f}")
    out = ROOT / "results/tables"
    runs = pd.DataFrame(per_run)
    runs.to_csv(out / "table4_consistency.csv", index=False)
    pd.DataFrame(per_bucket).to_csv(out / "table4_consistency_buckets.csv", index=False)
    num = runs.select_dtypes("number").columns
    summary = runs.groupby(["sequence", "method", "settings"])[num].agg(["mean", "std"])
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary.insert(0, "n_runs", runs.groupby(["sequence", "method", "settings"]).size())
    summary.reset_index().to_csv(out / "table4_consistency_summary.csv", index=False)
    print(f"wrote table4_consistency.csv ({len(runs)} runs), table4_consistency_buckets.csv, table4_consistency_summary.csv")


if __name__ == "__main__":
    main()
