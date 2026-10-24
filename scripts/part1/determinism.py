"""Part 1: run-to-run spread and how many runs a comparison needs.

Groups results/results.csv by (sequence, method, settings) and reports, per measure, the mean, std, min,
max and number of runs. Also prints, for each sequence, the number of runs needed so that a difference of
`--min-diff` (as a fraction of the baseline mean ATE) is 2 standard errors apart, i.e. n = (2*std/diff)^2.
It also answers "do repeated runs give the same trajectory?" directly: for every pair of runs of a configuration,
the trajectories are matched by timestamp and the largest position difference is reported (0 = identical).
The same table compares the C++ and Python cores (methods baseline and baseline_py).

usage: python scripts/part1/determinism.py [--csv results/results.csv] [--min-diff 0.1]
output: results/tables/table1_summary.csv (and the table on stdout)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from trajectory import associate, load_tum  # noqa: E402
MEASURES = ["ate_noscale", "ate_scale", "scale", "rpe_trans", "rpe_rot_deg", "tracked", "time_per_frame"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=ROOT / "results/results.csv")
    ap.add_argument("--min-diff", type=float, default=0.10, help="difference to resolve, as a fraction of mean ATE")
    ap.add_argument("--out", default=ROOT / "results/tables/table1_summary.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    df["settings"] = df["settings"].fillna("")
    keys = ["sequence", "method", "settings"]
    g = df.groupby(keys)[MEASURES]
    summary = g.agg(["mean", "std", "min", "max"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary["n_runs"] = g.size()
    summary = summary.reset_index()

    # runs needed: for a difference d = min_diff * mean ATE_scale, want d >= 2 * std / sqrt(n)
    std, mean = summary["ate_scale_std"], summary["ate_scale_mean"]
    summary["runs_needed_for_min_diff"] = np.ceil((2 * std / (a.min_diff * mean)) ** 2)

    # are the trajectories of repeated runs identical? max position difference over all pairs of runs
    def max_pair_diff(run_ids):
        trajs = [load_tum(ROOT / "results/trajectories" / f"{r}.txt") for r in run_ids]
        worst = 0.0
        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                x, y = associate(trajs[i], trajs[j], max_dt=1e-6)
                if len(x) == 0:
                    return float("inf")
                worst = max(worst, float(np.abs(x[:, 1:4] - y[:, 1:4]).max()))
                worst = max(worst, abs(len(trajs[i]) - len(trajs[j])) and float("inf"))
        return worst

    summary["max_pairwise_position_diff"] = [
        max_pair_diff(df[(df.sequence == r.sequence) & (df.method == r.method) & (df.settings == r.settings)].run_id)
        if r.n_runs > 1 else float("nan") for r in summary.itertuples()]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(a.out, index=False)
    show = keys + ["n_runs", "ate_scale_mean", "ate_scale_std", "scale_mean", "rpe_trans_mean", "rpe_trans_std",
                   "tracked_mean", "time_per_frame_mean", "runs_needed_for_min_diff", "max_pairwise_position_diff"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(summary[show].to_string(index=False))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
