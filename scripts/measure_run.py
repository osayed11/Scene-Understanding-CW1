"""Score one or more runs and append rows to results/results.csv (the one measuring script).

usage: python scripts/measure_run.py --sequence tum_fr3 --method baseline --run 1 [--traj path] [--rpe-delta 10]
"""
import argparse, re, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS, ROOT, load_sequence, run_id
from metrics import evaluate
from trajectory import load_tum


def time_per_frame_from_log(log: Path, n_frames: int) -> float:
    if not log.exists():
        return float("nan")
    m = re.search(r"wall_seconds=([\d.]+)", log.read_text())
    return float(m.group(1)) / n_frames if m else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--settings", nargs="*", default=[], help="key=value pairs")
    ap.add_argument("--traj", help="override trajectory path")
    ap.add_argument("--rpe-delta", type=int, default=10)
    ap.add_argument("--max-dt", type=float, default=0.02)
    ap.add_argument("--csv", default=str(RESULTS / "results.csv"))
    a = ap.parse_args()

    settings = dict(kv.split("=", 1) for kv in a.settings)
    rid = run_id(a.sequence, a.method, settings, a.run)
    seq = load_sequence(a.sequence)
    est = load_tum(a.traj or RESULTS / "trajectories" / f"{rid}.txt")
    gt = load_tum(ROOT / seq["groundtruth"])
    tpf = time_per_frame_from_log(RESULTS / "logs" / f"{rid}.log", len(gt))
    m = evaluate(est, gt, rpe_delta=a.rpe_delta, max_dt=a.max_dt, time_per_frame=tpf)

    row = {"run_id": rid, "sequence": a.sequence, "method": a.method,
           "settings": ";".join(f"{k}={v}" for k, v in sorted(settings.items())), "run": a.run,
           "rpe_delta": a.rpe_delta, **m.as_row()}
    csv = Path(a.csv)
    df = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    df = df[df.get("run_id", pd.Series(dtype=str)) != rid] if len(df) else df
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv, index=False)
    print(pd.DataFrame([row]).T.to_string(header=False))


if __name__ == "__main__":
    main()
