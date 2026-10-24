"""Score one or more runs and append rows to results/results.csv (the one measuring script).

usage: python scripts/measure_run.py --sequence tum_fr3 --method baseline --run 1 [--traj path] [--rpe-delta 10]
"""
import argparse, re, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS, ROOT, count_frames, load_sequence, run_id
from metrics import evaluate
from trajectory import load_tum


def wall_per_frame(log: Path, n_frames: int) -> float:
    """Wall-clock seconds of the whole run / frames. With --speed 1 this is bounded below by the playback."""
    if not log.exists():
        return float("nan")
    m = re.search(r"wall_seconds=([\d.]+)", log.read_text())
    return float(m.group(1)) / n_frames if m else float("nan")


def tracking_time_per_frame(tracking_log: Path) -> float:
    """Mean of pySLAM's own per-frame 'Tracking: elapsed_time' (the compute time of the tracking thread per
    frame, independent of the playback speed)."""
    if not tracking_log.exists():
        return float("nan")
    t = [float(m) for m in re.findall(r"Tracking: elapsed_time:\s+([\d.eE+-]+)", tracking_log.read_text())]
    return sum(t) / len(t) if t else float("nan")


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
    n_frames = count_frames(seq)
    log_dir = RESULTS / "logs" / rid
    tpf = tracking_time_per_frame(log_dir / "pyslam_logs" / "tracking.log")
    m = evaluate(est, gt, n_frames=n_frames, rpe_delta=a.rpe_delta, max_dt=a.max_dt, time_per_frame=tpf)

    row = {"run_id": rid, "sequence": a.sequence, "method": a.method,
           "settings": ";".join(f"{k}={v}" for k, v in sorted(settings.items())), "run": a.run,
           "rpe_delta": a.rpe_delta, **m.as_row(),
           "wall_per_frame": wall_per_frame(log_dir / "stdout.log", n_frames)}
    csv = Path(a.csv)
    df = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    df = df[df.get("run_id", pd.Series(dtype=str)) != rid] if len(df) else df
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv, index=False)
    print(pd.DataFrame([row]).T.to_string(header=False))


if __name__ == "__main__":
    main()
