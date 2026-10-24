"""Run one (sequence, method) configuration N times, then score every run.

usage: python scripts/run_config.py --sequence tum_fr3 --method baseline --runs 5 [--dry-run]
Trust-parameter sweeps: every value listed in the method YAML is run `--runs` times.
"""
import argparse, itertools, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_method, load_sequence
from runner import run_pyslam


def expand(settings: dict):
    keys = list(settings)
    vals = [v if isinstance(v, list) else [v] for v in settings.values()]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    seq, meth = load_sequence(a.sequence), load_method(a.method)
    failed = []
    for settings in expand(meth.get("settings") or {}):
        for r in range(1, a.runs + 1):
            try:
                run_pyslam(seq, meth, settings, r, dry_run=a.dry_run)
                if not a.dry_run:
                    subprocess.run([sys.executable, "scripts/measure_run.py", "--sequence", a.sequence,
                                    "--method", a.method, "--run", str(r), "--settings",
                                    *[f"{k}={v}" for k, v in settings.items()]], check=True)
            except Exception as ex:  # a failed run is a result too: report it and go on with the batch
                print(f"[run_config] FAILED {a.sequence} {a.method} {settings} run {r}: {ex}", file=sys.stderr)
                failed.append((settings, r))
    if failed:
        print(f"[run_config] {len(failed)} failed run(s): {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
