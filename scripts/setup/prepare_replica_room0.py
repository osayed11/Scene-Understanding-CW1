"""Part 1: Replica room0 ground truth -> TUM format for scripts/measure_run.py.

pySLAM's own Replica reader (pyslam/io/dataset.py, ground_truth.py) reads results/frame*.jpg,
results/depth*.png and traj.txt directly and gives frame i the timestamp i / 25 fps. We write the same
poses with the same timestamps as groundtruth_tum.txt so our measuring script and pySLAM agree.
traj.txt: one 4x4 camera-to-world matrix per line, row-major.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

FPS = 25.0  # ReplicaDataset.fps in pySLAM
root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "data/replica/room0")
T = np.loadtxt(root / "traj.txt").reshape(-1, 4, 4)
t = np.arange(len(T)) / FPS
q = R.from_matrix(T[:, :3, :3]).as_quat()  # x y z w
out = np.column_stack([t, T[:, :3, 3], q])
np.savetxt(root / "groundtruth_tum.txt", out, fmt="%.6f", header="timestamp tx ty tz qx qy qz qw (from traj.txt, 25 fps)")
print(f"wrote {root/'groundtruth_tum.txt'}: {len(out)} poses")
