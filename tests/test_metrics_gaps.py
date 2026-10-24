import numpy as np
from scipy.spatial.transform import Rotation as R
from metrics import evaluate, rpe
from trajectory import to_matrices


def synthetic(n=200):
    t = np.arange(n) * 0.05
    xyz = np.stack([np.sin(t), np.cos(t), 0.1 * t], 1)
    return np.column_stack([t, xyz, R.from_euler("z", t).as_quat()])


def test_rpe_skips_pairs_across_a_gap():
    gt = synthetic()
    est = np.delete(gt, slice(50, 120), axis=0)      # tracking lost for 70 frames
    est[120 - 70:, 1:4] += 1.0                       # and the map jumped when it relocalised
    T_e, T_g = to_matrices(est), to_matrices(np.delete(gt, slice(50, 120), axis=0))
    naive, _ = rpe(T_e, T_g, 5)
    fixed, _ = rpe(T_e, T_g, 5, t=est[:, 0], frame_dt=0.05)
    assert naive > 0.1 and fixed < 1e-9           # the jump pollutes the naive RPE only


def test_tracked_counts_every_pose_and_n_frames_wins():
    gt = synthetic()
    m = evaluate(gt[::2], gt, n_frames=100)          # 100 frames processed, every one tracked
    assert m.tracked == 1.0 and m.n_est == 100 and m.n_assoc == 100
