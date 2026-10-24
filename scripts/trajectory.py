"""Trajectory utilities.

TUM format: one pose per line, `t tx ty tz qx qy qz qw`, lines starting with '#' ignored.
All functions work on numpy arrays; poses are (N, 8) float arrays in TUM column order.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def load_tum(path) -> np.ndarray:
    """Load a TUM trajectory file -> (N, 8) array sorted by timestamp."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.replace(",", " ").split()]
            if len(vals) < 8:
                raise ValueError(f"{path}: expected 8 columns, got {len(vals)}: {line}")
            rows.append(vals[:8])
    arr = np.asarray(rows, dtype=np.float64)
    return arr[np.argsort(arr[:, 0])]


def save_tum(path, traj: np.ndarray) -> None:
    np.savetxt(path, traj, fmt="%.6f", header="timestamp tx ty tz qx qy qz qw")


def associate(est: np.ndarray, gt: np.ndarray, max_dt: float = 0.02, offset: float = 0.0):
    """Match each estimated pose to the nearest ground-truth pose in time.

    Returns (est_matched, gt_matched), both (M, 8), M <= len(est).
    """
    gt_t = gt[:, 0]
    t = est[:, 0] + offset
    idx = np.clip(np.searchsorted(gt_t, t), 1, len(gt_t) - 1)
    left, right = idx - 1, idx
    pick = np.where(np.abs(gt_t[left] - t) < np.abs(gt_t[right] - t), left, right)
    ok = np.abs(gt_t[pick] - t) <= max_dt
    return est[ok], gt[pick[ok]]


def to_matrices(traj: np.ndarray) -> np.ndarray:
    """(N, 8) TUM rows -> (N, 4, 4) homogeneous poses."""
    T = np.tile(np.eye(4), (len(traj), 1, 1))
    T[:, :3, :3] = R.from_quat(traj[:, 4:8]).as_matrix()
    T[:, :3, 3] = traj[:, 1:4]
    return T


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool):
    """Least-squares similarity/rigid transform mapping src -> dst (both (N, 3)).

    Returns (s, R, t) such that dst ~= s * R @ src + t.
    With with_scale=False, s is fixed to 1 (SE(3) alignment).
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    # OpenBLAS on Apple silicon leaves FP exception flags set, so numpy reports a spurious
    # 'divide by zero in matmul'; the result is finite and identical to einsum (checked).
    with np.errstate(all="ignore"):
        cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    Rm = U @ S @ Vt
    s = (np.trace(np.diag(D) @ S) / xs.var(0).sum()) if with_scale else 1.0
    t = mu_d - s * Rm @ mu_s
    return s, Rm, t
