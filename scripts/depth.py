"""Depth-map IO shared by Parts 2-4.

Two stores for every predicted depth map:
  <name>.npy  float32 metres, lossless: the canonical prediction, used for all the scoring of Part 2
  <name>.png  16-bit, metres * factor: what pySLAM reads as the depth image of an RGB-D sequence
The PNG factor must (a) match DepthMapFactor in the pySLAM settings file of the sequence and (b) cover the
depth range: the largest storable depth is 65535 / factor. 5000 (TUM) caps at 13.1 m, fine indoors, useless
on KITTI, so the factor is per sequence (configs/sequences/*.yaml: depth_factor). Writing a depth above the
cap raises, rather than silently flattening distant scene to the cap.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MAX_U16 = 65535


def max_depth(factor: float) -> float:
    return MAX_U16 / factor


def save_depth(path, depth_m: np.ndarray, factor: float) -> None:
    """Save <path>.npy (float32 metres) and <path>.png (uint16 = metres * factor). `path` has no extension."""
    path = Path(path)
    depth_m = np.asarray(depth_m, dtype=np.float32)
    np.save(path.with_suffix(".npy"), depth_m)
    finite = np.isfinite(depth_m)
    if finite.any() and depth_m[finite].max() > max_depth(factor):
        raise ValueError(f"{path}: depth {depth_m[finite].max():.1f} m exceeds the 16-bit cap {max_depth(factor):.1f} m "
                         f"for factor {factor}; use a smaller factor for this sequence")
    d = np.where(finite, depth_m * factor, 0).round().astype(np.uint16)  # 0 = no depth, as TUM
    cv2.imwrite(str(path.with_suffix(".png")), d)


def load_depth_npy(path) -> np.ndarray:
    return np.load(path)


def load_depth_png(path, factor: float) -> np.ndarray:
    """16-bit PNG -> float32 metres; 0 (no depth) -> nan."""
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(path)
    out = d.astype(np.float32) / factor
    out[d == 0] = np.nan
    return out


def depth_at_keypoints(depth_m: np.ndarray, kps: np.ndarray, radius: int = 0):
    """Read depth at (u, v) keypoints. TODO (Part 2): choose nearest / bilinear / robust patch,
    and return an uncertainty per keypoint alongside the depth."""
    u = np.round(kps[:, 0]).astype(int)
    v = np.round(kps[:, 1]).astype(int)
    h, w = depth_m.shape
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    d = np.full(len(kps), np.nan, dtype=np.float32)
    d[ok] = depth_m[v[ok], u[ok]]
    sigma = np.full_like(d, np.nan)  # TODO
    return d, sigma
