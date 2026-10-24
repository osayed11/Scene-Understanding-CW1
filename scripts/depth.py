"""Depth-map IO shared by Parts 2-4.

Convention: depth maps are saved as 16-bit PNG in millimetres (TUM convention, depth_factor=5000
means 1 unit = 1/5000 m) so pySLAM can read them as the depth images of an RGB-D sequence.
"""
from __future__ import annotations

import numpy as np
import cv2

DEPTH_FACTOR = 5000.0  # TUM convention; must match the DepthMapFactor in the pySLAM settings file


def save_depth_png(path, depth_m: np.ndarray) -> None:
    d = np.clip(depth_m * DEPTH_FACTOR, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(path), d)


def load_depth_png(path) -> np.ndarray:
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(path)
    return d.astype(np.float32) / DEPTH_FACTOR


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
