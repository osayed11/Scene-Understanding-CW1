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
import os

import cv2
import numpy as np

MAX_U16 = 65535


def max_depth(factor: float) -> float:
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Depth factor must be finite and positive")
    return MAX_U16 / factor


def save_depth(path, depth_m: np.ndarray, factor: float, *, overflow: str = "raise") -> None:
    """Save canonical float32 NPY plus a metric PNG; append extensions to path.

    overflow='invalid' explicitly writes out-of-range pixels as PNG zero while
    retaining their original predictions in NPY. The default rejects overflow.
    """
    path = Path(path)
    depth_m = np.asarray(depth_m, dtype=np.float32)
    if depth_m.ndim != 2:
        raise ValueError("Depth map must have shape (height, width)")
    cap = max_depth(factor)
    if overflow not in ("raise", "invalid"):
        raise ValueError("overflow must be 'raise' or 'invalid'")
    finite = np.isfinite(depth_m)
    if np.any(depth_m[finite] < 0):
        raise ValueError("Negative depth cannot be encoded as an unsigned PNG")
    if overflow == "raise" and finite.any() and depth_m[finite].max() > cap:
        raise ValueError(f"{path}: depth {depth_m[finite].max():.1f} m exceeds the 16-bit cap {max_depth(factor):.1f} m "
                         f"for factor {factor}; use a smaller factor for this sequence")
    encodable = finite & (depth_m > 0) & (depth_m <= cap)
    d = np.where(encodable, depth_m * factor, 0).round().astype(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Append extensions: TUM stems such as '1341847980.722988' contain a decimal
    # point. Path.with_suffix would remove it and overwrite other frames.
    npy, png = Path(str(path) + ".npy"), Path(str(path) + ".png")
    temp_npy, temp_png = Path(str(path) + ".tmp.npy"), Path(str(path) + ".tmp.png")
    try:
        np.save(temp_npy, depth_m, allow_pickle=False)
        if not cv2.imwrite(str(temp_png), d):
            raise OSError(f"Could not write depth image {png}")
        os.replace(temp_png, png)
        os.replace(temp_npy, npy)
    finally:
        temp_npy.unlink(missing_ok=True)
        temp_png.unlink(missing_ok=True)


def load_depth_npy(path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def load_depth_png(path, factor: float) -> np.ndarray:
    """16-bit PNG -> float32 metres; 0 (no depth) -> nan."""
    max_depth(factor)
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(path)
    out = d.astype(np.float32) / factor
    out[d == 0] = np.nan
    return out


def depth_at_keypoints(depth_m: np.ndarray, kps: np.ndarray, radius: int = 1):
    """Local median depth and heuristic uncertainty in metres at image keypoints.

    sigma = max(1 mm, 2% of depth, 1.4826 * local median absolute deviation).
    This is a confidence heuristic, not a calibrated probabilistic uncertainty.
    Invalid/nonpositive depth and out-of-image keypoints produce NaNs. radius=0
    uses the nearest pixel with only the uncertainty floor.
    """
    depth_m = np.asarray(depth_m)
    kps = np.asarray(kps, dtype=float).reshape(-1, 2)
    if depth_m.ndim != 2 or radius < 0 or int(radius) != radius:
        raise ValueError("Expected a 2D depth map and a nonnegative integer patch radius")
    radius = int(radius)
    # The evaluation caches/GT are float32. Batch their small local patches,
    # rather than making two Python-level median calls for every keypoint.
    # Preserve scalar promotion for less common integer/float16 input dtypes.
    if depth_m.dtype not in (np.dtype("float32"), np.dtype("float64")):
        return _depth_at_keypoints_scalar(depth_m, kps, radius)
    h, w = depth_m.shape
    d = np.full(len(kps), np.nan, dtype=np.float32)
    sigma = np.full_like(d, np.nan)
    selected = np.flatnonzero(np.isfinite(kps).all(axis=1)
                              & (kps[:, 0] >= 0) & (kps[:, 0] < w)
                              & (kps[:, 1] >= 0) & (kps[:, 1] < h))
    if not selected.size:
        return d, sigma
    pixels = np.rint(kps[selected]).astype(np.int64)
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < w)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] < h))
    selected, pixels = selected[inside], pixels[inside]
    # Scalar NumPy arithmetic promotes float32 to float64 under NumPy 1.x;
    # NumPy 2.x keeps float32. Match the original scalar formula in either case.
    depth_scale_dtype = np.asarray(.02 * np.float32(1)).dtype
    if radius == 0:
        values = depth_m[pixels[:, 1], pixels[:, 0]]
        valid = np.isfinite(values) & (values > 0)
        selected, values = selected[valid], values[valid]
        d[selected] = values
        sigma[selected] = np.maximum(.001, .02 * d[selected].astype(depth_scale_dtype))
        return d, sigma
    dy = np.arange(-min(radius, h - 1), min(radius, h - 1) + 1)
    dx = np.arange(-min(radius, w - 1), min(radius, w - 1) + 1)
    # Bound temporary patch memory, including callers requesting large radii.
    batch_size = max(1, min(1024, 1_000_000 // (len(dy) * len(dx))))
    for start in range(0, len(selected), batch_size):
        ids, centers = selected[start:start + batch_size], pixels[start:start + batch_size]
        v = centers[:, 1, None, None] + dy[None, :, None]
        u = centers[:, 0, None, None] + dx[None, None, :]
        border = (v >= 0) & (v < h) & (u >= 0) & (u < w)
        patches = depth_m[np.clip(v, 0, h - 1), np.clip(u, 0, w - 1)].reshape(len(ids), -1)
        valid = border.reshape(len(ids), -1) & np.isfinite(patches) & (patches > 0)
        nonempty = valid.any(axis=1)
        if not nonempty.any():
            continue
        ids, patches, valid = ids[nonempty], patches[nonempty], valid[nonempty]
        median = np.nanmedian(np.where(valid, patches, np.nan), axis=1)
        d[ids] = median  # This float32 rounding precedes MAD in the scalar version.
        deviation = np.abs(patches - d[ids, None])
        mad = np.nanmedian(np.where(valid, deviation, np.nan), axis=1)
        mad_scale_dtype = np.asarray(1.4826 * mad.dtype.type(1)).dtype
        sigma[ids] = np.maximum(.001, np.maximum(
            .02 * d[ids].astype(depth_scale_dtype),
            1.4826 * mad.astype(mad_scale_dtype)))
    return d, sigma


def _depth_at_keypoints_scalar(depth_m, kps, radius):
    """Preserve dtype-dependent NumPy scalar arithmetic for uncommon inputs."""
    h, w = depth_m.shape
    d = np.full(len(kps), np.nan, dtype=np.float32)
    sigma = np.full_like(d, np.nan)
    for i, (x, y) in enumerate(kps):
        if not np.isfinite([x, y]).all() or not (0 <= x < w and 0 <= y < h):
            continue
        u, v = int(np.rint(x)), int(np.rint(y))
        if not (0 <= u < w and 0 <= v < h):
            continue
        patch = depth_m[max(0, v-radius):min(h, v+radius+1), max(0, u-radius):min(w, u+radius+1)]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if not valid.size:
            continue
        d[i] = np.median(valid)
        mad = np.median(np.abs(valid - d[i]))
        sigma[i] = max(.001, .02 * d[i], 1.4826 * mad)
    return d, sigma
