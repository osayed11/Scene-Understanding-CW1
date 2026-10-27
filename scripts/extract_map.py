"""Compact a pySLAM saved map (map.json from our --save-map patch, ~300 MB) into map_points.npz.

Arrays (one row per observation of a good map point by a good keyframe):
  obs_point      index into the point arrays        obs_kf      index into the keyframe arrays
  obs_kp_idx     keypoint index in the keyframe     obs_u, obs_v undistorted keypoint (kpsu)
  obs_depth_map  depth of the point in that keyframe's camera (map units; metres only for RGB-D runs)
Per point: point_id, point_xyz (map units), point_parallax_deg (largest angle between the rays of its observations).
Per keyframe: kf_id, kf_img_id (frame index in the sequence), kf_timestamp, kf_Tcw (4x4).
The pose convention is verified by reprojecting every point into its keyframes (median error must be < 20 px).
The heavy files (map.json, loop_closing.db) are deleted afterwards unless --keep.

usage: python scripts/extract_map.py results/logs/<run_id>/map [--keep]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _obj(x):
    """pySLAM nests JSON as strings at some levels."""
    return json.loads(x) if isinstance(x, str) else x


def extract(map_dir: Path, keep: bool = False) -> Path:
    m = _obj(json.load(open(map_dir / "map.json")))
    mp = _obj(m["map"])
    kfs = [kf for kf in map(_obj, mp["keyframes"]) if not kf.get("_is_bad")]
    kf_index = {kf["id"]: i for i, kf in enumerate(kfs)}
    kf_Tcw = np.array([kf["pose"] for kf in kfs], dtype=np.float64)
    kf_kpsu = [np.asarray(kf["kpsu"], dtype=np.float64) for kf in kfs]
    cam = _obj(kfs[0]["camera"])
    fx, fy, cx, cy = (float(cam[k]) for k in ("fx", "fy", "cx", "cy"))

    pts = [p for p in map(_obj, mp["points"]) if not p.get("_is_bad")]
    point_id = np.array([p["id"] for p in pts], dtype=np.int64)
    point_xyz = np.array([p["pt"] for p in pts], dtype=np.float64)
    obs_point, obs_kf, obs_kp = [], [], []
    for i, p in enumerate(pts):
        for kf_id, idx in p["_observations"]:
            if kf_id in kf_index:
                obs_point.append(i); obs_kf.append(kf_index[kf_id]); obs_kp.append(idx)
    obs_point, obs_kf, obs_kp = (np.array(a, dtype=np.int64) for a in (obs_point, obs_kf, obs_kp))

    # depth of each point in its observing keyframe, and reprojection check of the pose convention (Tcw)
    T = kf_Tcw[obs_kf]
    X = point_xyz[obs_point]
    Xc = np.einsum("nij,nj->ni", T[:, :3, :3], X) + T[:, :3, 3]
    obs_depth = Xc[:, 2]
    uv = np.stack([fx * Xc[:, 0] / Xc[:, 2] + cx, fy * Xc[:, 1] / Xc[:, 2] + cy], 1)
    kp_uv = np.array([kf_kpsu[k][j] for k, j in zip(obs_kf, obs_kp)])
    reproj = np.linalg.norm(uv - kp_uv, axis=1)
    med = float(np.median(reproj))
    if med > 20.0:  # a wrong convention gives hundreds of px; KITTI mono maps sit at ~2.5 px, TUM ~1 px
        raise RuntimeError(f"{map_dir}: median reprojection error {med:.1f} px: pose convention wrong?")

    # parallax per point: largest angle between the rays from the observing camera centres
    Ow = np.einsum("kij,kj->ki", -np.transpose(kf_Tcw[:, :3, :3], (0, 2, 1)), kf_Tcw[:, :3, 3])  # -R^T t
    parallax = np.zeros(len(pts))
    order = np.argsort(obs_point, kind="stable")
    starts = np.searchsorted(obs_point[order], np.arange(len(pts) + 1))
    for i in range(len(pts)):
        ks = obs_kf[order[starts[i]:starts[i + 1]]]
        if len(ks) < 2:
            continue
        rays = point_xyz[i] - Ow[ks]
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        cosang = np.clip(rays @ rays.T, -1, 1)
        parallax[i] = np.degrees(np.arccos(cosang.min()))

    out = map_dir / "map_points.npz"
    np.savez_compressed(
        out, sensor_type=str(m.get("sensor_type")), use_cpp_core=bool(m.get("USE_CPP_CORE")),
        fx=fx, fy=fy, cx=cx, cy=cy,
        kf_id=np.array([kf["id"] for kf in kfs]), kf_img_id=np.array([kf["img_id"] for kf in kfs]),
        kf_timestamp=np.array([kf["timestamp"] for kf in kfs], dtype=np.float64), kf_Tcw=kf_Tcw,
        point_id=point_id, point_xyz=point_xyz, point_parallax_deg=parallax,
        obs_point=obs_point, obs_kf=obs_kf, obs_kp_idx=obs_kp, obs_u=kp_uv[:, 0], obs_v=kp_uv[:, 1],
        obs_depth_map=obs_depth, obs_reproj_px=reproj,
    )
    if not keep:
        for f in ("map.json", "loop_closing.db", "kf_image_map.pkl"):
            (map_dir / f).unlink(missing_ok=True)
    print(f"{out}: {len(kfs)} keyframes, {len(pts)} points, {len(obs_point)} observations, "
          f"median reprojection {med:.2f} px, median parallax {np.median(parallax[parallax > 0]):.2f} deg")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("map_dir", type=Path)
    ap.add_argument("--keep", action="store_true", help="keep map.json and loop_closing.db")
    a = ap.parse_args()
    extract(a.map_dir, a.keep)
