"""Dataset access for offline Part 2; independent of the pySLAM native build.

Keep every RGB frame. TUM ground-truth depth is a nearest-timestamp match within
20 ms (matches may reuse a depth image); unavailable depth stays None.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import yaml

from config import ROOT, PYSLAM, load_sequence

SEQUENCES = ("kitti06", "tum_fr3", "replica_room0")


@dataclass(frozen=True)
class FrameRecord:
    index: int
    timestamp: float
    rgb: Path
    depth_gt: Path | None = None
    depth_timestamp: float | None = None

    @property
    def stem(self) -> str:
        return self.rgb.stem


@dataclass(frozen=True)
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.]])


def find_data_root(data_root=None) -> Path:
    explicit = data_root or os.environ.get("CW1_DATA_ROOT")
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Data root does not exist: {root}")
        return root
    for root in (ROOT / "data", ROOT.parent / "data"):
        if any((root / name).is_dir() for name in ("kitti06", "tum", "replica")):
            return root
    raise FileNotFoundError("No datasets found. Pass --data-root /path/to/data or set CW1_DATA_ROOT.")


def data_path(value, data_root=None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "data":
        return find_data_root(data_root).joinpath(*path.parts[1:])
    return ROOT / path


def _timestamp_list(path: Path) -> list[tuple[float, Path]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            values = line.split()
            rows.append((float(values[0]), path.parent / values[1]))
    rows.sort(key=lambda row: row[0])
    if not rows or len({t for t, _ in rows}) != len(rows):
        raise ValueError(f"Empty or duplicate timestamps in {path}")
    return rows


def load_frame_records(sequence_name: str, data_root=None) -> list[FrameRecord]:
    cfg = load_sequence(sequence_name)
    records = []
    if sequence_name.startswith("kitti"):
        images = sorted(data_path(cfg["images"], data_root).glob("*.png"))
        times = np.loadtxt(data_path(cfg["timestamps"], data_root), ndmin=1)
        if len(images) != len(times):
            raise ValueError(f"{sequence_name}: {len(images)} images but {len(times)} timestamps")
        records = [FrameRecord(i, float(t), image) for i, (t, image) in enumerate(zip(times, images))]
    elif sequence_name.startswith("replica"):
        folder = data_path(cfg["images"], data_root)
        camera_fps = 25.0  # supplied REPLICA.yaml / ReplicaDataset convention
        for i, image in enumerate(sorted(folder.glob("frame*.jpg"))):
            match = re.fullmatch(r"frame(\d+)", image.stem)
            if match is None or int(match[1]) != i:
                raise ValueError(f"{sequence_name}: non-contiguous Replica frame numbering at {image.name}")
            depth = folder / image.name.replace("frame", "depth").replace(".jpg", ".png")
            records.append(FrameRecord(i, i / camera_fps, image, depth if depth.exists() else None,
                                       i / camera_fps if depth.exists() else None))
    else:
        folder = data_path(cfg.get("root", str(Path(cfg["images"]).parent)), data_root)
        rgb = _timestamp_list(folder / "rgb.txt")
        depth_list = folder / "depth.txt"
        depths = _timestamp_list(depth_list) if depth_list.exists() else []
        depth_times = np.array([row[0] for row in depths])
        for i, (timestamp, image) in enumerate(rgb):
            match = None
            if len(depths):
                right = int(np.searchsorted(depth_times, timestamp))
                candidates = [j for j in (right - 1, right) if 0 <= j < len(depths)]
                j = min(candidates, key=lambda k: abs(depth_times[k] - timestamp))
                if abs(depth_times[j] - timestamp) <= .02:
                    match = depths[j]
            records.append(FrameRecord(i, timestamp, image, match[1] if match else None,
                                       match[0] if match else None))
    if not records:
        raise FileNotFoundError(f"{sequence_name}: no RGB frames found under {find_data_root(data_root)}")
    missing = [frame.rgb for frame in records if not frame.rgb.is_file()]
    if missing:
        raise FileNotFoundError(f"{sequence_name}: missing RGB image {missing[0]}")
    return records


def load_prediction(pred_root, sequence_name: str, frame: FrameRecord) -> np.ndarray:
    path = Path(pred_root) / sequence_name / f"{frame.stem}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction {path}; run predict_depth.py first")
    depth = np.load(path, allow_pickle=False)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D metric depth map, got {depth.shape} in {path}")
    return depth.astype(np.float32, copy=False)


def load_depth_gt(sequence_name: str, frame: FrameRecord, data_root=None) -> np.ndarray | None:
    if frame.depth_gt is None:
        return None
    cfg = load_sequence(sequence_name)
    raw = cv2.imread(str(frame.depth_gt), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(frame.depth_gt)
    if raw.ndim != 2:
        raise ValueError(f"Expected single-channel ground-truth depth: {frame.depth_gt}")
    depth = raw.astype(np.float32) / float(cfg.get("gt_depth_factor", cfg["depth_factor"]))
    depth[raw == 0] = np.nan
    return depth


def load_camera(sequence_name: str, data_root=None) -> Camera:
    relative = Path(load_sequence(sequence_name)["settings_file"])
    candidates = [relative] if relative.is_absolute() else [
        Path(os.environ.get("CW1_PYSLAM_ROOT", str(PYSLAM))) / relative,
        ROOT.parent / "COMP0227_26-27" / "pyslam" / relative,
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Camera settings {relative} unavailable; set CW1_PYSLAM_ROOT")
    content = re.sub(r"\A\s*%YAML[^\n]*\n", "", path.read_text())
    content = re.sub(r"(?m)^([A-Za-z_][\w.]*):(?=[^\s:])", r"\1: ", content)
    cfg = yaml.safe_load(content)
    return Camera(*(float(cfg[f"Camera.{key}"]) for key in ("fx", "fy", "cx", "cy")),
                  int(cfg["Camera.width"]), int(cfg["Camera.height"]))


def load_frame_poses(sequence_name: str, frames: list[FrameRecord], data_root=None,
                     max_dt: float = .02) -> np.ndarray:
    """Evaluation-only C2W poses, interpolated within short GT intervals.

    Ground truth is never supplied to depth inference or online map-scale estimation.
    Unavailable poses are NaN, not an extrapolated reference.
    """
    cfg = load_sequence(sequence_name)
    path = data_path(cfg["groundtruth"], data_root)
    if sequence_name.startswith("replica") and not path.exists():
        matrices = np.loadtxt(data_path(cfg["root"], data_root) / "traj.txt").reshape(-1, 4, 4)
        gt_times = np.arange(len(matrices)) / 25.0
        rotations = Rotation.from_matrix(matrices[:, :3, :3])
        positions = matrices[:, :3, 3]
    else:
        rows = np.loadtxt(path, ndmin=2)[:, :8]
        rows = rows[np.argsort(rows[:, 0])]
        gt_times = rows[:, 0]
        rotations = Rotation.from_quat(rows[:, 4:8])
        positions = rows[:, 1:4]
    if len(gt_times) < 2 or np.any(np.diff(gt_times) <= 0):
        raise ValueError(f"{sequence_name}: need at least two distinct ordered GT timestamps")
    requested = np.array([f.timestamp for f in frames])
    out = np.full((len(frames), 4, 4), np.nan)
    nearest = np.clip(np.searchsorted(gt_times, requested), 1, len(gt_times) - 1)
    left, right = nearest - 1, nearest
    valid = ((requested >= gt_times[0]) & (requested <= gt_times[-1]) &
             (np.minimum(abs(gt_times[left] - requested), abs(gt_times[right] - requested)) <= max_dt))
    # Do not interpolate over long GT outages; exact recorded timestamps remain usable.
    exact = (abs(gt_times[left] - requested) < 1e-6) | (abs(gt_times[right] - requested) < 1e-6)
    valid &= exact | ((gt_times[right] - gt_times[left]) <= 2 * max_dt)
    if valid.any():
        times = requested[valid]
        out[valid] = np.eye(4)
        out[valid, :3, :3] = Slerp(gt_times, rotations)(times).as_matrix()
        for axis in range(3):
            out[valid, axis, 3] = np.interp(times, gt_times, positions[:, axis])
    return out
