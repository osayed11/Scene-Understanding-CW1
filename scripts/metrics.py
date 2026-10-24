"""The one measuring script (Part 1). Every run in the coursework is scored with these functions.

What is computed (state this exactly in the report):
  ATE_noscale : RMSE of translation after SE(3) alignment (scale fixed at 1)
  ATE_scale   : RMSE of translation after Sim(3) alignment (Umeyama with scale)
  scale       : the scale fitted by the Sim(3) alignment
  RPE_trans   : RMSE of relative translation error over a fixed interval `delta` (in frames)
  RPE_rot     : RMSE of relative rotation error (degrees) over the same interval
  tracked     : fraction of the sequence's frames for which the run produced a pose (n_est / n_frames);
                n_assoc of those have a ground-truth pose within max_dt and enter ATE/RPE
  lost_spans  : gaps of more than 3 frame intervals with no pose; the frames before monocular
                initialisation count as such a gap at the start
  lost_spans  : list of (start_t, end_t) gaps where tracking was lost
  time_per_frame : wall-clock seconds of the run / n_frames, taken from the run log
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Tuple

import numpy as np

from trajectory import associate, to_matrices, umeyama


@dataclass
class RunMetrics:
    ate_noscale: float
    ate_scale: float
    scale: float
    rpe_trans: float
    rpe_rot_deg: float
    tracked: float
    n_est: int
    n_assoc: int
    n_gt: int
    n_frames: int
    lost_spans: List[Tuple[float, float]] = field(default_factory=list)
    time_per_frame: float = float("nan")

    def as_row(self) -> dict:
        d = asdict(self)
        d["lost_spans"] = ";".join(f"{a:.3f}-{b:.3f}" for a, b in self.lost_spans)
        return d


def ate(est_xyz: np.ndarray, gt_xyz: np.ndarray, with_scale: bool):
    s, Rm, t = umeyama(est_xyz, gt_xyz, with_scale)
    with np.errstate(all="ignore"):  # spurious OpenBLAS warning, see trajectory.umeyama
        aligned = (s * (Rm @ est_xyz.T)).T + t
    err = np.linalg.norm(aligned - gt_xyz, axis=1)
    return float(np.sqrt(np.mean(err ** 2))), float(s)


def rpe(est_T: np.ndarray, gt_T: np.ndarray, delta: int, scale: float = 1.0,
        t: np.ndarray | None = None, frame_dt: float | None = None):
    """Relative pose error over `delta` frames. est is scaled by `scale` before comparison.
    With timestamps `t` and the nominal frame interval, pairs that span a tracking gap (more than 1.5x the
    nominal delta * frame_dt) are skipped, so the interval really is fixed."""
    est_T = est_T.copy()
    est_T[:, :3, 3] *= scale
    n = len(est_T) - delta
    if n <= 0:
        return float("nan"), float("nan")
    te, re = [], []
    for i in range(n):
        if t is not None and frame_dt and (t[i + delta] - t[i]) > 1.5 * delta * frame_dt:
            continue
        d_est = np.linalg.inv(est_T[i]) @ est_T[i + delta]
        d_gt = np.linalg.inv(gt_T[i]) @ gt_T[i + delta]
        e = np.linalg.inv(d_gt) @ d_est
        te.append(np.linalg.norm(e[:3, 3]))
        c = np.clip((np.trace(e[:3, :3]) - 1) / 2, -1, 1)
        re.append(np.degrees(np.arccos(c)))
    if not te:
        return float("nan"), float("nan")
    return float(np.sqrt(np.mean(np.square(te)))), float(np.sqrt(np.mean(np.square(re))))


def lost_spans(est_t: np.ndarray, gt_t: np.ndarray, n_frames: int, gap_factor: float = 3.0):
    """Gaps in the estimate longer than gap_factor * the frame interval (sequence duration / n_frames)."""
    if len(est_t) < 2:
        return [(float(gt_t[0]), float(gt_t[-1]))]
    dt = (gt_t[-1] - gt_t[0]) / max(n_frames - 1, 1)
    gaps = np.diff(est_t)
    spans = [(float(est_t[i]), float(est_t[i + 1])) for i in np.where(gaps > gap_factor * dt)[0]]
    if est_t[0] - gt_t[0] > gap_factor * dt:
        spans.insert(0, (float(gt_t[0]), float(est_t[0])))
    if gt_t[-1] - est_t[-1] > gap_factor * dt:
        spans.append((float(est_t[-1]), float(gt_t[-1])))
    return spans


def evaluate(est: np.ndarray, gt: np.ndarray, n_frames: int | None = None, rpe_delta: int = 10,
             max_dt: float = 0.02, time_per_frame: float = float("nan")) -> RunMetrics:
    """Score one run. `est`, `gt` are (N, 8) TUM arrays; n_frames is the number of frames of the sequence
    (defaults to the number of GT poses, right only when GT is one pose per frame)."""
    n_frames = n_frames or len(gt)
    e, g = associate(est, gt, max_dt=max_dt)
    if len(e) < 3:
        raise ValueError(f"only {len(e)} associated poses; check timestamps / max_dt")
    ate_ns, _ = ate(e[:, 1:4], g[:, 1:4], with_scale=False)
    ate_s, s = ate(e[:, 1:4], g[:, 1:4], with_scale=True)
    frame_dt = (gt[-1, 0] - gt[0, 0]) / max(n_frames - 1, 1)
    rpe_t, rpe_r = rpe(to_matrices(e), to_matrices(g), rpe_delta, scale=s, t=e[:, 0], frame_dt=frame_dt)
    return RunMetrics(
        ate_noscale=ate_ns, ate_scale=ate_s, scale=s,
        rpe_trans=rpe_t, rpe_rot_deg=rpe_r,
        tracked=len(est) / n_frames, n_est=len(est), n_assoc=len(e), n_gt=len(gt), n_frames=n_frames,
        lost_spans=lost_spans(e[:, 0], gt[:, 0], n_frames),
        time_per_frame=time_per_frame,
    )
