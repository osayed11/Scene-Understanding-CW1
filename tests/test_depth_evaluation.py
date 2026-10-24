"""Checks of depth evaluation conventions and a small on-disk scoring workflow."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from part2 import eval_depth


def test_raw_scale_error_and_per_frame_median_correction():
    gt = np.array([[1.0, 2.0], [4.0, 8.0]])
    raw = eval_depth.depth_metrics(gt * 2, gt)
    scaled = eval_depth.depth_metrics(gt * 2, gt, median_scale=True)
    assert raw["abs_rel"] == pytest.approx(1)
    assert raw["rmse_m"] == pytest.approx(np.sqrt(np.mean(gt ** 2)))
    assert raw["delta1"] == 0
    assert scaled["scale"] == pytest.approx(0.5)
    assert scaled["abs_rel"] == pytest.approx(0)
    assert scaled["rmse_m"] == pytest.approx(0)
    assert scaled["delta1"] == 1


def test_invalid_pixels_and_depth_range_are_explicit():
    gt = np.array([[0, np.nan, 1], [2, 3, 20]], dtype=float)
    pred = np.array([[100, 100, 1], [0, np.inf, 20]], dtype=float)
    result = eval_depth.depth_metrics(pred, gt, max_depth=10)
    assert result["valid_gt_pixels"] == 3
    assert result["valid_pixels"] == 1
    assert result["prediction_coverage"] == pytest.approx(1 / 3)
    assert result["abs_rel"] == 0
    assert result["delta1"] == 1


def test_empty_mask_does_not_look_like_perfect_prediction():
    result = eval_depth.depth_metrics(np.zeros((2, 2)), np.ones((2, 2)), median_scale=True)
    assert result["valid_gt_pixels"] == 4
    assert result["valid_pixels"] == 0
    assert result["prediction_coverage"] == 0
    assert np.isnan(result["abs_rel"]) and np.isnan(result["scale"])


def test_delta_threshold_is_strict_and_shape_mismatch_is_rejected():
    assert eval_depth.depth_metrics(np.array([[1.25]]), np.array([[1.0]]))["delta1"] == 0
    with pytest.raises(ValueError, match="aligned HxW"):
        eval_depth.depth_metrics(np.ones((1, 2)), np.ones((2, 1)))


@dataclass
class Frame:
    index: int
    timestamp: float
    rgb: Path
    depth_gt: Path | None
    depth_timestamp: float | None

    @property
    def stem(self):
        return self.rgb.stem


def test_sequence_uses_gt_factor_and_macro_average(tmp_path, monkeypatch):
    gt_path = tmp_path / "depth.png"
    assert cv2.imwrite(str(gt_path), np.array([[1000, 2000]], dtype=np.uint16))
    frames = [Frame(0, 0, tmp_path / "a.jpg", gt_path, 0),
              Frame(1, 0.04, tmp_path / "b.jpg", gt_path, 0.04),
              Frame(2, 0.08, tmp_path / "c.jpg", None, None)]
    monkeypatch.setattr(eval_depth, "load_sequence", lambda _: {"depth_gt": "depth", "depth_factor": 5000,
                                                               "gt_depth_factor": 1000})
    monkeypatch.setattr(eval_depth, "load_frame_records", lambda *a, **k: frames)
    predictions = {"a": np.array([[1., 2.]]), "b": np.array([[4., np.nan]])}
    monkeypatch.setattr(eval_depth, "load_prediction", lambda root, seq, frame: predictions[frame.stem])
    summary, detail = eval_depth.evaluate_sequence("replica_room0")
    raw, scaled = summary
    assert raw["frames_total"] == 3
    assert raw["frames_with_gt"] == 2 and raw["frames_without_gt"] == 1
    assert raw["frames_scored"] == 2
    assert raw["valid_pixels"] == 3 and raw["valid_gt_pixels"] == 4
    assert raw["prediction_coverage"] == 0.75
    assert raw["abs_rel"] == pytest.approx(1.5)  # mean(0, 3), not a pixel-weighted mean
    assert raw["rmse_m"] == pytest.approx(1.5)
    assert scaled["abs_rel"] == 0
    assert scaled["scale_mean"] == pytest.approx((1 + 0.25) / 2)
    assert len(detail) == 4


def test_missing_prediction_is_strict_unless_requested(tmp_path, monkeypatch):
    gt_path = tmp_path / "depth.png"
    assert cv2.imwrite(str(gt_path), np.ones((1, 1), dtype=np.uint16) * 1000)
    monkeypatch.setattr(eval_depth, "load_sequence", lambda _: {"depth_gt": "depth", "depth_factor": 1000})
    monkeypatch.setattr(eval_depth, "load_frame_records", lambda *a, **k: [Frame(0, 0, tmp_path / "a.jpg", gt_path, 0)])

    def missing(*args):
        raise FileNotFoundError("a.npy")

    monkeypatch.setattr(eval_depth, "load_prediction", missing)
    with pytest.raises(FileNotFoundError):
        eval_depth.evaluate_sequence("tum_fr3")
    summary, detail = eval_depth.evaluate_sequence("tum_fr3", skip_missing=True)
    assert summary[0]["status"] == "no_predictions"
    assert summary[0]["frames_missing_predictions"] == 1
    assert np.isnan(summary[0]["abs_rel"])
    assert summary[0]["prediction_coverage"] == 0
    assert detail[0]["status"] == "missing_prediction"


def test_kitti_no_gt_is_explicit_and_never_loads_predictions(monkeypatch):
    monkeypatch.setattr(eval_depth, "load_sequence", lambda _: {"depth_gt": None})
    monkeypatch.setattr(eval_depth, "load_frame_records", lambda *a, **k: [Frame(0, 0, Path("a.png"), None, None)])

    def unexpected(*args):
        pytest.fail("KITTI has no GT; predictions should not be loaded")

    monkeypatch.setattr(eval_depth, "load_prediction", unexpected)
    summary, detail = eval_depth.evaluate_sequence("kitti06")
    assert all(row["status"] == "no_ground_truth" for row in summary)
    assert all(row["frames_without_gt"] == 1 for row in summary)
    assert all(np.isnan(row["abs_rel"]) for row in summary)
    assert detail == []
