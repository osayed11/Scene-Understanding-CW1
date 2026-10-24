import numpy as np
import pytest

from part2.keypoint_depth import estimate_map_scale


def test_map_scale_robust_to_outliers_and_has_correct_direction():
    learned = np.linspace(1.0, 5.0, 12)
    ratios = np.array([2.0, 2.01, 1.99, 2.02, 1.98, 2.0, 2.01, 1.99, 2.0, 2.0, 100.0, 0.02])
    result = estimate_map_scale(learned, learned * ratios, parallax_deg=np.full(12, 3.0))
    assert result["map_scale_status"] == "estimated_from_paired_map_depths"
    assert result["scale_map_units_per_metre"] == pytest.approx(2.0)
    assert result["scale_metres_per_map_unit"] == pytest.approx(0.5)
    assert result["n_scale_support"] == 10


def test_map_scale_zero_mad_rejects_large_outlier():
    result = estimate_map_scale(np.ones(6), np.array([3, 3, 3, 3, 3, 300.0]))
    assert result["scale_map_units_per_metre"] == 3.0
    assert result["n_scale_support"] == 5


def test_map_scale_rejects_weak_parallax_and_uncertain_depth():
    result = estimate_map_scale(
        np.full(8, 2.0), np.array([4, 4, 4, 4, 4, 200, 200, 200.0]),
        sigma_m=np.array([.1, .1, .1, .1, .1, .1, .1, 1.0]),
        parallax_deg=np.array([2, 2, 2, 2, 2, .2, .8, 2.0]),
    )
    assert result["n_scale_candidates"] == result["n_scale_support"] == 5
    assert result["scale_map_units_per_metre"] == 2.0
    assert result["scale_parallax_gate_applied"]
    assert result["scale_uncertainty_gate_applied"]


def test_map_scale_invalids_and_insufficient_support_are_unavailable():
    result = estimate_map_scale(np.array([0, -1, np.nan, 1, 1, np.inf]), np.array([1, 1, 1, 0, 2, 2.0]))
    assert result["map_scale_status"] == "unavailable_insufficient_support"
    assert result["n_scale_candidates"] == 1
    assert np.isnan(result["scale_map_units_per_metre"])
    weak = estimate_map_scale(np.ones(8), np.full(8, 2.0), parallax_deg=np.full(8, .1))
    assert weak["n_scale_support"] == 0
    assert np.isnan(weak["scale_map_units_per_metre"])


def test_map_scale_requires_matching_correspondences():
    with pytest.raises(ValueError, match="same shape"):
        estimate_map_scale(np.ones(5), np.ones(4))
    with pytest.raises(ValueError, match="sigma_m"):
        estimate_map_scale(np.ones(5), np.ones(5), sigma_m=np.ones(4))
    with pytest.raises(ValueError, match="parallax_deg"):
        estimate_map_scale(np.ones(5), np.ones(5), parallax_deg=np.ones(4))


@pytest.mark.parametrize("has_gt", [True, False])
def test_keypoint_cli_exports_diagnostic_samples_and_explicit_scale_status(tmp_path, monkeypatch, has_gt):
    import csv
    import cv2
    from part2 import common
    from part2.keypoint_depth import main

    rgb = tmp_path / "1341847981.726650.png"
    image = np.random.default_rng(4).integers(0, 256, (160, 160), dtype=np.uint8)
    assert cv2.imwrite(str(rgb), image)
    frame = common.FrameRecord(index=7, timestamp=1341847981.726650, rgb=rgb)
    monkeypatch.setattr(common, "load_frame_records", lambda *args: [frame])
    monkeypatch.setattr(common, "load_prediction", lambda *args: np.full(image.shape, 3.0, dtype=np.float32))
    monkeypatch.setattr(common, "load_depth_gt", lambda *args: np.full(image.shape, 2.0, dtype=np.float32) if has_gt else None)
    table = tmp_path / "summary.csv"
    output = tmp_path / "keypoints"
    main(["--sequence", "tum_fr3", "--max-frames", "1", "--nfeatures", "50",
          "--output", str(table), "--keypoints-root", str(output)])
    with table.open() as f:
        row = next(csv.DictReader(f))
    assert row["keypoint_source"] == "opencv_ORB_diagnostic"
    assert row["matches_saved_slam_keypoints"] == "False"
    assert row["map_scale_status"] == "unavailable_no_map_samples"
    with np.load(output / "tum_fr3" / f"{frame.stem}.npz", allow_pickle=False) as samples:
        assert len(samples["uv"]) > 0
        assert samples["uv"].shape[1] == 2
        assert samples["depth_m"].shape == samples["sigma_m"].shape == (len(samples["uv"]),)
        assert np.all(samples["depth_m"] == 3.0)
    if has_gt:
        assert float(row["keypoint_rmse_m"]) == pytest.approx(1.0)
        assert float(row["keypoint_absrel"]) == pytest.approx(0.5)
    else:
        assert row["gt_status"] == "unavailable_or_no_valid_pairs"
        assert np.isnan(float(row["keypoint_rmse_m"]))
