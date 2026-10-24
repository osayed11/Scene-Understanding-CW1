"""Temporal depth tests use known geometry, rather than matching implementation formulas."""
import numpy as np

from part2.temporal_consistency import compare_depth_pair


def camera(width=9, height=7, focal=5.0):
    return np.array([[focal, 0, (width - 1) / 2],
                     [0, focal, (height - 1) / 2], [0, 0, 1.0]])


def test_translated_plane_requires_motion_compensation():
    source = np.full((7, 9), 4.0)
    target = np.full((7, 9), 3.0)
    source_pose = np.eye(4)
    target_pose = np.eye(4)
    target_pose[2, 3] = 1.0  # Camera moved one metre towards the plane.
    result = compare_depth_pair(source, target, source_pose, target_pose, camera(),
                                target_gt_depth=target, pixel_stride=1)
    assert result["n_compared"] > 0
    assert result["n_projected"] < source.size  # The closer plane extends out of view.
    assert result["median_abs_m"] < 1e-12
    assert result["p90_rel"] < 1e-12
    assert np.median(np.abs(source - target)) == 1.0


def test_target_sensor_depth_removes_new_foreground_occlusion():
    source = np.full((7, 9), 4.0)
    target = source.copy()
    target[:, :3] = 2.0  # Newly visible foreground hides the source background.
    result = compare_depth_pair(source, target, np.eye(4), np.eye(4), camera(),
                                target_gt_depth=target, pixel_stride=1)
    assert result["visibility_mode"] == "target_gt_and_source_zbuffer"
    assert result["n_visibility_occluded"] == 21
    assert result["n_compared"] == 42
    assert result["median_abs_m"] == 0.0


def test_invalid_sensor_depth_is_not_treated_as_visible():
    source = np.full((7, 9), 4.0)
    gt = source.copy()
    gt[0, :3] = [0, np.nan, np.inf]
    target = source.copy()
    target[1, 1] = np.nan
    source[2, 2] = 0
    result = compare_depth_pair(source, target, np.eye(4), np.eye(4), camera(),
                                target_gt_depth=gt, pixel_stride=1)
    assert result["n_sampled"] == 63
    assert result["n_source_valid"] == 62
    assert result["n_visibility_invalid"] == 3
    assert result["n_target_invalid"] == 1
    assert result["n_compared"] == 58


def test_without_sensor_depth_visibility_is_explicitly_prediction_dependent():
    source = np.full((7, 9), 4.0)
    target = source.copy()
    target[:, :3] = 2.0
    result = compare_depth_pair(source, target, np.eye(4), np.eye(4), camera(), pixel_stride=1)
    assert result["visibility_mode"] == "target_prediction_and_source_zbuffer"
    assert result["n_visibility_occluded"] == 21
    assert result["n_compared"] == 42


def test_empty_valid_depths_return_counts_and_undefined_errors():
    invalid = np.zeros((7, 9))
    result = compare_depth_pair(invalid, invalid, np.eye(4), np.eye(4), camera(), pixel_stride=1)
    assert result["n_sampled"] == 63
    assert result["n_source_valid"] == result["n_compared"] == 0
    assert np.isnan(result["median_abs_m"])


def test_all_points_behind_camera_produce_no_comparisons():
    source = np.full((7, 9), 4.0)
    target_pose = np.eye(4)
    target_pose[2, 3] = 5.0
    result = compare_depth_pair(source, source, np.eye(4), target_pose, camera(), pixel_stride=1)
    assert result["n_projected"] == result["n_compared"] == 0


def test_lateral_motion_zbuffer_keeps_nearer_of_colliding_points():
    source = np.zeros((7, 9))
    # With focal=4, these world points are (1,0,2) and (1,0,4).
    source[3, 6] = 2.0
    source[3, 5] = 4.0
    target = np.zeros_like(source)
    target[3, 4] = 2.0
    target_pose = np.eye(4)
    target_pose[0, 3] = 1.0  # Both points project to the target optical axis.
    result = compare_depth_pair(source, target, np.eye(4), target_pose, camera(focal=4),
                                target_gt_depth=target, pixel_stride=1)
    assert result["n_source_valid"] == result["n_projected"] == 2
    assert result["n_zbuffer_occluded"] == 1
    assert result["n_visibility_occluded"] == 0
    assert result["n_compared"] == 1
    assert result["median_abs_m"] == 0.0


def test_biased_but_consistent_predictions_remain_visible_with_gt_geometry():
    truth = np.ones((7, 9))
    prediction = 2 * truth
    result = compare_depth_pair(prediction, prediction, np.eye(4), np.eye(4), camera(),
                                source_gt_depth=truth, target_gt_depth=truth, pixel_stride=1)
    assert result["source_geometry_mode"] == "source_gt"
    assert result["n_source_geometry_valid"] == result["n_compared"] == 63
    assert result["n_visibility_occluded"] == 0
    assert result["median_abs_m"] == 0.0  # Temporal consistency is distinct from accuracy.


def test_gt_geometry_retains_large_temporal_errors_instead_of_masking_them():
    truth = np.ones((7, 9))
    result = compare_depth_pair(3 * truth, 2 * truth, np.eye(4), np.eye(4), camera(),
                                source_gt_depth=truth, target_gt_depth=truth, pixel_stride=1)
    assert result["n_compared"] == 63
    assert result["n_visibility_occluded"] == 0
    assert result["median_abs_m"] == 1.0
    assert np.isclose(result["median_rel"], 1 / 3)


def test_invalid_source_gt_is_excluded_without_prediction_fallback():
    truth = np.ones((7, 9))
    source_gt = truth.copy()
    source_gt[0, :3] = [0, np.nan, np.inf]
    result = compare_depth_pair(2 * truth, 2 * truth, np.eye(4), np.eye(4), camera(),
                                source_gt_depth=source_gt, target_gt_depth=truth, pixel_stride=1)
    assert result["n_source_valid"] == 63
    assert result["n_source_geometry_invalid"] == 3
    assert result["n_source_geometry_valid"] == result["n_compared"] == 60


def test_invalid_predicted_foreground_still_occludes_with_source_gt():
    source_gt = np.zeros((7, 9))
    source_gt[3, 6], source_gt[3, 5] = 2.0, 4.0
    source_prediction = source_gt.copy()
    source_prediction[3, 6] = np.nan
    target = np.zeros_like(source_gt)
    target[3, 4] = 2.0
    target_pose = np.eye(4)
    target_pose[0, 3] = 1.0
    result = compare_depth_pair(source_prediction, target, np.eye(4), target_pose, camera(focal=4),
                                source_gt_depth=source_gt, target_gt_depth=target, pixel_stride=1)
    assert result["n_source_geometry_valid"] == 2
    assert result["n_source_valid"] == 1
    assert result["n_zbuffer_occluded"] == 1
    assert result["n_source_prediction_invalid"] == 1
    assert result["n_compared"] == 0
