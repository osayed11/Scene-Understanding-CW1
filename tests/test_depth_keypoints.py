"""Check batched keypoint sampling against the original scalar definition."""
import numpy as np
import pytest

from depth import depth_at_keypoints


def scalar_reference(depth, coordinates, radius):
    """Independent per-point oracle, including the intermediate float32 depth."""
    depth = np.asarray(depth)
    coordinates = np.asarray(coordinates, dtype=float).reshape(-1, 2)
    result = np.full(len(coordinates), np.nan, np.float32)
    uncertainty = np.full_like(result, np.nan)
    height, width = depth.shape
    for index, (x, y) in enumerate(coordinates):
        if not np.isfinite([x, y]).all() or not (0 <= x < width and 0 <= y < height):
            continue
        u, v = int(np.rint(x)), int(np.rint(y))
        if not (0 <= u < width and 0 <= v < height):
            continue
        patch = depth[max(0, v - radius):min(height, v + radius + 1),
                      max(0, u - radius):min(width, u + radius + 1)]
        values = patch[np.isfinite(patch) & (patch > 0)]
        if values.size:
            result[index] = np.median(values)
            mad = np.median(np.abs(values - result[index]))
            uncertainty[index] = max(.001, .02 * result[index], 1.4826 * mad)
    return result, uncertainty


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16, np.int16, np.int64])
@pytest.mark.parametrize("radius", [0, 1, 3, 30])
def test_matches_scalar_on_random_invalids_borders_and_rounding(dtype, radius):
    rng = np.random.default_rng(6027)
    for height, width in [(1, 1), (2, 3), (17, 23)]:
        depth = rng.uniform(-5, 80, (height, width)).astype(dtype)
        if np.issubdtype(dtype, np.floating):
            depth.flat[::7] = np.nan
            depth.flat[1::17] = np.inf
            depth.flat[2::19] = -np.inf
        depth.flat[3::11] = 0
        coordinates = rng.uniform([-.7, -.7], [width + .7, height + .7], (135, 2))
        special = np.array([[0, 0], [width - 1, height - 1], [width, height],
                            [width - .5, height - .5], [width - 1e-9, height - 1e-9],
                            [.5, .5], [np.nextafter(.5, 0), np.nextafter(.5, 1)],
                            [-1e-10, 0], [np.nan, 0], [0, np.inf], [-np.inf, 0]])
        coordinates = np.vstack((coordinates, special))
        expected = scalar_reference(depth, coordinates, radius)
        actual = depth_at_keypoints(depth, coordinates, radius)
        for observed, reference in zip(actual, expected):
            assert observed.dtype == np.float32
            np.testing.assert_array_equal(observed, reference, strict=True)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batch_boundaries_and_float32_intermediate_match_exactly(dtype):
    rng = np.random.default_rng(27)
    depth = rng.lognormal(2, 1, (31, 43)).astype(dtype)
    depth.flat[::5] = np.nan
    coordinates = rng.uniform([0, 0], [42.4, 30.4], (2501, 2))
    actual = depth_at_keypoints(depth, coordinates, radius=3)
    expected = scalar_reference(depth, coordinates, radius=3)
    for observed, reference in zip(actual, expected):
        np.testing.assert_array_equal(observed, reference, strict=True)


def test_empty_inputs_and_all_invalid_patches_are_quiet():
    for shape in [(0, 0), (0, 5), (5, 0)]:
        depth, sigma = depth_at_keypoints(np.empty(shape, np.float32), [[0, 0]], radius=3)
        assert np.isnan(depth).all() and np.isnan(sigma).all()
    with np.errstate(all="raise"):
        depth, sigma = depth_at_keypoints(np.full((4, 5), np.nan, np.float32), [[0, 0], [2, 3]], radius=3)
        assert np.isnan(depth).all() and np.isnan(sigma).all()
        depth, sigma = depth_at_keypoints(np.ones((4, 5), np.float32), [], radius=3)
        assert depth.shape == sigma.shape == (0,)


def test_bad_input_validation():
    with pytest.raises(ValueError):
        depth_at_keypoints(np.ones((3, 3, 1)), [[1, 1]])
    for radius in [-1, .5]:
        with pytest.raises(ValueError):
            depth_at_keypoints(np.ones((3, 3)), [[1, 1]], radius)
