import numpy as np, pytest
from depth import save_depth, load_depth_png, load_depth_npy, max_depth


def test_roundtrip_and_cap(tmp_path):
    d = np.array([[0.5, 12.0], [np.nan, 60.0]], dtype=np.float32)
    save_depth(tmp_path / "f", d, factor=256)                 # cap 256 m: fine
    back = load_depth_png(tmp_path / "f.png", 256)
    assert np.allclose(back[np.isfinite(d)], d[np.isfinite(d)], atol=1 / 256) and np.isnan(back[1, 0])
    assert np.array_equal(load_depth_npy(tmp_path / "f.npy"), d, equal_nan=True)
    with pytest.raises(ValueError):                           # cap 13.1 m at factor 5000: 60 m must not be squashed
        save_depth(tmp_path / "g", d, factor=5000)
    assert abs(max_depth(5000) - 13.107) < 1e-3


def test_timestamp_stems_do_not_overwrite_each_other(tmp_path):
    for stem, value in [("1341847980.722988", 2.), ("1341847980.754743", 3.)]:
        save_depth(tmp_path / stem, np.full((2, 2), value), 1000)
        assert np.all(load_depth_npy(tmp_path / (stem + ".npy")) == value)
    assert len(list(tmp_path.glob("*.npy"))) == 2
    assert len(list(tmp_path.glob("*.png"))) == 2


def test_explicit_png_overflow_keeps_canonical_prediction(tmp_path):
    depth = np.array([[2., 100.]], dtype=np.float32)
    save_depth(tmp_path / "f", depth, 5000, overflow="invalid")
    assert np.array_equal(load_depth_npy(tmp_path / "f.npy"), depth)
    png = load_depth_png(tmp_path / "f.png", 5000)
    assert png[0, 0] == 2 and np.isnan(png[0, 1])


def test_invalid_encoding_fails_before_writing(tmp_path):
    for factor in [0, -1, np.nan, np.inf]:
        with pytest.raises(ValueError):
            save_depth(tmp_path / "f", np.ones((2, 2)), factor)
    with pytest.raises(ValueError, match="Negative"):
        save_depth(tmp_path / "f", np.array([[-1.]]), 1000)
    with pytest.raises(ValueError, match="exceeds"):
        save_depth(tmp_path / "f", np.array([[100.]]), 5000)
    assert not list(tmp_path.iterdir())
