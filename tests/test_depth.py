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
