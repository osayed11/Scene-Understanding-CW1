"""No model weights or Torch runtime are loaded by these adapter regression tests."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from part2.model_adapters import DAV2Predictor


def test_v2_cpu_request_never_auto_selects_available_cuda():
    events = []

    class Tensor:
        def __init__(self, array):
            self.array = np.asarray(array)

        def unsqueeze(self, axis):
            return Tensor(np.expand_dims(self.array, axis))

        def to(self, *, device, dtype):
            events.append((device, dtype))
            assert device == "cpu"
            return self

        def __getitem__(self, key):
            return Tensor(self.array[key])

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    class Network:
        def image2tensor(self, *args):
            raise AssertionError("official auto-device method must never be called")

        def __call__(self, inputs):
            # Test the supplied BGR is converted before the tensor transfer.
            assert inputs.array.shape == (1, 3, 2, 4)
            assert np.allclose(inputs.array[0, :, 0, 0], np.array([30, 20, 10]) / 255)
            return Tensor(np.full((1, 2, 4), 2.0, dtype=np.float32))

    def interpolate(depth, native_size, *, mode, align_corners):
        assert depth.array.shape == (1, 1, 2, 4)
        assert native_size == (2, 4) and mode == "bilinear" and align_corners is True
        return depth

    predictor = DAV2Predictor.__new__(DAV2Predictor)
    predictor.device, predictor.dtype = "cpu", "float32"
    predictor.model = Network()
    predictor.transforms = (lambda sample: {"image": np.transpose(sample["image"], (2, 0, 1))},)
    predictor.torch = SimpleNamespace(
        inference_mode=nullcontext, from_numpy=Tensor,
        cuda=SimpleNamespace(is_available=lambda: True),
        nn=SimpleNamespace(functional=SimpleNamespace(interpolate=interpolate)),
    )
    result = predictor.predict(np.full((2, 4, 3), [10, 20, 30], dtype=np.uint8), None)
    assert events == [("cpu", "float32")]
    assert result.shape == (2, 4) and np.all(result == 2.0)


def test_official_v2_transforms_preserve_normalization_rounding_and_layout():
    # Optional source-integration check: load only the pure NumPy/OpenCV
    # transforms, not dpt.py, a Torch model, or a checkpoint.
    path = (Path(__file__).resolve().parents[1] / "weights/Depth-Anything-V2/metric_depth"
            / "depth_anything_v2/util/transform.py")
    if not path.is_file():
        pytest.skip("official V2 source checkout not present")
    spec = importlib.util.spec_from_file_location("cw1_test_v2_transforms", path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    resize = official.Resize(width=28, height=28, resize_target=False,
                             keep_aspect_ratio=True, ensure_multiple_of=14,
                             resize_method="lower_bound")
    normalize = official.NormalizeImage(mean=[.485, .456, .406], std=[.229, .224, .225])
    prepare = official.PrepareForNet()
    rgb = np.full((17, 39, 3), [30, 20, 10], dtype=np.float64) / 255
    result = prepare(normalize(resize({"image": rgb})))["image"]
    assert result.shape == (3, 28, 70)  # short side lower bound and nearest multiple of 14
    assert result.dtype == np.float32 and result.flags.c_contiguous
    expected = (np.array([30, 20, 10]) / 255 - [.485, .456, .406]) / [.229, .224, .225]
    assert np.allclose(result[:, 0, 0], expected)
