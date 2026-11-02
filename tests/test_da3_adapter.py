import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from part2.common import Camera
from part2.da3_adapter import DA3Predictor, canonical_to_metric_depth


def test_metric_conversion_uses_both_processed_focals_and_native_shape():
    canonical = np.array([[1, 2], [3, 4]], dtype=np.float32)
    # Different focal scales can arise from independent dimension rounding.
    K = np.array([[150, 0, 0], [0, 450, 0], [0, 0, 1.0]])
    metric = canonical_to_metric_depth(canonical, K, (4, 4))
    assert metric.dtype == np.float32 and metric.shape == (4, 4)
    assert metric[0, 0] == 1.0 and metric[-1, -1] == 4.0
    assert metric[1, 1] == pytest.approx(1.75)


def test_crop_changes_principal_point_not_the_metric_focal_scale():
    canonical = np.full((14, 14), 4.0)
    cropped_K = np.array([[600, 0, 2], [0, 600, 3], [0, 0, 1.0]])
    metric = canonical_to_metric_depth(canonical, cropped_K, canonical.shape)
    assert np.all(metric == 8.0)


@pytest.mark.parametrize("K", [np.eye(2), np.diag([-1, 1, 1]), np.full((3, 3), np.nan)])
def test_invalid_intrinsics_are_rejected(K):
    with pytest.raises(ValueError, match="intrinsics"):
        canonical_to_metric_depth(np.ones((14, 14)), K, (14, 14))


def _fake_official_model(tmp_path, monkeypatch, *, crop=False):
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_name": "da3metric-large", "config": {"mock": True}}))
    (checkpoint / "model.safetensors").write_bytes(b"mocked-network-no-weights")
    source = tmp_path / "source"
    (source / "src/depth_anything_3").mkdir(parents=True)
    (source / "src/depth_anything_3/api.py").write_text("# mocked official API")
    calls = {}

    class Network(torch.nn.Module):
        def load_state_dict(self, state, strict=True):
            calls["weights"] = state
            calls["strict"] = strict

        def forward(self, images, **kwargs):
            calls["network_shape"] = tuple(images.shape)
            calls["network_dtype"] = images.dtype
            calls["cpu_autocast"] = torch.is_autocast_enabled("cpu")
            calls["grad_enabled"] = torch.is_grad_enabled()
            return {"depth": torch.full((1, 1, 28, 56), 2.0)}

    class Processor:
        def __call__(self, *, image, intrinsics, **kwargs):
            calls["rgb"] = image[0].copy()
            calls["processor_kwargs"] = kwargs
            h, w = image[0].shape[:2]
            K = intrinsics.copy()
            K[:, 0] *= 56 / w
            K[:, 1] *= 28 / h
            if crop:
                K[:, 0, 2] -= 1
            return torch.zeros((1, 3, 28, 56)), None, torch.from_numpy(K)

    config_module = types.ModuleType("depth_anything_3.cfg")
    config_module.create_object = lambda config: Network()
    processor_module = types.ModuleType("depth_anything_3.utils.io.input_processor")
    processor_module.InputProcessor = Processor
    safetensors_module = types.ModuleType("safetensors.torch")

    def load_file(path, device="cpu"):
        calls["load_path"] = path
        calls["load_device"] = device
        return {"model.mock_weight": torch.tensor(1.0)}

    safetensors_module.load_file = load_file
    monkeypatch.setitem(sys.modules, "depth_anything_3.cfg", config_module)
    monkeypatch.setitem(sys.modules, "depth_anything_3.utils.io.input_processor", processor_module)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_module)
    return checkpoint, source, calls, torch


def test_predict_passes_rgb_and_exact_intrinsics_without_cpu_autocast(tmp_path, monkeypatch):
    checkpoint, source, calls, torch = _fake_official_model(tmp_path, monkeypatch)
    predictor = DA3Predictor(checkpoint, "cpu", source)
    camera = Camera(120, 80, 32, 16, 64, 32)
    bgr = np.full((32, 64, 3), [10, 20, 30], dtype=np.uint8)
    # An outer caller's autocast must not silently lower the CPU comparison precision.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        depth = predictor.predict(bgr, camera)
    assert np.all(calls["rgb"] == [30, 20, 10])
    assert calls["network_shape"] == (1, 1, 3, 28, 56)
    assert calls["network_dtype"] == torch.float32
    assert not calls["cpu_autocast"] and not calls["grad_enabled"]
    assert calls["load_device"] == "cpu"
    assert calls["load_path"] == str(checkpoint.resolve() / "model.safetensors")
    assert set(calls["weights"]) == {"mock_weight"} and calls["strict"]
    assert calls["processor_kwargs"]["process_res_method"] == "upper_bound_resize"
    # Resized fx=105, fy=70; canonical=2 -> 2*(105+70)/600 metres.
    assert depth.shape == (32, 64) and depth.dtype == np.float32
    assert np.allclose(depth, 175 / 300)
    assert predictor.last_preprocessing["metric_scale"] == pytest.approx(175 / 600)


def test_predict_rejects_unexpected_crop_instead_of_stretching_it(tmp_path, monkeypatch):
    checkpoint, source, _, _ = _fake_official_model(tmp_path, monkeypatch, crop=True)
    predictor = DA3Predictor(checkpoint, "cpu", source)
    with pytest.raises(ValueError, match="crop/intrinsics"):
        predictor.predict(np.zeros((32, 64, 3), dtype=np.uint8), Camera(120, 80, 32, 16, 64, 32))


def test_relative_checkpoint_and_half_cpu_are_not_silently_accepted(tmp_path, monkeypatch):
    checkpoint, source, _, _ = _fake_official_model(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="CUDA"):
        DA3Predictor(checkpoint, "cpu", source, precision="float16")
    (checkpoint / "config.json").write_text(json.dumps({"model_name": "da3-large"}))
    with pytest.raises(ValueError, match="DA3METRIC-LARGE"):
        DA3Predictor(checkpoint, "cpu", source)
