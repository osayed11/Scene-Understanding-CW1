"""Offline, camera-calibrated adapter for the official DA3METRIC-LARGE model.

The official InputProcessor supplies both the normalized image and its exact
resized intrinsics. Only its single-image ``upper_bound_resize`` path is used:
there is no crop whose field of view would need to be restored. Canonical depth
is multiplied by mean(processed_fx, processed_fy)/300 before linear resizing to
the native image. Neither ground-truth poses nor depth are input to the model.

The high-level DA3 API chooses autocast precision automatically, including FP16
on a CPU. We instead call its underlying network under an explicit precision
context, retaining the official preprocessing and network/sky postprocessing.
Sources: Depth-Anything-3/utils/io/input_processor.py, utils/alignment.py and
model/da3.py in https://github.com/ByteDance-Seed/Depth-Anything-3 .
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from part2.common import Camera

MODEL_NAME = "da3metric-large"
PROCESS_METHOD = "upper_bound_resize"


def canonical_to_metric_depth(
    canonical_depth: np.ndarray,
    processed_intrinsics: np.ndarray,
    native_shape: tuple[int, int],
) -> np.ndarray:
    """Convert processed canonical camera-z depth to native HxW float32 metres.

Use the K returned by official preprocessing: a crop changes principal point,
whereas resizing changes focal lengths. Inferring focal from output dimensions
alone would be wrong for a cropped input. This adapter itself uses resize only.
"""
    depth = np.asarray(canonical_depth, dtype=np.float32)
    intrinsics = np.asarray(processed_intrinsics, dtype=np.float64)
    if depth.ndim != 2 or min(depth.shape, default=0) < 1:
        raise ValueError("canonical depth must be a nonempty HxW map")
    if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
            or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0
            or not np.allclose(intrinsics[2], [0, 0, 1])):
        raise ValueError("processed intrinsics must be a finite pinhole K")
    if (len(native_shape) != 2 or any(not isinstance(s, (int, np.integer)) or s < 1
                                      for s in native_shape)):
        raise ValueError("native_shape must be positive integer (height, width)")
    focal = float((intrinsics[0, 0] + intrinsics[1, 1]) / 2)
    metric = depth * np.float32(focal / 300.0)
    if metric.shape != tuple(native_shape):
        metric = cv2.resize(metric, (int(native_shape[1]), int(native_shape[0])),
                            interpolation=cv2.INTER_LINEAR)
    return np.asarray(metric, dtype=np.float32)


def _checkpoint_directory(checkpoint: Path) -> Path:
    path = Path(checkpoint).expanduser().resolve()
    if path.is_file() and path.name == "model.safetensors":
        path = path.parent
    if not path.is_dir() or not (path / "config.json").is_file() or not (path / "model.safetensors").is_file():
        raise FileNotFoundError("DA3 checkpoint must be a local directory containing config.json and model.safetensors")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if str(config.get("model_name", "")).lower() != MODEL_NAME:
        raise ValueError("Only the DA3METRIC-LARGE checkpoint is supported; relative-depth and nested models use different scale semantics")
    return path


class DA3Predictor:
    """Camera-aware predictor; model weights are loaded from local files only."""

    def __init__(self, checkpoint: Path, device: str, source_root: Path,
                 precision: str = "float32", process_res: int = 504):
        if precision not in ("float32", "float16", "bfloat16"):
            raise ValueError("precision must be float32, float16 or bfloat16")
        if not isinstance(process_res, (int, np.integer)) or process_res < 14:
            raise ValueError("process_res must be an integer of at least 14 pixels")
        model_dir = _checkpoint_directory(checkpoint)
        source = Path(source_root).expanduser().resolve()
        if not (source / "src/depth_anything_3/api.py").is_file():
            raise FileNotFoundError(f"Official DA3 source missing under {source}")
        sys.path.insert(0, str(source / "src"))
        try:
            import torch
            from omegaconf import OmegaConf
            from safetensors.torch import load_file
            from depth_anything_3.cfg import create_object
            from depth_anything_3.utils.io.input_processor import InputProcessor
        except ImportError as exc:
            raise RuntimeError("DA3 dependencies are unavailable; install the pinned official source and its dependencies") from exc
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda", "mps"):
            raise ValueError("DA3 adapter supports cpu, cuda and mps devices")
        if precision != "float32" and self.device.type != "cuda":
            raise ValueError("Reduced precision is supported only on CUDA; use float32 for CPU/MPS")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback is performed")
        if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This CUDA device does not support bfloat16")
        self.precision = precision
        self.process_res = int(process_res)
        # Build precisely the architecture saved by the official HF wrapper,
        # without importing API export/viewer dependencies. The safetensors keys
        # include the wrapper's `model.` prefix; this instance is its network.
        model_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        if not isinstance(model_config.get("config"), dict):
            raise ValueError("DA3 checkpoint config.json has no saved architecture config")
        self.model = create_object(OmegaConf.create(model_config["config"]))
        weights = load_file(str(model_dir / "model.safetensors"), device="cpu")
        if not weights or not all(key.startswith("model.") for key in weights):
            raise ValueError("Expected official DA3 wrapper-prefixed safetensors weights")
        self.model.load_state_dict({key[len("model."):]: value for key, value in weights.items()}, strict=True)
        del weights
        self.model = self.model.to(device=self.device, dtype=torch.float32).eval()
        self.input_processor = InputProcessor()
        self.metadata = {
            "model": "da3_metric_large",
            "model_variant": MODEL_NAME, "process_res": self.process_res,
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "process_res_method": PROCESS_METHOD, "input_color": "RGB",
            "precision": precision,
            "precision_policy": "float32_no_autocast" if precision == "float32" else f"explicit_{precision}_autocast",
            "metric_conversion": "canonical_depth * mean(processed_fx, processed_fy) / 300",
            "native_output_interpolation": "bilinear_after_metric_conversion",
            "calibration_source": "official_input_processor_returned_intrinsics",
        }
        self.last_preprocessing = None

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            self.torch.mps.synchronize()

    def predict(self, bgr: np.ndarray, camera: Camera) -> np.ndarray:
        image = np.asarray(bgr)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("DA3 input must be an HxWx3 uint8 BGR image")
        if image.shape[:2] != (camera.height, camera.width):
            raise ValueError("Image dimensions differ from camera calibration")
        K = np.asarray(camera.K, dtype=np.float32)
        if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("Camera intrinsics must be finite with positive focal lengths")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with self.torch.inference_mode():
            images, _, processed_K = self.input_processor(
                image=[rgb], intrinsics=K[None].copy(), process_res=self.process_res,
                process_res_method=PROCESS_METHOD, num_workers=1,
                sequential=True, print_progress=False,
            )
            # Official _stack_batch returns Nx3xHxW; add batch dimension as the
            # public API's _prepare_model_inputs does. Reject unexpected layouts.
            if images.ndim != 4 or tuple(images.shape[:2]) != (1, 3):
                raise ValueError("DA3 processor must return one Nx3xHxW image")
            height, width = map(int, images.shape[-2:])
            if min(height, width) < 14 or height % 14 or width % 14:
                raise ValueError("DA3 processing resolution must yield nonempty dimensions divisible by 14")
            if processed_K is None:
                raise ValueError("DA3 preprocessing did not return calibrated intrinsics")
            processed_K = processed_K.detach().float().cpu().numpy()
            if processed_K.shape != (1, 3, 3):
                raise ValueError("DA3 processor must return one calibrated 3x3 matrix")
            # A single resize-only image must retain its entire field of view.
            # If upstream preprocessing adds a crop, refuse to stretch it back.
            expected_K = K.copy()
            expected_K[0] *= width / camera.width
            expected_K[1] *= height / camera.height
            if not np.allclose(processed_K[0], expected_K, rtol=1e-5, atol=1e-4):
                raise ValueError("Unexpected DA3 crop/intrinsics transform; cannot restore the full native image by resizing")
            inputs = images.to(device=self.device, dtype=self.torch.float32)[None]
            reduced = self.precision != "float32"
            dtype = getattr(self.torch, self.precision)
            with self.torch.autocast(device_type=self.device.type, dtype=dtype, enabled=reduced):
                # Calling the underlying network bypasses DepthAnything3.forward,
                # whose automatic autocast would override our FP32 comparison.
                raw = self.model(inputs, extrinsics=None, intrinsics=None,
                                 export_feat_layers=[], infer_gs=False)
            canonical = raw["depth"].detach().float().cpu().numpy()
        while canonical.ndim > 2:
            if canonical.shape[0] == 1:
                canonical = canonical[0]
            elif canonical.shape[-1] == 1:
                canonical = canonical[..., 0]
            else:
                raise ValueError(f"Expected one DA3 depth map, got {canonical.shape}")
        if canonical.shape != (height, width):
            raise ValueError(f"DA3 depth {canonical.shape} does not match processed image {(height, width)}")
        self.last_preprocessing = {
            "processed_shape": [height, width],
            "processed_intrinsics": processed_K[0].tolist(),
            "metric_scale": float((processed_K[0, 0, 0] + processed_K[0, 1, 1]) / 600),
        }
        return canonical_to_metric_depth(canonical, processed_K[0], image.shape[:2])
