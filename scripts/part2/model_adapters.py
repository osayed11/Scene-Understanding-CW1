"""Explicit metric-depth adapters for controlled model-selection benchmarks.

Models are lazy-loaded; importing this module does not import Torch. Checkpoints
are local and never silently downloaded. A domain-specific V2 checkpoint is
identified in metadata; using different domains is a separate declared case.
"""
from __future__ import annotations

from pathlib import Path
import sys


MODEL_NAMES = ("depth_pro", "dav2_small", "dav2_base", "dav2_large", "da3_metric_large")


class _TorchAdapter:
    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            self.torch.mps.synchronize()


class DepthProAdapter:
    def __init__(self, checkpoint, source_root, device="cpu", precision="float32"):
        from part2.predict_depth import DepthProPredictor
        self.delegate = DepthProPredictor(Path(checkpoint), device, Path(source_root), precision=precision)
        self.metadata = {"model": "depth_pro", "precision": precision,
                         "parameter_count": sum(p.numel() for p in self.delegate.model.parameters()),
                         "internal_size": [1536, 1536], "input_color": "RGB",
                         "metric_conversion": "calibrated_native_focal_px"}

    def synchronize(self):
        self.delegate.synchronize()

    def predict(self, bgr, camera):
        return self.delegate.predict(bgr, camera.fx)


class DAV2Predictor(_TorchAdapter):
    CONFIGS = {
        "dav2_small": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "dav2_base": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "dav2_large": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    def __init__(self, model_name, checkpoint, source_root, device="cpu", precision="float32",
                 domain="hypersim", process_res=518):
        if domain not in ("hypersim", "vkitti"):
            raise ValueError("V2 metric domain must be hypersim or vkitti")
        checkpoint = Path(checkpoint)
        config = self.CONFIGS[model_name]
        expected = f"depth_anything_v2_metric_{domain}_{config['encoder']}.pth"
        if checkpoint.name != expected:
            raise ValueError(f"Expected the declared metric checkpoint {expected}, got {checkpoint.name}")
        sys.path.insert(0, str(Path(source_root) / "metric_depth"))
        import torch
        from depth_anything_v2.dpt import DepthAnythingV2
        from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
        import cv2
        self.torch = torch
        self.device = torch.device(device)
        if precision not in ("float32", "float16") or (precision == "float16" and self.device.type != "cuda"):
            raise ValueError("Use float32 on CPU; float16 is available for CUDA comparisons")
        self.dtype = torch.float16 if precision == "float16" else torch.float32
        self.process_res = int(process_res)
        if self.process_res < 14:
            raise ValueError("V2 process_res must be at least 14 pixels")
        # Same transforms/settings as official image2tensor, without its
        # automatic CUDA/MPS selection. Device choice belongs to this adapter.
        self.transforms = (
            Resize(width=self.process_res, height=self.process_res,
                   resize_target=False, keep_aspect_ratio=True,
                   ensure_multiple_of=14, resize_method="lower_bound",
                   image_interpolation_method=cv2.INTER_CUBIC),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        )
        max_depth = 20. if domain == "hypersim" else 80.
        self.model = DepthAnythingV2(**config, max_depth=max_depth)
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        self.model.load_state_dict(weights, strict=True)
        del weights
        self.model.to(device=self.device, dtype=self.dtype).eval()
        self.metadata = {"model": model_name, "precision": precision, "domain": domain,
                         "parameter_count": sum(p.numel() for p in self.model.parameters()),
                         "process_res": self.process_res, "resize_policy": "lower_bound_short_side",
                         "max_depth_m": max_depth, "input_color": "BGR_to_RGB_in_official_processor",
                         "metric_conversion": "matching_trained_max_depth"}

    def predict(self, bgr, camera):
        import cv2
        # Official image2tensor auto-selects CUDA whenever available, even for a
        # requested CPU benchmark. Reuse its numerical transforms directly so
        # there is exactly one transfer, to the explicitly requested device.
        with self.torch.inference_mode():
            native_size = bgr.shape[:2]
            sample = {"image": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0}
            for transform in self.transforms:
                sample = transform(sample)
            image = self.torch.from_numpy(sample["image"]).unsqueeze(0)
            image = image.to(device=self.device, dtype=self.dtype)
            depth = self.model(image)
            depth = self.torch.nn.functional.interpolate(
                depth[:, None], native_size, mode="bilinear", align_corners=True)[0, 0]
        return depth.float().cpu().numpy()


def create_predictor(model_name, checkpoint, source_root, device="cpu", precision="float32",
                     domain="hypersim", process_res=None):
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model_name}; choose from {MODEL_NAMES}")
    if not Path(checkpoint).exists():
        raise FileNotFoundError(checkpoint)
    if not Path(source_root).is_dir():
        raise FileNotFoundError(source_root)
    if model_name == "depth_pro":
        if process_res is not None:
            raise ValueError("Depth Pro has a fixed internal resolution; process_res does not apply")
        return DepthProAdapter(checkpoint, source_root, device, precision)
    if model_name.startswith("dav2_"):
        return DAV2Predictor(model_name, checkpoint, source_root, device, precision, domain,
                             518 if process_res is None else process_res)
    from part2.da3_adapter import DA3Predictor
    return DA3Predictor(Path(checkpoint), device, Path(source_root), precision=precision,
                        process_res=504 if process_res is None else process_res)
