import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from part2 import benchmark_models as benchmark
from part2.common import Camera, FrameRecord


def test_selection_is_reproducible_and_supports_original_three():
    assert benchmark.select_indices(2000) == [200, 1000, 1799]
    assert benchmark.select_indices(2000, indices=[0, 1, 2]) == [0, 1, 2]
    assert benchmark.select_indices(1, 3) == [0]
    with pytest.raises(ValueError):
        benchmark.select_indices(3, indices=[1, 1])


def test_checkpoint_fingerprint_covers_directory_config_and_weights(tmp_path):
    checkpoint = tmp_path / "da3"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_name":"da3metric-large"}')
    (checkpoint / "model.safetensors").write_bytes(b"weights one")
    first = benchmark.checkpoint_fingerprint(checkpoint)
    assert first["kind"] == "directory"
    assert list(first["files"]) == ["config.json", "model.safetensors"]
    (checkpoint / "README.md").write_text("unrelated metadata")
    assert benchmark.checkpoint_fingerprint(checkpoint) == first
    (checkpoint / "config.json").write_text('{"model_name":"different"}')
    second = benchmark.checkpoint_fingerprint(checkpoint)
    assert second["sha256"] != first["sha256"]
    (checkpoint / "model.safetensors").write_bytes(b"weights two")
    assert benchmark.checkpoint_fingerprint(checkpoint)["sha256"] != second["sha256"]
    single = benchmark.checkpoint_fingerprint(checkpoint / "model.safetensors")
    assert single["kind"] == "file"
    assert single["sha256"] == benchmark.sha256(checkpoint / "model.safetensors")
    (checkpoint / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        benchmark.checkpoint_fingerprint(checkpoint)


def test_timing_synchronizes_both_ends():
    events = []
    class Predictor:
        def synchronize(self):
            events.append("sync")
        def predict(self, image, camera):
            events.append("predict")
            return np.ones(image.shape[:2])
    ticks = iter([2., 5.])
    prediction, seconds = benchmark.timed_prediction(Predictor(), np.zeros((2, 2, 3)), None,
                                                    clock=lambda: next(ticks))
    assert events == ["sync", "predict", "sync"] and seconds == 3
    assert prediction.shape == (2, 2)


def test_portable_samples_preserve_original_indices_and_intrinsics(tmp_path):
    path = tmp_path / "sample_manifest.json"
    path.write_text(json.dumps([{"sequence": "replica_room0", "index": 1799, "timestamp": 71.96,
        "rgb": "data/frame001799.jpg", "depth_gt": "data/depth001799.png", "depth_timestamp": 71.96,
        "fx": 600, "fy": 601, "cx": 599.5, "cy": 339.5, "width": 1200, "height": 680,
        "gt_depth_factor": 6553.5}]))
    sample = benchmark.load_samples(["replica_room0"], sample_manifest=path)[0]
    assert sample["frame"].index == 1799 and sample["frame"].timestamp == 71.96
    assert sample["camera"].fy == 601
    assert sample["frame"].rgb == tmp_path / "data/frame001799.jpg"


def test_benchmark_warmup_resume_failures_and_no_gt(tmp_path):
    camera = Camera(10, 10, 1, 1, 2, 2)
    samples = []
    for i in range(3):
        rgb = tmp_path / f"frame{i:06}.jpg"
        assert cv2.imwrite(str(rgb), np.full((2, 2, 3), i * 50, dtype=np.uint8))
        gt = tmp_path / f"depth{i:06}.png" if i == 0 else None
        if gt is not None:
            assert cv2.imwrite(str(gt), np.ones((2, 2), dtype=np.uint16) * 1000)
        samples.append({"sequence": "replica_room0", "frame": FrameRecord(i, i / 25, rgb, gt),
                        "camera": camera, "gt_depth_factor": 1000})
    class Predictor:
        calls = 0
        def synchronize(self):
            pass
        def predict(self, image, camera):
            self.calls += 1
            if image[0, 0, 0] > 90:
                raise RuntimeError("synthetic failure")
            return np.ones((2, 2), dtype=np.float32) * 2
    predictor = Predictor()
    output, cache = tmp_path / "result.json", tmp_path / "predictions"
    result = benchmark.run_benchmark(predictor, samples, signature={"device": "cpu"},
                                     output=output, pred_root=cache)
    assert predictor.calls == 4  # one excluded warmup plus three measured attempts
    assert result["status"] == "completed_with_failures"
    assert result["warmup"]["status"] == "ok"
    assert result["frames"][0]["raw"]["abs_rel"] == 1
    assert result["frames"][0]["median_scaled"]["abs_rel"] == 0
    assert result["frames"][1]["gt_status"] == "no_ground_truth"
    assert result["frames"][2]["status"] == "failed"
    assert result["summaries"]["overall"]["frames_failed"] == 1
    assert json.loads(output.read_text())["frames"][2]["error"].endswith("synthetic failure")
    retry = Predictor()
    benchmark.run_benchmark(retry, samples, signature={"device": "cpu"}, output=output, pred_root=cache)
    assert retry.calls == 1  # existing successes reused; failed frame fails its warmup on retry
    with pytest.raises(ValueError, match="configuration differs"):
        benchmark.run_benchmark(Predictor(), samples, signature={"device": "cuda"}, output=output, pred_root=cache)


def test_aggregation_does_not_treat_missing_gt_as_zero_error():
    rows = [{"status": "ok", "inference_seconds": 2, "gt_status": "ok",
             "raw": {"abs_rel": .2}, "median_scaled": {"abs_rel": .1}},
            {"status": "ok", "inference_seconds": 4, "gt_status": "no_ground_truth",
             "raw": None, "median_scaled": None}]
    result = benchmark.aggregate(rows)
    assert result["mean_inference_seconds"] == 3
    assert result["raw"]["abs_rel"] == .2 and result["frames_no_gt"] == 1
