"""Cache integrity and recovery tests; no model weights or Torch are needed."""
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pytest

from part2.common import Camera, FrameRecord
from part2 import predict_da3 as runner


class FakePredictor:
    def __init__(self, depth=None, fail_on=None):
        self.calls = []
        self.syncs = 0
        self.depth = np.full((2, 3), 2, dtype=np.float32) if depth is None else depth
        self.fail_on = fail_on
        self.last_preprocessing = {"processed_shape": [2, 3], "metric_scale": .1,
                                   "processed_intrinsics": [[30., 0., 1.], [0., 30., 1.], [0., 0., 1.]]}

    def synchronize(self):
        self.syncs += 1

    def predict(self, image, camera):
        self.calls.append((image.copy(), camera))
        if self.fail_on == len(self.calls):
            raise RuntimeError("injected inference failure")
        return self.depth.copy()


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    root = tmp_path / "original" / "data"
    folder = root / "tum" / "rgb"
    folder.mkdir(parents=True)
    names = ["1341847980.722988", "1341847980.756320"]
    for index, stem in enumerate(names):
        image = np.full((2, 3, 3), 30 + index, dtype=np.uint8)
        assert cv2.imwrite(str(folder / f"{stem}.png"), image)
    camera = Camera(10., 12., .75, .6, 3, 2)

    def records(sequence, data_root):
        return [FrameRecord(i, float(stem), Path(data_root) / "tum/rgb" / f"{stem}.png",
                            Path(data_root) / "ground_truth_must_not_be_read.png", float(stem))
                for i, stem in enumerate(names)]

    monkeypatch.setattr(runner, "load_frame_records", records)
    monkeypatch.setattr(runner, "load_camera", lambda *args: camera)
    monkeypatch.setattr(runner, "load_sequence", lambda *args: {"depth_factor": 1000.})
    return {"data_root": root, "pred_root": tmp_path / "cache", "camera": camera,
            "names": names, "identity": {"model": "da3_metric_large", "checkpoint": {"sha256": "fixed"},
                                           "source_commit": "fixed", "precision": "float32", "device": "cpu",
                                           "process_res": 504}}


def run(dataset, predictor=None, **kwargs):
    options = {key: dataset[key] for key in ("data_root", "pred_root")}
    options.update(kwargs)
    return runner.predict_sequence_da3("tum_fr3", predictor or FakePredictor(), dataset["identity"], **options)


def manifest(dataset):
    return json.loads((dataset["pred_root"] / "tum_fr3/manifest.json").read_text())


def test_native_camera_decimal_stems_portable_hashes_and_cached_resume(dataset):
    predictor = FakePredictor()
    result = run(dataset, predictor, session_metadata={"gpu_name": "first GPU"})
    saved = manifest(dataset)
    assert result["full_complete"] and saved["status"] == "complete"
    assert saved["schema_version"] == 2 and saved["full_count"] == 2
    assert saved["signature"]["camera"]["K"] == dataset["camera"].K.tolist()
    assert len(predictor.calls) == 3  # excluded warmup, then the two actual frames
    assert predictor.syncs == 6
    assert all(camera == dataset["camera"] for _, camera in predictor.calls)
    assert all(image.shape == (2, 3, 3) for image, _ in predictor.calls)
    folder = dataset["pred_root"] / "tum_fr3"
    for stem, entry in saved["frames"].items():
        assert entry["rgb"] == f"data/tum/rgb/{stem}.png"
        assert entry["npy_sha256"] == runner.sha256(folder / f"{stem}.npy")
        assert entry["png_sha256"] == runner.sha256(folder / f"{stem}.png")
        assert np.load(folder / f"{stem}.npy").dtype == np.float32
        assert entry["preprocessing"]["metric_scale"] == .1
    associations = (folder / "associations.txt").read_text()
    assert all(f"data/tum/rgb/{stem}.png" in associations for stem in dataset["names"])
    assert not (folder / "1341847980.npy").exists()
    resumed = FakePredictor()
    summary = run(dataset, resumed, session_metadata={"gpu_name": "second GPU"})
    assert summary["new_frames"] == 0 and summary["reused_frames"] == 2
    assert resumed.calls == []
    assert manifest(dataset)["frames"] == saved["frames"]
    assert manifest(dataset)["sessions"] == saved["sessions"]


def test_partial_selection_can_extend_without_losing_entries_or_timing(dataset):
    run(dataset, max_frames=1, session_metadata={"gpu_name": "old GPU"})
    old = manifest(dataset)
    assert old["status"] == "complete" and not old["full_complete"]
    assert old["selection"] == {"stride": 1, "max_frames": 1, "requested_count": 1, "indices": [0]}
    predictor = FakePredictor()
    run(dataset, predictor, session_metadata={"gpu_name": "new GPU"})
    saved = manifest(dataset)
    assert saved["full_complete"] and len(predictor.calls) == 2
    assert saved["frames"][dataset["names"][0]] == old["frames"][dataset["names"][0]]
    assert len(saved["sessions"]) == 2
    # A subsequent smaller request retains all completed frames for export.
    run(dataset, max_frames=1)
    assert len(manifest(dataset)["frames"]) == 2


def test_relocation_preserves_portable_identity_and_reuses_outputs(dataset, tmp_path):
    run(dataset)
    relocated = tmp_path / "colab_copy" / "data"
    shutil.copytree(dataset["data_root"], relocated)
    predictor = FakePredictor()
    result = run(dataset, predictor, data_root=relocated)
    assert result["reused_frames"] == 2 and predictor.calls == []


def test_changed_rgb_requires_explicit_overwrite(dataset):
    run(dataset)
    rgb = dataset["data_root"] / "tum/rgb" / f"{dataset['names'][0]}.png"
    assert cv2.imwrite(str(rgb), np.full((2, 3, 3), 99, dtype=np.uint8))
    with pytest.raises(ValueError, match="RGB checksum changed"):
        run(dataset)
    old_second = manifest(dataset)["frames"][dataset["names"][1]]
    run(dataset, overwrite=True, max_frames=1)
    assert manifest(dataset)["frames"][dataset["names"][1]] == old_second
    assert manifest(dataset)["frames"][dataset["names"][0]]["rgb_sha256"] == runner.sha256(rgb)


def test_overwrite_cannot_hide_changed_unselected_rgb(dataset):
    run(dataset)
    rgb = dataset["data_root"] / "tum/rgb" / f"{dataset['names'][1]}.png"
    assert cv2.imwrite(str(rgb), np.full((2, 3, 3), 99, dtype=np.uint8))
    with pytest.raises(ValueError, match="RGB checksum changed"):
        run(dataset, overwrite=True, max_frames=1)


@pytest.mark.parametrize("suffix", ["npy", "png"])
def test_corrupted_output_checksum_is_rejected_and_can_be_recomputed(dataset, suffix):
    run(dataset)
    path = dataset["pred_root"] / "tum_fr3" / f"{dataset['names'][0]}.{suffix}"
    path.write_bytes(b"corrupted output")
    with pytest.raises(ValueError, match="checksum differs"):
        run(dataset)
    run(dataset, overwrite=True, max_frames=1)
    assert runner.sha256(path) == manifest(dataset)["frames"][dataset["names"][0]][f"{suffix}_sha256"]


@pytest.mark.parametrize("field,value", [("fy", 13.), ("cx", .8), ("cy", .8)])
def test_full_calibration_change_requires_separate_cache(dataset, monkeypatch, field, value):
    run(dataset)
    new_camera = replace(dataset["camera"], **{field: value})
    monkeypatch.setattr(runner, "load_camera", lambda *args: new_camera)
    with pytest.raises(ValueError, match="calibration"):
        run(dataset, overwrite=True)


def test_model_or_precision_identity_cannot_be_overwritten(dataset):
    run(dataset)
    dataset["identity"]["precision"] = "float16"
    with pytest.raises(ValueError, match="model, calibration"):
        run(dataset, overwrite=True)


@pytest.mark.parametrize("overwrite", [False, True])
def test_unidentified_outputs_are_never_adopted(dataset, overwrite):
    folder = dataset["pred_root"] / "tum_fr3"
    folder.mkdir(parents=True)
    np.save(folder / f"{dataset['names'][0]}.npy", np.ones((2, 3), np.float32))
    with pytest.raises(ValueError, match="Unidentified prediction"):
        run(dataset, overwrite=overwrite)
    assert not (folder / "manifest.json").exists()


def test_failure_preserves_commits_and_resume_only_processes_remaining(dataset):
    predictor = FakePredictor(fail_on=3)  # warmup succeeds, first frame commits, second fails
    with pytest.raises(RuntimeError, match="injected inference failure"):
        run(dataset, predictor)
    failed = manifest(dataset)
    assert failed["status"] == "failed" and len(failed["frames"]) == 1
    assert failed["failure"]["frame"] == dataset["names"][1]
    resumed = FakePredictor()
    result = run(dataset, resumed)
    assert result["new_frames"] == 1 and result["reused_frames"] == 1
    assert len(resumed.calls) == 2
    assert manifest(dataset)["frames"][dataset["names"][0]] == failed["frames"][dataset["names"][0]]


def test_interruption_between_outputs_and_manifest_commit_is_recoverable(dataset, monkeypatch):
    real_save = runner.save_depth

    def interrupted_save(*args, **kwargs):
        real_save(*args, **kwargs)
        raise OSError("simulated interruption after output renames")

    monkeypatch.setattr(runner, "save_depth", interrupted_save)
    with pytest.raises(OSError, match="simulated interruption"):
        run(dataset, max_frames=1)
    failed = manifest(dataset)
    assert failed["frames"] == {} and failed["pending"]["stem"] == dataset["names"][0]
    monkeypatch.setattr(runner, "save_depth", real_save)
    run(dataset, max_frames=1)
    assert manifest(dataset)["pending"] is None
    assert len(manifest(dataset)["frames"]) == 1


def test_png_overflow_invalid_preserves_metric_npy(dataset):
    depth = np.array([[1., np.nan, 100.], [0., 2., 3.]], dtype=np.float32)
    run(dataset, FakePredictor(depth), max_frames=1)
    folder = dataset["pred_root"] / "tum_fr3"
    stem = dataset["names"][0]
    np.testing.assert_equal(np.load(folder / f"{stem}.npy"), depth)
    png = cv2.imread(str(folder / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(png, [[1000, 0, 0], [0, 2000, 3000]])
    entry = manifest(dataset)["frames"][stem]
    assert entry["png_overflow_fraction"] == pytest.approx(1 / 6)
    assert entry["invalid_fraction"] == pytest.approx(2 / 6)


@pytest.mark.parametrize("depth,match", [(np.ones((1, 3), np.float32), "shape"),
                                        (np.full((2, 3), -1., np.float32), "negative"),
                                        (np.full((2, 3), np.nan, np.float32), "no valid")])
def test_invalid_network_output_never_commits(dataset, depth, match):
    with pytest.raises(ValueError, match=match):
        run(dataset, FakePredictor(depth))
    assert manifest(dataset)["frames"] == {}


def test_dry_run_reads_rgb_and_calibration_without_importing_torch_or_weights(tmp_path):
    data = tmp_path / "data"
    images = data / "replica/room0/results"
    images.mkdir(parents=True)
    assert cv2.imwrite(str(images / "frame000000.jpg"), np.full((2, 3, 3), 50, np.uint8))
    calibration = tmp_path / "calibration/settings"
    calibration.mkdir(parents=True)
    (calibration / "REPLICA.yaml").write_text(
        "Camera.fx: 10\nCamera.fy: 12\nCamera.cx: 1\nCamera.cy: 1\nCamera.width: 3\nCamera.height: 2\n")
    cache = tmp_path / "must_not_be_created"
    scripts = Path(runner.__file__).resolve().parents[1]
    code = ("import sys; sys.path.insert(0,sys.argv[1]); "
            "from part2.predict_da3 import main; "
            "assert main(['--sequence','replica_room0','--data-root',sys.argv[2],"
            "'--pred-root',sys.argv[3],'--checkpoint','missing-checkpoint',"
            "'--source-root','missing-source','--dry-run']) == 0; "
            "assert 'torch' not in sys.modules")
    env = {**os.environ, "CW1_PYSLAM_ROOT": str(calibration.parent)}
    result = subprocess.run([sys.executable, "-B", "-c", code, str(scripts), str(data), str(cache)],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    row = json.loads(result.stdout)
    assert row["full_count"] == row["requested_frames"] == 1 and row["dry_run"]
    assert not cache.exists()
