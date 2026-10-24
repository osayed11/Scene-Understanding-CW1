import json

import cv2
import numpy as np
import pytest

from depth import depth_at_keypoints
from part2 import common, predict_depth
from part2.common import Camera, FrameRecord


def test_keypoint_patch_rejects_outlier_and_invalid_coordinates():
    depth = np.full((5, 5), 2., dtype=np.float32)
    depth[2, 2] = 20
    values, sigma = depth_at_keypoints(depth, np.array([[2, 2], [-.1, 0], [np.nan, 0]]))
    assert values[0] == 2 and sigma[0] == pytest.approx(.04)
    assert np.isnan(values[1:]).all() and np.isnan(sigma[1:]).all()


def test_tum_keeps_rgb_frames_without_depth_and_reports_timestamp_offset(tmp_path, monkeypatch):
    folder = tmp_path / "tum" / "sequence"
    folder.mkdir(parents=True)
    (folder / "rgb.txt").write_text("1 rgb1.png\n1.01 rgb2.png\n2 rgb3.png\n")
    (folder / "depth.txt").write_text("1.005 depth.png\n")
    for name in ("rgb1.png", "rgb2.png", "rgb3.png", "depth.png"):
        (folder / name).touch()
    monkeypatch.setattr(common, "load_sequence", lambda _: {"root": "data/tum/sequence", "images": "data/tum/sequence/rgb"})
    frames = common.load_frame_records("tum_fr3", tmp_path)
    assert len(frames) == 3
    assert frames[0].depth_gt == frames[1].depth_gt
    assert frames[0].depth_timestamp == 1.005
    assert frames[2].depth_gt is None


def test_pose_interpolation_does_not_extrapolate(tmp_path, monkeypatch):
    rows = np.array([[1, 0, 0, 0, 0, 0, 0, 1], [1.02, 2, 0, 0, 0, 0, 0, 1]])
    np.savetxt(tmp_path / "gt.txt", rows)
    monkeypatch.setattr(common, "load_sequence", lambda _: {"groundtruth": "data/gt.txt"})
    frames = [FrameRecord(i, t, tmp_path / "image.png") for i, t in enumerate([.99, 1.01, 1.03])]
    poses = common.load_frame_poses("tum_fr3", frames, tmp_path)
    assert np.isnan(poses[[0, 2]]).all()
    assert poses[1, 0, 3] == pytest.approx(1.)


def test_prediction_cache_resume_and_identity_guard(tmp_path, monkeypatch):
    image = tmp_path / "1341847980.722988.png"
    assert cv2.imwrite(str(image), np.zeros((3, 4, 3), dtype=np.uint8))
    frame = FrameRecord(0, 1341847980.722988, image)
    monkeypatch.setattr(predict_depth, "load_frame_records", lambda *args: [frame])
    monkeypatch.setattr(predict_depth, "load_camera", lambda *args: Camera(4, 4, 1.5, 1, 4, 3))
    monkeypatch.setattr(predict_depth, "load_sequence", lambda _: {"depth_factor": 1000})

    class Predictor:
        calls = 0

        def synchronize(self):
            pass

        def predict(self, bgr, focal_px):
            self.calls += 1
            assert focal_px == 4
            return np.full(bgr.shape[:2], 2., dtype=np.float32)

    model = Predictor()
    identity = {"model": "test_only", "checkpoint_sha256": "test"}
    root = tmp_path / "predictions"
    first = predict_depth.predict_sequence("tum_fr3", model, identity, pred_root=root)
    second = predict_depth.predict_sequence("tum_fr3", model, identity, pred_root=root)
    assert first["new_frames"] == 1 and second["new_frames"] == 0 and model.calls == 1
    manifest = json.loads((root / "tum_fr3" / "manifest.json").read_text())
    assert frame.stem in manifest["frames"]
    assert (root / "tum_fr3" / (frame.stem + ".npy")).exists()
    columns = (root / "tum_fr3" / "associations.txt").read_text().splitlines()[1].split()
    assert (root / "tum_fr3" / columns[1]).resolve() == image.resolve()
    assert not columns[1].startswith("/") and columns[3] == frame.stem + ".png"
    with pytest.raises(ValueError, match="different model"):
        predict_depth.predict_sequence("tum_fr3", model, {"model": "different"}, pred_root=root)
    assert cv2.imwrite(str(image), np.ones((3, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="cached RGB content"):
        predict_depth.predict_sequence("tum_fr3", model, identity, pred_root=root)


def test_replica_missing_frame_cannot_shift_ground_truth(tmp_path, monkeypatch):
    folder = tmp_path / "replica" / "room0" / "results"
    folder.mkdir(parents=True)
    (folder / "frame000000.jpg").touch()
    (folder / "frame000002.jpg").touch()
    monkeypatch.setattr(common, "load_sequence", lambda _: {"images": "data/replica/room0/results"})
    with pytest.raises(ValueError, match="non-contiguous"):
        common.load_frame_records("replica_room0", tmp_path)
