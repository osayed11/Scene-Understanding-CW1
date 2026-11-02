import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from part2 import prepare_da3_full as packager


def fixture_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(packager, "EXPECTED_COUNTS", {name: 2 for name in packager.EXPECTED_COUNTS})
    repo, data, calibration = tmp_path / "repo", tmp_path / "data", tmp_path / "calibration"
    for relative in packager.SCRIPT_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# allowed script\n")
    for sequence, name in packager.CALIBRATIONS.items():
        config = repo / f"configs/sequences/{sequence}.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("depth_factor: 5000\n")
        settings = calibration / "settings" / name
        settings.parent.mkdir(parents=True, exist_ok=True)
        settings.write_text("Camera.fx: 600\nCamera.fy: 601\nCamera.cx: 2\nCamera.cy: 1\nCamera.width: 4\nCamera.height: 2\n")
    tum = data / "tum/rgbd_dataset_freiburg3_long_office_household"
    folders = [data / "kitti06/image_2", tum / "rgb", data / "replica/room0/results"]
    for folder in folders:
        folder.mkdir(parents=True)
    (data / "kitti06/times.txt").write_text("0\n0.1\n")
    (tum / "rgb.txt").write_text("# timestamps must retain decimal stems\n100.123456 rgb/100.123456.png\n100.223456 rgb/100.223456.png\n")
    for index in range(2):
        (folders[0] / f"{index:06d}.png").write_bytes(f"KITTI {index}".encode())
        (folders[1] / ("100.123456.png" if index == 0 else "100.223456.png")).write_bytes(f"TUM {index}".encode())
        (folders[2] / f"frame{index:06d}.jpg").write_bytes(f"Replica {index}".encode())
        (folders[2] / f"depth{index:06d}.png").write_bytes(b"excluded ground truth")
    (data / "kitti06/video_color.mp4").write_bytes(b"excluded video")
    (repo / "secrets.env").write_bytes(b"excluded secret")
    return repo, data, calibration


def test_full_bundle_stream_hashes_and_strict_allowlist(tmp_path, monkeypatch):
    repo, data, calibration = fixture_inputs(tmp_path, monkeypatch)
    output = tmp_path / "out/inputs.zip"
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in data.rglob("*") if p.is_file()}
    report = packager.build_bundle(repo_root=repo, data_root=data, calibration_root=calibration, output=output)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert names[-1] == "cw1_da3_full/full_input_manifest.json"
        assert len(names) == len(set(names)) == 26  # 11 scripts + 3 configs + 3 cameras + 2 timestamp lists + 6 RGB + manifest
        assert not any("depth000" in name or "video" in name or "secrets" in name or "weights" in name for name in names)
        manifest = json.loads(archive.read(names[-1]))
        assert manifest["total_frames"] == 6
        assert manifest["sequences"]["tum_fr3"]["frames"][0]["rgb"].endswith("100.123456.png")
        assert [f["timestamp"] for f in manifest["sequences"]["replica_room0"]["frames"]] == [0, .04]
        for sequence in manifest["sequences"].values():
            for frame in sequence["frames"]:
                name = "cw1_da3_full/" + frame["rgb"]
                assert archive.getinfo(name).compress_type == zipfile.ZIP_STORED
                assert hashlib.sha256(archive.read(name)).hexdigest() == frame["rgb_sha256"]
        assert str(tmp_path) not in archive.read(names[-1]).decode()
    assert report["uncompressed_npy_bytes"] == 6 * (4 * 2 * 4 + 128)
    assert report["bytes"] == output.stat().st_size
    assert (output.parent / "full_input_manifest.json").read_bytes() == manifest_bytes(output)
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == snapshot for p, snapshot in before.items())


def manifest_bytes(output):
    with zipfile.ZipFile(output) as archive:
        return archive.read("cw1_da3_full/full_input_manifest.json")


def test_inventory_rejects_duplicate_paths_missing_frame_and_escape(tmp_path, monkeypatch):
    repo, data, calibration = fixture_inputs(tmp_path, monkeypatch)
    rgb = data / "tum/rgbd_dataset_freiburg3_long_office_household/rgb.txt"
    rgb.write_text("100 rgb/100.123456.png\n101 rgb/100.123456.png\n")
    with pytest.raises(ValueError, match="Duplicate"):
        packager.inventory_inputs(repo_root=repo, data_root=data, calibration_root=calibration)
    rgb.write_text("100 rgb/../../secrets.png\n101 rgb/100.223456.png\n")
    with pytest.raises(ValueError, match="only files within"):
        packager.inventory_inputs(repo_root=repo, data_root=data, calibration_root=calibration)


def test_checked_file_rejects_symlink(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("input")
    alias = tmp_path / "alias.txt"
    try:
        alias.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation not available")
    with pytest.raises(ValueError, match="Symlink"):
        packager.checked_file(alias, tmp_path)
