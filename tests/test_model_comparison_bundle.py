"""Sparse comparison inputs retain their source identities and remain bounded."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from part2.prepare_model_comparison import (
    CALIBRATIONS, CONFIG_FILES, PREFIX, SCRIPT_FILES, package_bundle, select_sample_frames,
)


def test_sampling_matches_rounded_full_sequence_positions():
    frames = [SimpleNamespace(index=i, timestamp=i / 25) for i in range(2000)]
    selected = select_sample_frames(frames)
    assert [frame.index for frame in selected] == [200, 1000, 1799]
    assert [frame.timestamp for frame in selected] == [8.0, 40.0, 71.96]


def test_missing_tum_gt_uses_nearest_unused_available_frame_with_earlier_ties(tmp_path):
    depth = tmp_path / "sensor.png"
    depth.write_bytes(b"depth")
    frames = [SimpleNamespace(index=100 + i, timestamp=10 + i * .1,
                              depth_gt=depth, depth_timestamp=10 + i * .1 + .01)
              for i in range(11)]
    frames[1].depth_gt = None
    frames[9].depth_gt = None
    frames[10].depth_timestamp += .1  # Unusable timestamp association.
    selected = select_sample_frames(frames, require_gt=True)
    assert [frame.index for frame in selected] == [100, 105, 108]
    assert selected[0] is frames[0]
    assert selected[2].timestamp == frames[8].timestamp


def test_sampling_rejects_unavailable_gt_and_large_requests():
    frames = [SimpleNamespace(index=i, timestamp=i, depth_gt=None, depth_timestamp=None)
              for i in range(10)]
    with pytest.raises(ValueError, match="associated ground-truth depth"):
        select_sample_frames(frames, require_gt=True)
    with pytest.raises(ValueError, match="between 1 and 3"):
        select_sample_frames(frames, 4)


def project_fixture(tmp_path):
    repo = tmp_path / "project"
    data = tmp_path / "data"
    calibration = tmp_path / "calibration"
    for relative in (*SCRIPT_FILES, *CONFIG_FILES):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# allowlisted fixture\n")
    for name in CALIBRATIONS.values():
        path = calibration / "settings" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Camera.fx: 600.0\n")
    manifest = []
    for sequence, index in (("replica_room0", 200), ("tum_fr3", 258), ("kitti06", 110)):
        rgb = f"data/{sequence}/original_{index}.png"
        depth = f"data/{sequence}/depth_{index}.png" if sequence != "kitti06" else None
        for relative in (rgb, depth):
            if relative is not None:
                path = data / Path(relative).relative_to("data")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(relative.encode())
        manifest.append({
            "sequence": sequence, "index": index, "timestamp": 12.5,
            "rgb": rgb, "depth_gt": depth, "depth_timestamp": 12.51 if depth else None,
            "fx": 600.0, "fy": 601.0, "cx": 599.5, "cy": 339.5,
            "width": 1200, "height": 680, "gt_depth_factor": 5000 if depth else None,
        })
    (repo / "secrets.env").write_text("must not appear")
    (repo / "model.pt").write_bytes(b"weights must not appear")
    unused = data / "replica_room0/original_999.png"
    unused.write_bytes(b"unselected image must not appear")
    return repo, data, calibration, manifest


def test_bundle_exact_allowlist_preserves_sparse_manifest_and_is_reproducible(tmp_path):
    repo, data, calibration, manifest = project_fixture(tmp_path)
    output = tmp_path / "bundle.zip"
    result = package_bundle(manifest, repo_root=repo, data_root=data,
                            calibration_root=calibration, output=output)
    expected = {f"{PREFIX}/{name}" for name in (*SCRIPT_FILES, *CONFIG_FILES, "sample_manifest.json")}
    expected |= {f"{PREFIX}/calibration/settings/{name}" for name in CALIBRATIONS.values()}
    expected |= {f"{PREFIX}/{relative}" for row in manifest for relative in (row["rgb"], row["depth_gt"])
                 if relative is not None}
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == expected
        recovered = json.loads(archive.read(f"{PREFIX}/sample_manifest.json"))
        assert recovered == manifest
        assert recovered[0]["index"] == 200 and recovered[0]["timestamp"] == 12.5
        assert all(not name.startswith("/") and ".." not in name.split("/") for name in archive.namelist())
        assert not any("original_999" in name or "model.pt" in name or "secrets" in name for name in archive.namelist())
    assert json.loads(output.with_name("sample_manifest.json").read_text()) == manifest
    assert result["frames"] == 3
    assert result["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    repeated = package_bundle(manifest, repo_root=repo, data_root=data,
                              calibration_root=calibration, output=tmp_path / "bundle_again.zip")
    assert result["sha256"] == repeated["sha256"]


def test_bundle_fails_when_harness_missing_and_rejects_path_escape(tmp_path):
    repo, data, calibration, manifest = project_fixture(tmp_path)
    harness = repo / "scripts/part2/benchmark_models.py"
    harness.unlink()
    with pytest.raises(FileNotFoundError):
        package_bundle(manifest, repo_root=repo, data_root=data, calibration_root=calibration,
                       output=tmp_path / "incomplete.zip")
    assert not (tmp_path / "incomplete.zip").exists()
    harness.write_text("# harness\n")
    manifest[0]["rgb"] = "data/../secrets.env"
    with pytest.raises(ValueError, match="within data"):
        package_bundle(manifest, repo_root=repo, data_root=data, calibration_root=calibration,
                       output=tmp_path / "escaped.zip")
    assert not (tmp_path / "escaped.zip").exists()
