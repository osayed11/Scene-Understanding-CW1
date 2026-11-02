"""The sharing bundle stays narrowly scoped and removes local manifest paths."""
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from part2.prepare_gpu_smoke import MAX_FRAMES, SCRIPT_FILES, SEQUENCE_CONFIG, build_bundle


def fixture_project(tmp_path):
    repo = tmp_path / "project"
    for relative in (*SCRIPT_FILES, SEQUENCE_CONFIG):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# allowed fixture content\n")
    data = tmp_path / "data"
    sequence = data / "replica/room0"
    results = sequence / "results"
    results.mkdir(parents=True)
    entries = {}
    for index in range(4):
        rgb = f"rgb fixture {index}".encode()
        (results / f"frame{index:06d}.jpg").write_bytes(rgb)
        (results / f"depth{index:06d}.png").write_bytes(f"depth fixture {index}".encode())
        entries[f"frame{index:06d}"] = {
            "index": index, "timestamp": index / 25,
            "rgb": "/private/user/dataset/file.jpg", "private_token": "do-not-share",
            "rgb_sha256": hashlib.sha256(rgb).hexdigest(), "inference_seconds": 70 + index,
        }
    (sequence / "traj.txt").write_text("\n".join([" ".join([str(index)] * 16) for index in range(4)]) + "\n")
    calibration = tmp_path / "REPLICA.yaml"
    calibration.write_text("Camera.fx: 600\n")
    manifest = repo / "depth_maps/replica_room0/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"sequence": "replica_room0", "signature": {
        "model": "depth_pro", "device": "cpu", "checkpoint_sha256": "a" * 64,
        "precision": "float32", "input_color": "RGB", "depth_factor": 6553.5,
        "focal_px": 600, "input_size": [680, 1200], "rgb_source_root": "/private/user/dataset",
        "private_token": "do-not-share",
    }, "frames": entries}))
    (repo / "secrets.env").write_text("do-not-share")
    (repo / "weights").mkdir()
    (repo / "weights/model.pt").write_text("do-not-share")
    return repo, data, calibration, manifest


def test_bundle_exact_allowlist_and_sanitized_reference(tmp_path):
    repo, data, calibration, manifest = fixture_project(tmp_path)
    output = tmp_path / "inputs.zip"
    result = build_bundle(repo_root=repo, data_root=data, calibration=calibration,
                          cpu_manifest=manifest, output=output)
    expected = {f"cw1_gpu_smoke/{name}" for name in (*SCRIPT_FILES, SEQUENCE_CONFIG,
        "calibration/settings/REPLICA.yaml", "data/replica/room0/traj.txt", "cpu_reference.json")}
    expected |= {f"cw1_gpu_smoke/data/replica/room0/results/{kind}{i:06d}.{ext}"
                 for i in range(3) for kind, ext in (("frame", "jpg"), ("depth", "png"))}
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == expected
        assert len(archive.read("cw1_gpu_smoke/data/replica/room0/traj.txt").splitlines()) == 3
        text = archive.read("cw1_gpu_smoke/cpu_reference.json").decode()
        assert "/private/" not in text and "do-not-share" not in text
        reference = json.loads(text)
        assert reference["mean_inference_seconds"] == 71
        assert reference["reference_frame_count"] == reference["bundle_frame_count"] == 3
        assert [row["index"] for row in reference["frames"]] == [0, 1, 2]
        archive.extractall(tmp_path / "extracted")
    assert (tmp_path / "extracted/cw1_gpu_smoke/scripts/part2/predict_depth.py").is_file()
    assert result["bytes"] == output.stat().st_size


def test_bundle_rejects_large_requests_and_mismatched_cpu_inputs(tmp_path):
    repo, data, calibration, manifest = fixture_project(tmp_path)
    with pytest.raises(ValueError, match="frames must"):
        build_bundle(repo_root=repo, frames=MAX_FRAMES + 1)
    path = data / "replica/room0/results/frame000000.jpg"
    path.write_bytes(b"different image")
    with pytest.raises(ValueError, match="checksum"):
        build_bundle(repo_root=repo, data_root=data, calibration=calibration, cpu_manifest=manifest)
    assert not (repo / "output/jupyter-notebook/cw1_gpu_smoke_inputs.zip").exists()
