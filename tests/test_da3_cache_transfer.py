from dataclasses import asdict
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from depth import save_depth
from part2.common import Camera, FrameRecord
from part2 import transfer_da3_cache as transfer


@pytest.fixture
def cache(tmp_path, monkeypatch):
    data = tmp_path / "data"
    rgb = data / "tum" / "rgb" / "1341847989.402777.png"
    rgb.parent.mkdir(parents=True)
    rgb.write_bytes(b"fixed input RGB")
    frame = FrameRecord(0, 1341847989.402777, rgb)
    camera = Camera(2., 2.1, .5, .5, 2, 2)
    monkeypatch.setattr(transfer, "find_data_root", lambda _: data)
    monkeypatch.setattr(transfer, "load_frame_records", lambda *_: [frame])
    monkeypatch.setattr(transfer, "load_camera", lambda *_: camera)
    monkeypatch.setattr(transfer, "load_sequence", lambda _: {"depth_factor": 5000})
    source = tmp_path / "cloud_cache"
    folder = source / "tum_fr3"
    folder.mkdir(parents=True)
    depth = np.array([[1., 2.], [20., np.nan]], dtype=np.float32)
    save_depth(folder / frame.stem, depth, 5000, overflow="invalid")
    manifest = {"schema_version": 2, "sequence": "tum_fr3", "status": "complete", "full_count": 1,
                "signature": {"model": "da3_metric_large", "camera": asdict(camera),
                              "depth_factor": 5000, "png_overflow": "invalid"},
                "frames": {frame.stem: {"index": 0, "timestamp": frame.timestamp,
                    "rgb": "data/tum/rgb/" + rgb.name, "rgb_sha256": transfer.digest_file(rgb),
                    "npy_sha256": transfer.digest_file(folder / f"{frame.stem}.npy"),
                    "png_sha256": transfer.digest_file(folder / f"{frame.stem}.png")}}}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    (folder / "associations.txt").write_text("# old Colab paths must be replaced\n")
    return source, frame, data, manifest


def test_lossless_transfer_rebases_associations_and_resumes(cache, tmp_path):
    source, frame, data, manifest = cache
    archive = tmp_path / "tum_results.zip"
    transfer.export_cache("tum_fr3", source, archive)
    target = tmp_path / "local_cache"
    result = transfer.import_cache(archive, target, data)
    assert result["full_sequence"] and result["imported_frames"] == 1
    folder = target / "tum_fr3"
    for suffix in ("npy", "png"):
        assert (folder / f"{frame.stem}.{suffix}").read_bytes() == (source / "tum_fr3" / f"{frame.stem}.{suffix}").read_bytes()
    tokens = (folder / "associations.txt").read_text().splitlines()[1].split()
    assert (folder / tokens[1]).resolve() == frame.rgb.resolve()
    assert (folder / tokens[3]).resolve() == (folder / f"{frame.stem}.png").resolve()
    assert json.loads((folder / "manifest.json").read_text()) == manifest
    assert transfer.import_cache(archive, target, data)["imported_frames"] == 1


def test_export_rejects_changed_depth(cache, tmp_path):
    source, frame, *_ = cache
    (source / "tum_fr3" / f"{frame.stem}.npy").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="changed cached"):
        transfer.export_cache("tum_fr3", source, tmp_path / "output.zip")


def test_export_uses_committed_manifest_snapshot(cache, tmp_path, monkeypatch):
    source, _, _, manifest = cache
    original_digest = transfer.digest_file
    path = source / "tum_fr3/manifest.json"

    def concurrently_updated_manifest(value):
        if Path(value).suffix == ".npy":
            path.write_text(json.dumps({**manifest, "frames": {}}))
        return original_digest(value)

    monkeypatch.setattr(transfer, "digest_file", concurrently_updated_manifest)
    archive = tmp_path / "snapshot.zip"
    transfer.export_cache("tum_fr3", source, archive)
    with zipfile.ZipFile(archive) as handle:
        assert json.loads(handle.read("tum_fr3/manifest.json")) == manifest


def test_import_checks_expected_archive_digest(cache, tmp_path):
    source, _, data, _ = cache
    archive = tmp_path / "result.zip"
    transfer.export_cache("tum_fr3", source, archive)
    with pytest.raises(ValueError, match="archive SHA256"):
        transfer.import_cache(archive, tmp_path / "local", data, "0" * 64)


def test_partial_import_can_extend_to_complete_cache(cache, tmp_path, monkeypatch):
    source, first, data, manifest = cache
    second_rgb = first.rgb.with_name("1341847989.442777.png")
    second_rgb.write_bytes(b"second RGB frame")
    second = FrameRecord(1, first.timestamp + .04, second_rgb)
    monkeypatch.setattr(transfer, "load_frame_records", lambda *_: [first, second])
    folder = source / "tum_fr3"
    manifest["full_count"] = 2
    (folder / "manifest.json").write_text(json.dumps(manifest))
    partial = tmp_path / "partial.zip"
    transfer.export_cache("tum_fr3", source, partial)
    target = tmp_path / "local"
    assert not transfer.import_cache(partial, target, data)["full_sequence"]
    save_depth(folder / second.stem, np.full((2, 2), 2., dtype=np.float32), 5000, overflow="invalid")
    manifest["frames"][second.stem] = {"index": 1, "timestamp": second.timestamp,
        "rgb": "data/tum/rgb/" + second_rgb.name, "rgb_sha256": transfer.digest_file(second_rgb),
        "npy_sha256": transfer.digest_file(folder / f"{second.stem}.npy"),
        "png_sha256": transfer.digest_file(folder / f"{second.stem}.png")}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    full = tmp_path / "full.zip"
    transfer.export_cache("tum_fr3", source, full)
    result = transfer.import_cache(full, target, data)
    assert result["full_sequence"] and result["imported_frames"] == 2
    assert len((target / "tum_fr3/associations.txt").read_text().splitlines()) == 3


def test_import_rejects_changed_rgb_before_target_publication(cache, tmp_path):
    source, frame, data, _ = cache
    archive = tmp_path / "result.zip"
    transfer.export_cache("tum_fr3", source, archive)
    frame.rgb.write_bytes(b"changed input")
    target = tmp_path / "local"
    with pytest.raises(ValueError, match="RGB content"):
        transfer.import_cache(archive, target, data)
    assert not (target / "tum_fr3").exists()


@pytest.mark.parametrize("malicious", ["../escape.txt", "tum_fr3/../../escape.txt", "C:/escape.txt"])
def test_import_rejects_archive_traversal(cache, tmp_path, malicious):
    source, _, data, _ = cache
    archive = tmp_path / "result.zip"
    transfer.export_cache("tum_fr3", source, archive)
    with zipfile.ZipFile(archive, "a") as handle:
        handle.writestr(malicious, b"bad")
    with pytest.raises(ValueError, match="Unsafe archive"):
        transfer.import_cache(archive, tmp_path / "local", data)


def test_import_rejects_tampered_archive_data(cache, tmp_path):
    source, frame, data, _ = cache
    archive = tmp_path / "result.zip"
    transfer.export_cache("tum_fr3", source, archive)
    bad = tmp_path / "tampered.zip"
    with zipfile.ZipFile(archive) as original, zipfile.ZipFile(bad, "w") as changed:
        for name in original.namelist():
            payload = original.read(name)
            if name.endswith(frame.stem + ".png"):
                payload = payload[:-1] + bytes([payload[-1] ^ 1])
            changed.writestr(name, payload)
    target = tmp_path / "local"
    with pytest.raises(ValueError, match="checksum"):
        transfer.import_cache(bad, target, data)
    assert not (target / "tum_fr3").exists()
