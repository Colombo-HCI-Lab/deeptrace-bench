"""Downloads are staged and checked before they reach their final path."""

from __future__ import annotations

import zipfile

import pytest
from pydantic import ValidationError

from deeptrace_bench import fetch
from deeptrace_bench.fetch import ChecksumMismatchError, ManualStepRequiredError, fetch_weight
from deeptrace_bench.registry import Access, WeightSpec


@pytest.fixture
def lock(tmp_path, monkeypatch):
    path = tmp_path / "weights.lock.yaml"
    monkeypatch.setattr(fetch, "LOCK_PATH", path)
    return path


def _serve(monkeypatch, payloads: dict[str, bytes]):
    """Make _download_weight write fixed bytes (a file, or a folder of files)."""

    def fake(spec, dest):
        if spec.kind == "gdrive_folder":
            dest.mkdir(parents=True)
            for name, data in payloads.items():
                (dest / name).write_bytes(data)
        else:
            dest.write_bytes(payloads[spec.name])

    monkeypatch.setattr(fetch, "_download_weight", fake)


def _spec(name="w.pth", kind="url"):
    if kind == "gdrive_folder":
        return WeightSpec(name=name, kind=kind, drive_id="x")
    return WeightSpec(name=name, kind=kind, url="https://example.invalid/w")


def test_html_page_never_reaches_the_final_path(tmp_path, lock, monkeypatch):
    _serve(monkeypatch, {"w.pth": b"  <!DOCTYPE html><html>quota exceeded</html>"})
    with pytest.raises(ManualStepRequiredError, match="HTML"):
        fetch_weight("m", _spec(), tmp_path / "weights", record=True)
    assert not (tmp_path / "weights" / "w.pth").exists()
    assert not lock.exists()


def test_mismatch_never_reaches_the_final_path(tmp_path, lock, monkeypatch):
    fetch.record_hash("m", "w.pth", "0" * 64)
    _serve(monkeypatch, {"w.pth": b"weights"})
    with pytest.raises(ChecksumMismatchError):
        fetch_weight("m", _spec(), tmp_path / "weights")
    assert not (tmp_path / "weights" / "w.pth").exists()


def test_good_download_is_recorded_then_rechecked(tmp_path, lock, monkeypatch):
    _serve(monkeypatch, {"w.pth": b"weights"})
    path = fetch_weight("m", _spec(), tmp_path / "weights", record=True)
    assert path.read_bytes() == b"weights"
    assert fetch.read_lock()["m"]["w.pth"] == fetch.sha256_file(path)
    path.write_bytes(b"tampered")
    with pytest.raises(ChecksumMismatchError):
        fetch_weight("m", _spec(), tmp_path / "weights")


def test_folder_weights_are_hashed_per_file(tmp_path, lock, monkeypatch):
    _serve(monkeypatch, {"a.pth": b"a", "b.pth": b"b"})
    fetch_weight("m", _spec("pretrained", "gdrive_folder"), tmp_path / "weights", record=True)
    assert set(fetch.read_lock()["m"]) == {"pretrained/a.pth", "pretrained/b.pth"}


@pytest.mark.parametrize("name", ["../escape.pth", "/abs/w.pth", "a/../../w.pth"])
def test_unsafe_weight_names_are_rejected(name):
    with pytest.raises(ValidationError, match="plain relative path"):
        WeightSpec(name=name, kind="url", url="https://example.invalid/w")


def test_unsafe_archive_names_are_rejected():
    with pytest.raises(ValidationError, match="plain relative path"):
        Access(kind="url", urls={"../x.zip": "https://example.invalid/x"})


def test_zip_members_cannot_escape(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../../escaped.txt", "x")
    dest = tmp_path / "dataset"
    dest.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        fetch._extract(archive, dest)
    assert not (tmp_path / "escaped.txt").exists()
