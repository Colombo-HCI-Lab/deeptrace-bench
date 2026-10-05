"""Converting a downloaded checkpoint to safetensors once, and refusing a stale conversion."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deeptrace_bench import fetch
from deeptrace_bench.models.base import Detector
from deeptrace_bench.registry import ModelConfig


class _Tiny(Detector):
    def load(self, device: str) -> None:
        self.model = torch.nn.Linear(3, 2)
        self.load_converted(self.model, source="ckpt.pth")

    def score(self, inputs):
        return np.zeros(len(inputs))


@pytest.fixture
def detector(tmp_path, monkeypatch):
    monkeypatch.setenv("DTB_ROOT", str(tmp_path))
    monkeypatch.delenv("DTB_NAMESPACE", raising=False)
    monkeypatch.setattr(fetch, "LOCK_PATH", tmp_path / "weights.lock.yaml")
    config = ModelConfig(id="tiny", name="tiny", modality="audio", status="test", licence="-")
    return _Tiny(config)


def _download(detector, state) -> None:
    detector.weights_dir.mkdir(parents=True, exist_ok=True)
    path = detector.weights_dir / "ckpt.pth"
    torch.save(state, path)
    fetch.record_hash("tiny", "ckpt.pth", fetch.sha256_file(path))


def test_conversion_strips_wrappers_and_loads_strictly(detector):
    source = torch.nn.Linear(3, 2)
    _download(detector, {"state_dict": {f"module.{k}": v for k, v in source.state_dict().items()}})
    state = torch.load(detector.weights_dir / "ckpt.pth", weights_only=True)
    detector.save_converted(state, "ckpt.pth")

    assert detector.converted_is_current("ckpt.pth")
    detector.load("cpu")
    assert torch.equal(detector.model.weight, source.weight)


def test_a_conversion_from_another_download_is_refused(detector):
    _download(detector, torch.nn.Linear(3, 2).state_dict())
    detector.save_converted(
        torch.load(detector.weights_dir / "ckpt.pth", weights_only=True), "ckpt.pth"
    )
    _download(detector, torch.nn.Linear(3, 2).state_dict())  # a new file, a new hash

    assert not detector.converted_is_current("ckpt.pth")
    with pytest.raises(RuntimeError, match="setup_models"):
        detector.load("cpu")


def test_conversion_needs_a_pinned_source(detector):
    detector.weights_dir.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="--record"):
        detector.save_converted({"weight": torch.zeros(2, 3)}, "unpinned.pth")


def test_a_conversion_of_another_file_is_refused(detector):
    # Both downloads are pinned, but the adapter now expects the other one.
    _download(detector, torch.nn.Linear(3, 2).state_dict())
    detector.save_converted(
        torch.load(detector.weights_dir / "ckpt.pth", weights_only=True), "ckpt.pth"
    )
    fetch.record_hash("tiny", "other.pth", "0" * 64)
    with pytest.raises(RuntimeError, match="other.pth"):
        detector.load_converted(torch.nn.Linear(3, 2), source="other.pth")


def test_a_conversion_from_older_conversion_code_is_refused(detector, monkeypatch):
    _download(detector, torch.nn.Linear(3, 2).state_dict())
    detector.save_converted(
        torch.load(detector.weights_dir / "ckpt.pth", weights_only=True), "ckpt.pth"
    )
    monkeypatch.setattr(type(detector), "conversion_version", 99)
    assert not detector.converted_is_current("ckpt.pth")
    with pytest.raises(RuntimeError, match="setup_models"):
        detector.load_converted(torch.nn.Linear(3, 2), source="ckpt.pth")


def test_a_download_named_like_the_conversion_is_never_overwritten(detector):
    from deeptrace_bench.models.base import CONVERTED

    detector.weights_dir.mkdir(parents=True, exist_ok=True)
    download = detector.weights_dir / CONVERTED
    download.write_bytes(b"the original download")
    fetch.record_hash("tiny", CONVERTED, fetch.sha256_file(download))
    with pytest.raises(ValueError, match="rename"):
        detector.save_converted(torch.nn.Linear(3, 2).state_dict(), CONVERTED)
    assert download.read_bytes() == b"the original download"
