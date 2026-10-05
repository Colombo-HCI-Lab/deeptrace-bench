"""SBI builds without fetching ImageNet weights and scores 380 px crops (needs its weights)."""

from __future__ import annotations

import numpy as np

from deeptrace_bench.models.base import load_detector

from .test_df_arena import _ready


def test_sbi_scores_crops(registry, monkeypatch):
    import pytest

    if not _ready(registry, "sbi"):
        pytest.skip("run scripts/setup_models.py sbi first")
    # efficientnet_pytorch fetches ImageNet weights by URL; any download fails the test
    monkeypatch.setattr("torch.utils.model_zoo.load_url", _no_download)
    detector = load_detector(registry.model("sbi"))
    detector.convert_checkpoint()
    detector.load("cpu")
    crops = np.random.default_rng(0).integers(0, 255, (2, 380, 380, 3), dtype=np.uint8)
    scores = detector.score(list(crops))
    assert scores.shape == (2,)
    assert np.all((scores >= 0) & (scores <= 1))


def _no_download(*args, **kwargs):
    raise AssertionError("SBI tried to download ImageNet weights")
