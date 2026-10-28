"""GenConViT's ED branch loads strictly after the Swin key move and scores face crops (needs
its weights)."""

from __future__ import annotations

import numpy as np
import pytest

from deeptrace_bench.models.base import load_detector

from .test_df_arena import _ready


def test_genconvit_scores_crops(registry):
    if not _ready(registry, "genconvit"):
        pytest.skip("run scripts/setup_models.py genconvit first")
    detector = load_detector(registry.model("genconvit"))
    detector.convert_checkpoint()
    detector.load("cpu")
    crops = np.random.default_rng(0).integers(0, 255, (3, 224, 224, 3), dtype=np.uint8)
    scores = detector.score(crops)
    assert scores.shape == (3,)
    assert np.all((scores >= 0) & (scores <= 1))
    # deterministic, unlike upstream's VAE branch
    assert np.array_equal(scores, detector.score(crops))
