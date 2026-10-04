"""GenD loads from the pinned checkout and weights and scores crops of any size.

Skipped unless the weights are in the store (``setup_models.py gend``): CI has no store.
"""

from __future__ import annotations

import numpy as np
import pytest

from deeptrace_bench.models.base import load_detector
from deeptrace_bench.paths import RootNotConfiguredError, weights_dir
from deeptrace_bench.upstream import checkout_dir


def _ready(registry) -> bool:
    model = registry.model("gend")
    try:
        have_weights = (weights_dir("gend") / "model.safetensors").exists()
    except RootNotConfiguredError:
        return False
    return have_weights and checkout_dir(model.upstream).exists()


def test_gend_scores_crops_of_different_sizes(registry):
    if not _ready(registry):
        pytest.skip("GenD weights or checkout not set up")
    detector = load_detector(registry.model("gend"))
    detector.load("cpu")
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 255, (s, s, 3), dtype=np.uint8) for s in (180, 300)]
    scores = detector.score(crops)
    assert scores.shape == (2,)
    assert ((scores >= 0) & (scores <= 1)).all()
