"""AASIST and AASIST-L load strictly and score windows (skipped until their weights are set up)."""

from __future__ import annotations

import numpy as np
import pytest

from deeptrace_bench.models.base import load_detector
from deeptrace_bench.paths import RootNotConfiguredError, weights_dir


def _ready(registry, model_id: str) -> bool:
    try:
        source = registry.model(model_id).weights[0].name
        return (weights_dir(model_id) / source).exists()
    except RootNotConfiguredError:
        return False


@pytest.mark.parametrize("model_id", ["aasist", "aasist_l"])
def test_aasist_scores_windows(registry, model_id):
    if not _ready(registry, model_id):
        pytest.skip(f"run scripts/setup_models.py {model_id} first")
    detector = load_detector(registry.model(model_id))
    detector.convert_checkpoint()
    detector.load("cpu")
    rng = np.random.default_rng(0)
    scores = detector.score(rng.normal(0, 0.1, (3, 64_600)).astype(np.float32))
    assert scores.shape == (3,)
    assert np.all((scores >= 0) & (scores <= 1))
