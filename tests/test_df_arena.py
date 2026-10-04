"""DF Arena loads from its pinned checkout without reaching the Hub (skipped without weights)."""

from __future__ import annotations

import numpy as np
import pytest

from deeptrace_bench.models.base import load_detector
from deeptrace_bench.paths import RootNotConfiguredError, weights_dir


def _ready(registry, model_id: str) -> bool:
    try:
        model = registry.model(model_id)
        return all((weights_dir(model_id) / w.name).exists() for w in model.weights)
    except RootNotConfiguredError:
        return False


@pytest.mark.parametrize("model_id", ["df_arena_500m"])
def test_df_arena_scores_windows_offline(registry, model_id, monkeypatch):
    if not _ready(registry, model_id):
        pytest.skip(f"run scripts/setup_models.py {model_id} first")
    # Anything that tries the network fails the test instead of fetching silently.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    detector = load_detector(registry.model(model_id))
    detector.convert_checkpoint()
    detector.load("cpu")
    rng = np.random.default_rng(0)
    scores = detector.score(rng.normal(0, 0.1, (2, 64_600)).astype(np.float32))
    assert scores.shape == (2,)
    assert np.all((scores >= 0) & (scores <= 1))
