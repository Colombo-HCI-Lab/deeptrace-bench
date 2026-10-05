"""AntiDeepfake loads its converted fairseq checkpoints offline and scores windows."""

from __future__ import annotations

import numpy as np
import pytest

from deeptrace_bench.models.base import load_detector

from .test_df_arena import _ready


@pytest.mark.parametrize("model_id", ["antideepfake_w2v_small", "antideepfake_mms_300m"])
def test_antideepfake_scores_windows_offline(registry, model_id, monkeypatch):
    if not _ready(registry, model_id):
        pytest.skip(f"run scripts/setup_models.py {model_id} first")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    detector = load_detector(registry.model(model_id))
    detector.convert_checkpoint()
    detector.load("cpu")  # strict: every converted key has a home
    rng = np.random.default_rng(0)
    scores = detector.score(rng.normal(0, 0.1, (2, 64_600)).astype(np.float32))
    assert scores.shape == (2,)
    assert np.all((scores >= 0) & (scores <= 1))
