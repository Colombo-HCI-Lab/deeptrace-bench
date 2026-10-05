"""XLS-R + AASIST loads its converted fairseq checkpoint offline and scores windows."""

from __future__ import annotations

import numpy as np
import pytest

from deeptrace_bench.models.base import load_detector

from .test_df_arena import _ready


def test_xlsr_aasist_scores_windows_offline(registry, monkeypatch):
    if not _ready(registry, "xlsr_aasist"):
        pytest.skip("run scripts/setup_models.py xlsr_aasist first")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    detector = load_detector(registry.model("xlsr_aasist"))
    detector.convert_checkpoint()
    detector.load("cpu")  # a strict load: every converted key has a home and none is left out
    rng = np.random.default_rng(0)
    scores = detector.score(rng.normal(0, 0.1, (2, 64_600)).astype(np.float32))
    assert scores.shape == (2,)
    assert np.all((scores >= 0) & (scores <= 1))
