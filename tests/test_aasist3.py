"""AASIST3: pre-emphasis as upstream applies it, and an offline load (needs its weights)."""

from __future__ import annotations

import numpy as np
import pytest

from deeptrace_bench.models.aasist3 import preemphasis
from deeptrace_bench.models.base import load_detector

from .test_df_arena import _ready


def test_preemphasis_subtracts_the_scaled_previous_sample():
    x = np.array([[1.0, 2.0, 4.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(preemphasis(x), [[1.0, 1.03, 2.06], [0.0, 1.0, -0.97]], rtol=1e-6)
    assert x[0, 1] == 2.0  # the input is left alone


def test_aasist3_scores_windows_offline(registry, monkeypatch):
    if not _ready(registry, "aasist3"):
        pytest.skip("run scripts/setup_models.py aasist3 first")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    detector = load_detector(registry.model("aasist3"))
    detector.convert_checkpoint()
    detector.load("cpu")
    rng = np.random.default_rng(0)
    scores = detector.score(rng.normal(0, 0.1, (2, 64_600)).astype(np.float32))
    assert scores.shape == (2,)
    assert np.all((scores >= 0) & (scores <= 1))
