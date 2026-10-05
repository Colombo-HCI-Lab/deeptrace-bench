"""HAVIC loads its fine-tuned weights strictly and scores windows (needs its weights)."""

from __future__ import annotations

import numpy as np
import pytest

from deeptrace_bench.models.base import load_detector
from deeptrace_bench.preprocess import PreprocessError

from .test_df_arena import _ready


def test_havic_scores_audio_visual_windows(registry):
    if not _ready(registry, "havic"):
        pytest.skip("run scripts/setup_models.py havic first")
    detector = load_detector(registry.model("havic"))
    detector.convert_checkpoint()
    detector.load("cpu")
    rng = np.random.default_rng(0)
    inputs = {
        "faces": [rng.integers(0, 255, (120, 100, 3), dtype=np.uint8) for _ in range(19)],
        "audio": rng.normal(0, 0.1, 64_000).astype(np.float32),  # 4 s
        "sample_rate": 16_000,
        "n_sampled": 20,
    }
    scores = detector.score(inputs)
    assert scores.shape == (2,)  # 19 frames trimmed to 18: windows at 0 and 2
    assert np.all((scores >= 0) & (scores <= 1))
    with pytest.raises(PreprocessError, match="too_short"):
        detector.score({**inputs, "faces": inputs["faces"][:15]})
