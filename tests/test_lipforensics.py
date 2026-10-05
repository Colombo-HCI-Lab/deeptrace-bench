"""LipForensics loads strictly and scores 25-frame mouth clips (needs its weights)."""

from __future__ import annotations

import numpy as np
import pytest

from deeptrace_bench.models.base import load_detector
from deeptrace_bench.preprocess import PreprocessError

from .test_df_arena import _ready


def test_lipforensics_scores_mouth_clips(registry):
    if not _ready(registry, "lipforensics"):
        pytest.skip("run scripts/setup_models.py lipforensics first")
    detector = load_detector(registry.model("lipforensics"))
    detector.convert_checkpoint()
    detector.load("cpu")
    mouths = np.random.default_rng(0).integers(0, 255, (60, 96, 96), dtype=np.uint8)
    scores = detector.score(mouths)
    assert scores.shape == (2,)  # 60 frames: two whole clips, the rest left out
    assert np.all((scores >= 0) & (scores <= 1))
    with pytest.raises(PreprocessError, match="too_short"):
        detector.score(mouths[:24])
