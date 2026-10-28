"""RealForensics cuts clips as pytorchvideo's backpadded uniform sampler does, and the adapter
loads strictly and scores 25-frame clips of face patches (the second test needs its weights)."""

from __future__ import annotations

import numpy as np
import pytest

from deeptrace_bench.models.base import load_detector
from deeptrace_bench.models.realforensics import clip_starts
from deeptrace_bench.preprocess import PreprocessError

from .test_df_arena import _ready


def test_clip_starts_backpad_the_last_clip():
    assert clip_starts(100, 25) == [0, 25, 50, 75]
    assert clip_starts(60, 25) == [0, 25, 35]
    assert clip_starts(25, 25) == [0]
    assert clip_starts(24, 25) == []


def test_realforensics_scores_clips(registry):
    if not _ready(registry, "realforensics"):
        pytest.skip("run scripts/setup_models.py realforensics first")
    detector = load_detector(registry.model("realforensics"))
    detector.convert_checkpoint()
    detector.load("cpu")
    faces = np.random.default_rng(0).integers(0, 255, (60, 150, 150, 3), dtype=np.uint8)
    scores = detector.score(faces)
    assert scores.shape == (3,)
    assert np.all((scores >= 0) & (scores <= 1))
    with pytest.raises(PreprocessError, match="too_short"):
        detector.score(faces[:24])
