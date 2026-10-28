"""AVH-Align's audio features follow upstream's load_audio, and the rebuilt AV-HuBERT encoder and
the fusion network load strictly and give one score per item (the second test needs weights)."""

from __future__ import annotations

import numpy as np
import pytest

from deeptrace_bench.models.avh_align import audio_features
from deeptrace_bench.models.base import load_detector

from .test_df_arena import _ready


def test_audio_features_stack_four_filterbank_frames():
    wave = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
    feats = audio_features(wave, 16000)
    assert feats.shape == (25, 104)  # 99 filterbank frames padded to 100, stacked by four
    assert np.allclose(feats.mean(axis=1), 0, atol=1e-5)  # layer-normalised per stacked frame
    assert np.allclose(feats.std(axis=1), 1, atol=1e-2)


def test_avh_align_scores_one_value_per_item(registry):
    if not _ready(registry, "avh_align"):
        pytest.skip("run scripts/setup_models.py avh_align first")
    detector = load_detector(registry.model("avh_align"))
    detector.convert_checkpoint()
    detector.load("cpu")
    rng = np.random.default_rng(0)
    inputs = {
        "mouths": rng.integers(0, 255, (30, 96, 96), dtype=np.uint8),
        "audio": rng.standard_normal(16000 * 2).astype(np.float32) * 0.05,
        "sample_rate": 16000,
    }
    scores = detector.score(inputs)
    assert scores.shape == (1,)
    assert np.isfinite(scores[0])
