"""LipFD's crops keep upstream's indexing, and UnivFD, LipFD and AVAD load strictly and score
(the scoring tests need their weights)."""

from __future__ import annotations

import numpy as np
import pytest

from deeptrace_bench.models.base import load_detector
from deeptrace_bench.models.lipfd import network_inputs

from .test_df_arena import _ready


def test_lipfd_crops_slide_one_pixel_over_the_first_frame():
    import torch

    images = torch.arange(1000 * 2500, dtype=torch.float32).reshape(1, 1, 1000, 2500)
    images = images.repeat(2, 3, 1, 1)
    whole, crops = network_inputs(images)
    assert whole.shape == (2, 3, 1120, 1120)
    assert [len(scale) for scale in crops] == [5, 5, 5]
    assert crops[2][4].shape == (2, 3, 224, 224)
    # upstream's img[:, 500:, i:i + 500]: each crop starts one column later, not 500
    assert 0 < float((crops[0][1] - crops[0][0]).mean()) < 2


def test_univfd_scores_each_crop(registry):
    if not _ready(registry, "univfd"):
        pytest.skip("run scripts/setup_models.py univfd first")
    detector = load_detector(registry.model("univfd"))
    detector.load("cpu")
    crops = np.random.default_rng(0).integers(0, 255, (3, 224, 224, 3), dtype=np.uint8)
    scores = detector.score(crops)
    assert scores.shape == (3,) and ((scores >= 0) & (scores <= 1)).all()


def test_lipfd_scores_each_image(registry):
    if not _ready(registry, "lipfd"):
        pytest.skip("run scripts/setup_models.py lipfd first")
    detector = load_detector(registry.model("lipfd"))
    detector.load("cpu")
    images = np.random.default_rng(0).integers(0, 255, (2, 1000, 2500, 3), dtype=np.uint8)
    scores = detector.score(images)
    assert scores.shape == (2,) and ((scores >= 0) & (scores <= 1)).all()


def test_avad_gives_one_score_and_needs_two_windows(registry):
    from deeptrace_bench.preprocess import PreprocessError

    if not _ready(registry, "avad"):
        pytest.skip("run scripts/setup_models.py avad first")
    detector = load_detector(registry.model("avad"))
    detector.load("cpu")
    rng = np.random.default_rng(0)
    inputs = {
        "faces": rng.integers(0, 255, (12, 224, 224, 3), dtype=np.uint8),
        "audio": (rng.standard_normal(16000) * 0.1).astype(np.float32),
        "sample_rate": 16000,
    }
    scores = detector.score(inputs)
    assert scores.shape == (1,) and 0 <= scores[0] <= 1
    with pytest.raises(PreprocessError, match="too_short"):
        detector.score({**inputs, "faces": inputs["faces"][:5]})
