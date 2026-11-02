"""LAA-Net's test-time warp follows upstream's crop-by-margin and affine resize, and the adapter
loads strictly and scores face crops (the warp test needs the pinned checkout, the second its
weights)."""

from __future__ import annotations

import numpy as np
import pytest

from deeptrace_bench.models._snt import InputWarp
from deeptrace_bench.models.base import load_detector
from deeptrace_bench.upstream import checkout_dir

from .test_df_arena import _ready


def _warp(registry) -> InputWarp:
    root = checkout_dir(registry.model("laa_net").upstream)
    if not (root / "package_utils" / "transform.py").is_file():
        pytest.skip("run scripts/setup_models.py laa_net first")
    return InputWarp(root, "laa_net_test", size=384, margin=5, mean=[0.5] * 3, std=[0.5] * 3)


def test_warp_cuts_the_margin_and_normalises(registry):
    warp = _warp(registry)
    crop = np.full((256, 256, 3), 255, dtype=np.uint8)
    crop[5:-5, 5:-5] = 0  # a bright 5 px frame round a black face
    out = warp(crop)
    assert out.shape == (3, 384, 384) and out.dtype == np.float64
    # the frame is gone and black normalises to (0 / 255 - 0.5) / 0.5
    assert np.allclose(out, -1.0)
    grey = warp(np.full((256, 256, 3), 51, dtype=np.uint8))
    assert np.allclose(grey, (51 / 255 - 0.5) / 0.5)


def test_warp_keeps_the_picture_upright(registry):
    warp = _warp(registry)
    crop = np.zeros((256, 256, 3), dtype=np.uint8)
    crop[:128] = 200  # bright top half
    out = warp(crop)
    assert out[0, 20, 192] > out[0, 360, 192]


def test_laa_net_scores_crops(registry):
    if not _ready(registry, "laa_net"):
        pytest.skip("run scripts/setup_models.py laa_net first")
    detector = load_detector(registry.model("laa_net"))
    detector.convert_checkpoint()
    detector.load("cpu")
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(2)]
    scores = detector.score(crops)
    assert scores.shape == (2,)
    assert np.all((scores > 0) & (scores < 1))
    # the flip test averages a crop's logit with its mirror's, so mirroring changes nothing
    mirrored = detector.score([c[:, ::-1] for c in crops])
    assert np.allclose(scores, mirrored, atol=1e-3)
    assert detector.score([]).shape == (0,)
