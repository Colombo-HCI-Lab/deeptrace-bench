"""FakeSTormer's runs follow upstream's non-overlapping cut of the evenly spaced crops, its
test-time warp cuts a 15 px margin, and the adapter loads strictly and scores runs (the warp
test needs the pinned checkout, the last its weights)."""

from __future__ import annotations

import numpy as np
import pytest

from deeptrace_bench.models._snt import InputWarp
from deeptrace_bench.models.base import load_detector
from deeptrace_bench.preprocess import PreprocessError
from deeptrace_bench.preprocess.frame_groups import group_crops
from deeptrace_bench.upstream import checkout_dir

from .test_df_arena import _ready


def _crops(n: int) -> list[np.ndarray]:
    return [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(n)]


def test_runs_do_not_overlap_and_drop_the_remainder():
    runs = group_crops(_crops(10), 4)
    assert runs.shape == (2, 4, 8, 8, 3)
    assert runs[:, :, 0, 0, 0].tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert group_crops(_crops(32), 4).shape[0] == 8  # upstream's 32 crops make 8 runs


def test_too_few_crops_for_one_run_is_too_short():
    with pytest.raises(PreprocessError, match="too_short"):
        group_crops(_crops(3), 4)


def test_warp_cuts_the_margin(registry):
    root = checkout_dir(registry.model("fakestormer").upstream)
    if not (root / "package_utils" / "transform.py").is_file():
        pytest.skip("run scripts/setup_models.py fakestormer first")
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    warp = InputWarp(root, "fakestormer_test", size=224, margin=15, mean=mean, std=std)
    crop = np.full((256, 256, 3), 255, dtype=np.uint8)
    crop[15:-15, 15:-15] = 0  # a bright 15 px frame round a black face
    out = warp(crop)
    assert out.shape == (3, 224, 224)
    assert np.allclose(out[:, 100, 100], [-m / s for m, s in zip(mean, std, strict=True)])
    assert np.allclose(out[0], out[0, 0, 0])  # nothing of the frame is left


def test_fakestormer_scores_runs(registry):
    if not _ready(registry, "fakestormer"):
        pytest.skip("run scripts/setup_models.py fakestormer first")
    detector = load_detector(registry.model("fakestormer"))
    detector.convert_checkpoint()
    detector.load("cpu")
    runs = np.random.default_rng(0).integers(0, 255, (2, 4, 256, 256, 3), dtype=np.uint8)
    scores = detector.score(runs)
    assert scores.shape == (2,)
    assert np.all((scores > 0) & (scores < 1))
    with pytest.raises(ValueError, match="runs of 4"):
        detector.score(runs[:, :3])
