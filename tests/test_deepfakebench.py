"""The DeepfakeBench adapter imports one detector cleanly and scores crops (needs the checkout)."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from deeptrace_bench.models.base import load_detector
from deeptrace_bench.paths import RootNotConfiguredError, weights_dir
from deeptrace_bench.upstream import checkout_dir

DETECTORS = ["xception", "efficientnet_b4", "ucf", "f3net", "spsl"]
GENERIC_NAMES = ["metrics", "networks", "detectors", "loss", "torch.utils.tensorboard"]


def _checkout(registry):
    path = checkout_dir(registry.model("xception").upstream)
    if not (path / "training" / "detectors").exists():
        pytest.skip("run scripts/setup_models.py xception first")


def _weights_ready(registry, model_id: str) -> bool:
    try:
        return (weights_dir(model_id) / registry.model(model_id).weights[0].name).exists()
    except RootNotConfiguredError:
        return False


@pytest.mark.parametrize("model_id", DETECTORS)
def test_building_a_detector_leaves_no_generic_modules_behind(registry, model_id):
    _checkout(registry)
    network = load_detector(registry.model(model_id))._build()
    assert sum(p.numel() for p in network.parameters()) > 1_000_000
    assert not [name for name in GENERIC_NAMES if name in sys.modules]


@pytest.mark.parametrize("model_id", DETECTORS)
def test_a_detector_loads_strictly_and_scores_crops(registry, model_id):
    _checkout(registry)
    if not _weights_ready(registry, model_id):
        pytest.skip(f"run scripts/setup_models.py {model_id} first")
    detector = load_detector(registry.model(model_id))
    detector.convert_checkpoint()
    detector.load("cpu")
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 255, (s, s, 3), dtype=np.uint8) for s in (256, 300, 256)]
    scores = detector.score(crops)
    assert scores.shape == (3,)
    assert np.all((scores >= 0) & (scores <= 1))
