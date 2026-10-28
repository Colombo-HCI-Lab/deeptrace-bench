"""The DFDC winner's letterbox crop matches upstream's resize-and-centre, and the seven-model
ensemble loads strictly and scores crops (the second test needs its weights)."""

from __future__ import annotations

import numpy as np
import pytest

from deeptrace_bench.models.base import load_detector
from deeptrace_bench.preprocess.faces import (
    isotropically_resize_image,
    letterbox_crop,
    put_to_center,
)

from .test_df_arena import _ready


def test_letterbox_keeps_the_aspect_and_centres():
    image = np.full((300, 400, 3), 200, dtype=np.uint8)
    box = np.array([150.0, 100.0, 210.0, 190.0])  # 60 x 90 face
    crop = letterbox_crop(image, box, 380, 5 / 3)
    assert crop.shape == (380, 380, 3)
    assert crop[190, 0].sum() == 0 and crop[190, 190].sum() > 0  # black bars left and right
    tall = isotropically_resize_image(np.zeros((90, 60, 3), np.uint8), 380)
    assert tall.shape[:2] == (380, 253)
    assert put_to_center(tall, 380).shape == (380, 380, 3)


def test_dfdc_b7_scores_crops(registry):
    if not _ready(registry, "dfdc_b7"):
        pytest.skip("run scripts/setup_models.py dfdc_b7 first")
    detector = load_detector(registry.model("dfdc_b7"))
    detector.convert_checkpoint()
    detector.load("cpu")
    assert len(detector.models) == 7
    crops = np.random.default_rng(0).integers(0, 255, (1, 380, 380, 3), dtype=np.uint8)
    scores = detector.score(crops)
    assert scores.shape == (1,)
    assert 0 <= scores[0] <= 1
