"""Mouth crops: upstream's warp-and-crop loop over consecutive frames (needs the checkout)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from deeptrace_bench.preprocess import PreprocessError
from deeptrace_bench.preprocess.mouths import MouthLoader
from deeptrace_bench.upstream import checkout_dir


def _loader(registry, records, frames, min_frames=25) -> MouthLoader:
    upstream = checkout_dir(registry.model("lipforensics").upstream)
    if not (upstream / "preprocessing" / "utils.py").exists():
        pytest.skip("run scripts/setup_models.py lipforensics first")
    faces = SimpleNamespace(consecutive=True, detections=lambda row: (frames, records))
    loader = MouthLoader(faces, fan_path=None, upstream_dir=upstream, min_frames=min_frames)
    loader._setup = lambda: None  # no FAN: landmarks are stubbed below
    from deeptrace_bench.models._upstream import load_module

    prep = upstream / "preprocessing"
    loader._utils = load_module(prep / "utils.py", "dtb_upstream_lipforensics_prep")
    loader._mean_face = np.load(prep / "20words_mean_face.npy")
    # the mean face itself, so each frame is warped onto it unchanged
    loader.landmarks = lambda image, box: loader._mean_face.astype(np.float64)
    return loader


def _video(n: int, missing: set[int] = frozenset()):
    rng = np.random.default_rng(0)
    frames = {i: rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for i in range(n)}
    records = [
        {"frame_index": i, "box": None if i in missing else [60.0, 40.0, 200.0, 220.0]}
        for i in range(n)
    ]
    return frames, records


def test_every_frame_gives_a_grayscale_mouth(registry):
    frames, records = _video(30, missing={3, 4})
    out = _loader(registry, records, frames)(pd.Series({"item_id": "toy/v"}))
    assert out.shape == (30, 96, 96) and out.dtype == np.uint8  # frames 3, 4 reuse frame 2's


def test_too_few_faces_is_no_face(registry):
    frames, records = _video(30, missing=set(range(10)))
    with pytest.raises(PreprocessError, match="no_face"):
        _loader(registry, records, frames)(pd.Series({"item_id": "toy/v"}))
