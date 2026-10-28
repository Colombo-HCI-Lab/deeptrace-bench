"""FTCN's clip windows follow upstream's padding, and the adapter loads strictly and scores
clips (the second test needs its weights)."""

from __future__ import annotations

import numpy as np
import pytest

from deeptrace_bench.models.base import load_detector
from deeptrace_bench.preprocess import PreprocessError
from deeptrace_bench.preprocess.clips import clip_windows, pad_indices

from .test_df_arena import _ready


def _upstream_padding(super_clip_size: int, clip_size: int) -> list[int]:
    """The padding loop of FTCN's ``test_on_raw_video.py``, as written there."""
    inner_index = list(range(super_clip_size))
    pad_length = clip_size - 1
    post_module = inner_index[1:-1][::-1] + inner_index
    l_post = len(post_module)
    post_module = post_module * (pad_length // l_post + 1)
    post_module = post_module[:pad_length]
    assert len(post_module) == pad_length
    pre_module = inner_index + inner_index[1:-1][::-1]
    l_pre = len(post_module)
    pre_module = pre_module * (pad_length // l_pre + 1)
    pre_module = pre_module[-pad_length:]
    assert len(pre_module) == pad_length
    return pre_module + inner_index + post_module


@pytest.mark.parametrize("length", [9, 12, 16, 20, 31])
def test_padding_matches_upstream(length):
    assert pad_indices(length, 32) == _upstream_padding(length, 32)


def test_short_tracks_are_too_short_as_upstream_asserts():
    with pytest.raises(AssertionError):
        _upstream_padding(8, 32)
    with pytest.raises(PreprocessError, match="too_short"):
        pad_indices(8, 32)


def test_windows_step_by_the_stride():
    windows = clip_windows(96, 32, 16)
    assert [w[0] for w in windows] == [0, 16, 32, 48, 64]
    assert all(len(w) == 32 for w in windows)
    assert clip_windows(32, 32, 16) == [list(range(32))]
    assert len(clip_windows(20, 32, 16)) == 4  # padded to 82 indices: starts 0, 16, 32, 48


def test_ftcn_scores_clips(registry):
    if not _ready(registry, "ftcn"):
        pytest.skip("run scripts/setup_models.py ftcn first")
    detector = load_detector(registry.model("ftcn"))
    detector.convert_checkpoint()
    detector.load("cpu")
    clips = np.random.default_rng(0).integers(0, 255, (2, 32, 224, 224, 3), dtype=np.uint8)
    scores = detector.score(clips)
    assert scores.shape == (2,)
    assert np.all((scores >= 0) & (scores <= 1))
