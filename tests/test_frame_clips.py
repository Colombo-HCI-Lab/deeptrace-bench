"""Whole-frame clips for the generated-video detectors (DeMamba, D3) and GenVideo-Val labels."""

from __future__ import annotations

import numpy as np
import pytest

from deeptrace_bench.preprocess.frame_clips import demamba_indices, rate_indices, trim_and_resize


def test_demamba_reads_the_centre_eight_of_its_two_fps_frames():
    # 6 s at 24 fps: every 12th frame, 12 kept, the centre 8 of them
    assert demamba_indices(144, 24.0) == list(range(24, 120, 12))
    # 1 s: ceil(8 / 1) = 8 fps, so every frame of an 8 fps clip
    assert demamba_indices(8, 8.0) == list(range(8))
    # shorter than a clip: what there is, padded later by the adapter
    assert demamba_indices(4, 8.0) == [0, 1, 2, 3]


def test_d3_reads_eight_fps_from_the_first_three_seconds():
    assert rate_indices(144, 24.0, 8, 3, 16) == list(range(0, 48, 3))
    # a 1 s, 30 fps clip has only 8 such frames, each the nearest one
    assert rate_indices(30, 30.0, 8, 3, 16) == [0, 4, 8, 11, 15, 19, 22, 26]


def test_trim_cuts_the_long_sides_then_squares():
    frame = np.zeros((704, 1280, 3), np.uint8)
    frame[:, 128:-128] = 255  # only the trimmed-away borders stay black
    out = trim_and_resize(frame)
    assert out.shape == (224, 224, 3)
    assert out.min() == 255


@pytest.mark.parametrize(
    ("path", "label"),
    [
        ("GenVideo-Val/Real/msrvtt_1.mp4", "real"),
        ("GenVideo-Val/Fake/Sora/a.mov", "fake"),
        ("GenVideo-Val/Fake/HotShot/b.gif", "fake"),
        ("__MACOSX/GenVideo-Val/Real/._msrvtt_1.mp4", None),
        ("GenVideo-Val/Fake/Crafter/notes.ini", None),
    ],
)
def test_genvideo_val_labels(path, label):
    from deeptrace_bench.datasets import genvideo_val

    assert genvideo_val.label_from_path(path) == label
