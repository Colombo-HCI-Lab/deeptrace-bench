"""SCRFD's decoding and suppression, on hand-made arrays (no model file needed)."""

from __future__ import annotations

import numpy as np

from deeptrace_bench.preprocess.scrfd import distance2bbox, distance2kps, nms


def test_distances_decode_to_boxes():
    centers = np.array([[10.0, 20.0]])
    assert distance2bbox(centers, np.array([[1.0, 2.0, 3.0, 4.0]])).tolist() == [
        [9.0, 18.0, 13.0, 24.0]
    ]


def test_offsets_decode_to_keypoints():
    centers = np.array([[10.0, 20.0]])
    offsets = np.array([[1.0, -1.0, 2.0, -2.0]])
    assert distance2kps(centers, offsets).tolist() == [[11.0, 19.0, 12.0, 18.0]]


def test_overlapping_boxes_keep_the_best():
    dets = np.array(
        [
            [0, 0, 10, 10, 0.9],
            [1, 1, 11, 11, 0.8],  # mostly the same face
            [50, 50, 60, 60, 0.7],  # another face
        ],
        dtype=np.float32,
    )
    assert nms(dets, threshold=0.4) == [0, 2]
