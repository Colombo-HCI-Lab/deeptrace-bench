"""Face tracks: gaps and jitter are smoothed, SyncNet's crop stays inside its padding, and the
two loaders built on them give LipFD and AVAD the shapes they read."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deeptrace_bench.preprocess import PreprocessError, tracks
from deeptrace_bench.preprocess.faces import CropSpec, window_frame_indices


def _record(index: int, box: list[float] | None) -> dict:
    return {"frame_index": index, "box": box, "kps": None}


class _Faces:
    """A stand-in FaceLoader: grey frames with a fixed box in each."""

    def __init__(self, indices: list[int], size: int, windows=None, missing=()) -> None:
        self.crop = CropSpec(size=size)
        self.windows = windows
        self.indices = indices
        self.missing = set(missing)

    def detections(self, row):
        images = {i: np.full((120, 160, 3), i % 255, dtype=np.uint8) for i in self.indices}
        box = [60.0, 30.0, 100.0, 80.0]
        return images, [_record(i, None if i in self.missing else box) for i in self.indices]

    def close(self) -> None:
        pass


ROW = pd.Series({"dataset": "x", "rel_path": "a.mp4", "modality": "video"})


def test_frames_without_a_face_take_the_nearest_box():
    records = [
        _record(0, None),
        _record(1, [0, 0, 2, 2]),
        _record(2, None),
        _record(3, [4, 4, 6, 6]),
    ]
    indices, boxes = tracks.fill_boxes(records)
    assert indices.tolist() == [0, 1, 2, 3]
    assert boxes[:, 0].tolist() == [0, 0, 0, 4]


def test_no_face_at_all_is_a_status():
    with pytest.raises(PreprocessError, match="no_face"):
        tracks.fill_boxes([_record(0, None)])


def test_smoothing_removes_a_one_frame_jump_and_keeps_the_ends():
    boxes = np.tile([10.0, 10.0, 30.0, 50.0], (20, 1))
    boxes[9] += 40
    track = tracks.smooth_track(boxes)
    assert np.allclose(track[:, 0], 20) and np.allclose(track[:, 1], 30)
    assert np.allclose(track[:, 2], 20)  # half the longer side, not shrunk at the edges


def test_track_crop_near_the_edge_keeps_its_size():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    assert tracks.track_crop(frame, 2.0, 2.0, 20.0, 224).shape == (224, 224, 3)


def test_windows_spread_over_the_video_and_never_repeat():
    assert window_frame_indices(5, 10, 5) == []
    indices = window_frame_indices(100, 10, 5)
    assert indices[0] == 0 and indices[-1] == 98 and len(indices) == 50
    short = window_frame_indices(12, 10, 5)  # the windows overlap
    assert short == sorted(set(short)) and short[-1] == 10


def test_mel_image_is_rgb_with_librosas_128_bands():
    wave = np.random.default_rng(0).standard_normal(22050).astype(np.float32)
    image = tracks.mel_image(wave)
    assert image.shape[0] == 128 and image.shape[2] == 3 and image.dtype == np.uint8


def test_lip_composites_stack_frames_under_their_mel(monkeypatch):
    indices = window_frame_indices(100, 10, 5)
    monkeypatch.setattr(tracks, "video_stats", lambda path: (100, 25.0))
    wave = np.random.default_rng(0).standard_normal(22050 * 4).astype(np.float32)
    monkeypatch.setattr(tracks, "load_audio", lambda path, rate: wave)
    monkeypatch.setattr(tracks, "dataset_dir", lambda dataset: Path("x"))
    faces = _Faces(indices, 500, windows=(10, 5), missing=[0])
    images = tracks.LipCompositeLoader(faces, min_faces=5)(ROW)
    assert images.shape == (10, 1000, 2500, 3)
    assert (images[0, 500:, 500:1000] == 1).mean() > 0.5  # the second frame of the first window


def test_synced_faces_map_24_fps_onto_25_by_repeating_frames(monkeypatch):
    monkeypatch.setattr(tracks, "video_stats", lambda path: (100, 24.0))
    monkeypatch.setattr(tracks, "load_audio", lambda path: np.zeros(16000 * 4, dtype=np.float32))
    monkeypatch.setattr(tracks, "dataset_dir", lambda dataset: Path("x"))
    loader = tracks.SyncedFaceLoader(_Faces(list(range(100)), 224), 54, 25.0, 6)
    out = loader(ROW)
    assert out["faces"].shape == (54, 224, 224, 3) and out["sample_rate"] == 16000
    # the box's centre pixel carries the frame's index, so the order of frames shows through
    picked = out["faces"][:, 112, 112, 0].astype(int)
    assert picked.tolist() == (np.arange(54) / 25 * 24).astype(int).tolist()


def test_too_few_faces_is_no_face():
    faces = _Faces([0, 1, 2], 224, missing=[0, 1])
    with pytest.raises(PreprocessError, match="no_face"):
        tracks.face_track(faces, ROW, min_faces=2)
