"""Frames, alignment, the detection cache and the loader, on synthetic images and videos."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from deeptrace_bench.preprocess import PreprocessError
from deeptrace_bench.preprocess.faces import (
    _TEMPLATE,
    CropSpec,
    FaceLoader,
    align_face,
    crop_face,
    read_video_frames,
    sample_frame_indices,
)


def _template_dst(size: int, margin: float) -> np.ndarray:
    pad = size * (margin - 1) / 2
    return (_TEMPLATE * size + pad) * (size / (size + 2 * pad))


def _dots(points: np.ndarray, shape=(600, 600)) -> np.ndarray:
    image = np.zeros((*shape, 3), np.uint8)
    for x, y in points:
        cv2.circle(image, (round(x), round(y)), 3, (255, 255, 255), -1)
    return image


def _centroids(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    n, _, _, centers = cv2.connectedComponentsWithStats((gray > 100).astype(np.uint8))
    return centers[1:n]


def _rotated(scale: float, degrees: float, offset: tuple[float, float]) -> np.ndarray:
    theta = np.deg2rad(degrees)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return (_TEMPLATE * scale) @ rot.T + np.array(offset)


def test_alignment_lands_landmarks_on_the_template():
    landmarks = _rotated(200, 15, (180, 150))
    crop = align_face(_dots(landmarks), landmarks, size=None, margin=1.3)
    assert crop.shape == (260, 260, 3)  # 200 px face spread x 1.3 margin
    found = _centroids(crop)
    expected = _template_dst(260, 1.3)
    for point in expected:
        assert np.min(np.linalg.norm(found - point, axis=1)) < 1.5


def test_a_fixed_size_is_respected():
    landmarks = _rotated(120, -10, (200, 220))
    assert align_face(_dots(landmarks), landmarks, size=224).shape == (224, 224, 3)


def _video(path, n_frames: int, size=(64, 48)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, size)
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), i * 5, np.uint8))
    writer.release()


def test_video_frames_are_the_evenly_spread_ones(tmp_path):
    path = tmp_path / "clip.mp4"
    _video(path, 40)
    frames, n = read_video_frames(path, 8)
    assert n == 40
    assert [i for i, _ in frames] == sample_frame_indices(40, 8)
    for index, rgb in frames:
        assert abs(float(rgb.mean()) - index * 5) < 4  # the codec is lossy


def test_garbage_is_unreadable(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a video")
    with pytest.raises(PreprocessError) as info:
        read_video_frames(path, 8)
    assert info.value.reason == "unreadable"


# --- the loader, with a stub detector ---------------------------------------------------


FACE_KPS = _rotated(60, 0, (40, 30))


class StubDetector:
    """Finds one face, in every frame or only in images bigger than ``min_side``."""

    def __init__(self, min_side: int = 0, every: int = 1) -> None:
        self.calls = 0
        self.min_side = min_side
        self.every = every

    def __call__(self, rgb: np.ndarray):
        self.calls += 1
        if min(rgb.shape[:2]) < self.min_side or (self.calls - 1) % self.every:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        # Centre the stub face in whatever image it is given.
        h, w = rgb.shape[:2]
        kps = (FACE_KPS - FACE_KPS.mean(axis=0) + [w / 2, h / 2]).astype(np.float32)
        box = np.array([[kps[:, 0].min(), kps[:, 1].min(), kps[:, 0].max(), kps[:, 1].max(), 0.9]])
        return box.astype(np.float32), kps[None]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DTB_ROOT", str(tmp_path))
    monkeypatch.setenv("DTB_NAMESPACE", "smoke")
    folder = tmp_path / "smoke" / "datasets" / "toy"
    folder.mkdir(parents=True)
    cv2.imwrite(str(folder / "face.png"), np.full((80, 80, 3), 128, np.uint8))
    _video(folder / "clip.mp4", 40, size=(96, 96))
    return tmp_path


def _loader(detector, **kwargs) -> FaceLoader:
    return FaceLoader(
        crop=CropSpec(size=None, margin=1.3, align="five_point"),
        make_detector=lambda: detector,
        detector_id="stub",
        settings={"detector": "stub"},
        frames_per_clip=8,
        min_face_frames=4,
        image_min_face_frames=1,
        **kwargs,
    )


def _row(rel: str, modality: str) -> pd.Series:
    return pd.Series(
        {"item_id": f"toy/{rel}", "dataset": "toy", "rel_path": rel, "modality": modality}
    )


def test_a_video_gives_one_crop_per_frame_with_a_face(store):
    crops = _loader(StubDetector())(_row("clip.mp4", "video"))
    assert len(crops) == 8 and all(c.dtype == np.uint8 and c.ndim == 3 for c in crops)


def test_too_few_faces_is_no_face_for_video_but_one_is_enough_for_an_image(store):
    sparse = StubDetector(every=3)  # a face in 3 of 8 frames, fewer than min_face_frames
    with pytest.raises(PreprocessError) as info:
        _loader(sparse)(_row("clip.mp4", "video"))
    assert info.value.reason == "no_face"
    assert len(_loader(StubDetector())(_row("face.png", "image"))) == 1


def test_detections_are_cached_and_reused(store):
    detector = StubDetector()
    loader = _loader(detector)
    loader(_row("clip.mp4", "video"))
    loader.close()
    calls = detector.calls
    again = _loader(detector)
    assert len(again(_row("clip.mp4", "video"))) == 8
    assert detector.calls == calls  # every frame came from the cache
    assert list((store / "smoke" / "faces" / "toy").glob("stub__*/part-*.parquet"))


def test_a_tight_face_crop_is_found_on_the_padded_retry(store):
    detector = StubDetector(min_side=100)  # only "sees" faces once the 80 px image is padded
    loader = _loader(detector, image_pad_retry=0.5)
    crops = loader(_row("face.png", "image"))
    assert len(crops) == 1
    record = loader._cache("toy").get("toy/face.png")[0]
    assert record["pad"] == 0.5
    kps = np.asarray(record["kps"]).reshape(5, 2)
    assert (kps >= 0).all() and (kps <= 80).all()  # back in the original image's coordinates
    with pytest.raises(PreprocessError):
        _loader(StubDetector(min_side=100))(_row("face.png", "image"))


def test_crops_are_saved_for_people_to_look_at(store, tmp_path):
    loader = _loader(StubDetector(), save_crops_to=tmp_path / "crops", save_crops=2)
    loader(_row("clip.mp4", "video"))
    assert len(list((tmp_path / "crops" / "toy" / "clip.mp4").glob("frame_*.png"))) == 2


def test_crop_spec_refuses_unknown_keys():
    # Normalisation belongs to each adapter; a crop spec only says how faces are cut.
    with pytest.raises(ValidationError):
        CropSpec(size=256, normalize="imagenet")


def test_box_crops_follow_sbis_test_rounding():
    image = np.arange(100 * 100 * 3, dtype=np.uint32).reshape(100, 100, 3).astype(np.uint8)
    box = np.array([10.6, 20.2, 50.9, 80.7], dtype=np.float32)
    spec = CropSpec(size=None, margin=1.25, align="box")
    # SBI: margins of w/8 and h/8, int() towards zero, +1 at the far edges
    crop = crop_face(image, box, np.zeros((5, 2)), spec)
    np.testing.assert_array_equal(crop, image[12:89, 5:56])
    resized = crop_face(image, box, np.zeros((5, 2)), CropSpec(size=380, margin=1.25, align="box"))
    assert resized.shape == (380, 380, 3)
