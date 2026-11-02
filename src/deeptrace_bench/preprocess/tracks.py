"""Steady face tracks for audio-visual models trained on LRS-style clips (LipFD, AVAD).

LRS2 and LRS3, which both models learned from, are face tracks cut by the SyncNet pipeline
(``syncnet_python``'s ``crop_video``, MIT): the face box's centre and size median-filtered over
13 frames, then a square from ``s`` above the centre to ``s * (1 + 2 * cs)`` below it and
``s * (1 + cs)`` either side, with ``s`` half the box and ``cs`` 0.4, on a frame padded with grey
110. Neither model's scoring code crops anything, because its test clips (FakeAVCeleb, AVLips)
were already face tracks. Benchmark videos often aren't, so both start from that same crop:

- **Boxes**: the shared detector's largest face per frame (``FaceLoader.detections``, cached as
  usual). A frame without a face takes the nearest frame that has one.
- **Smoothing**: ``median_filter`` with nearest-edge padding rather than ``medfilt``'s zeros,
  which would shrink the first and last boxes.

``LipCompositeLoader`` makes LipFD's inputs, ``SyncedFaceLoader`` AVAD's.
"""

from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
import pandas as pd

from ..paths import dataset_dir
from . import PreprocessError
from .audio import load_audio
from .faces import FaceLoader

_CROP_SCALE = 0.4  # syncnet_python's --crop_scale
_MEDIAN_FRAMES = 13
_PAD_VALUE = 110

# LipFD's preprocess.py
_LIP_SIDE = 500
_LIP_SAMPLE_RATE = 22050


def fill_boxes(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Frame indices and ``[n, 4]`` boxes, frames without a face taking the nearest box.

    Raises:
        PreprocessError: ``no_face`` if no frame has a face.
    """
    found = [i for i, record in enumerate(records) if record["box"] is not None]
    if not found:
        raise PreprocessError("no_face", f"0 of {len(records)} frames have a face")
    order = np.asarray(found)
    positions = np.arange(len(records))
    nearest = order[np.abs(positions[:, None] - order[None, :]).argmin(axis=1)]
    boxes = np.array([records[i]["box"] for i in nearest], dtype=np.float32)
    return np.array([r["frame_index"] for r in records]), boxes


def smooth_track(boxes: np.ndarray) -> np.ndarray:
    """``[n, 3]`` centre x, centre y and half size, median-filtered over time."""
    from scipy.ndimage import median_filter

    half = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) / 2
    track = np.stack(
        [(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2, half], axis=1
    )
    return median_filter(track, size=(_MEDIAN_FRAMES, 1), mode="nearest")


def track_crop(frame: np.ndarray, x: float, y: float, s: float, size: int) -> np.ndarray:
    """SyncNet's face crop at one point of a track, resized to ``size``."""
    pad = int(s * (1 + 2 * _CROP_SCALE))
    padded = np.pad(frame, ((pad, pad), (pad, pad), (0, 0)), constant_values=_PAD_VALUE)
    my, mx = y + pad, x + pad
    face = padded[
        int(my - s) : int(my + s * (1 + 2 * _CROP_SCALE)),
        int(mx - s * (1 + _CROP_SCALE)) : int(mx + s * (1 + _CROP_SCALE)),
    ]
    if face.size == 0:
        raise PreprocessError("no_face", "track lies outside the frame")
    return cv2.resize(face, (size, size))


def face_track(faces: FaceLoader, row: pd.Series, min_faces: int) -> tuple[np.ndarray, np.ndarray]:
    """One item's decoded frame indices and their track crops, ``[n, size, size, 3]`` RGB.

    Raises:
        PreprocessError: ``unreadable`` or ``no_face``.
    """
    images, records = faces.detections(row)
    records = [r for r in records if r["frame_index"] in images]
    have = sum(r["box"] is not None for r in records)
    if have < min_faces:
        raise PreprocessError("no_face", f"{have} of {len(records)} frames have a face")
    indices, boxes = fill_boxes(records)
    track = smooth_track(boxes)
    assert faces.crop.size is not None
    crops = [
        track_crop(images[int(i)], *point, faces.crop.size)
        for i, point in zip(indices, track, strict=True)
    ]
    return indices, np.stack(crops)


def video_stats(path: Any) -> tuple[int, float]:
    """A video's frame count and frame rate from its header."""
    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()


def mel_image(wave: np.ndarray) -> np.ndarray:
    """``get_spectrogram`` of LipFD's ``preprocess.py``: the mel spectrogram as an RGB image.

    Upstream saves it with ``plt.imsave`` (viridis, stretched to its own range) and reads the
    PNG back, then scales by 255 and truncates, which this keeps, down to the one-level
    rounding losses.
    """
    import librosa
    from matplotlib import image as mpimg

    mel = librosa.feature.melspectrogram(y=wave, sr=_LIP_SAMPLE_RATE)
    mel = librosa.power_to_db(mel, ref=np.min)
    buffer = io.BytesIO()
    mpimg.imsave(buffer, mel, format="png")
    buffer.seek(0)
    return (mpimg.imread(buffer, format="png") * 255).astype(np.uint8)[:, :, :3]


class LipCompositeLoader:
    """Turns a manifest row into LipFD's images, ``[n_windows, 1000, 2500, 3]`` RGB uint8.

    ``preprocess.py`` of LipFD (no licence stated, so rebuilt here from what it does; the
    parity check runs upstream's reader on the result). Each image is five consecutive frames
    side by side, 500 x 500 each, under the stretch of mel spectrogram they span, resized to
    the same width. Upstream's frame indices go through ``dtype=np.uint8``, which only matters
    past frame 255, where it wraps and scrambles the windows; plain integers here. Audio is
    decoded with the shared loader (soxr) rather than ``librosa.load``'s default resampler.
    """

    def __init__(self, faces: FaceLoader, min_faces: int) -> None:
        if faces.windows is None:
            raise ValueError("LipFD needs input.frame_sampling: windows")
        self.faces = faces
        self.min_faces = min_faces

    def __call__(self, row: pd.Series) -> np.ndarray:
        path = dataset_dir(row["dataset"]) / row["rel_path"]
        assert self.faces.windows is not None
        count, length = self.faces.windows
        n_frames, _ = video_stats(path)
        indices, crops = face_track(self.faces, row, self.min_faces)
        by_frame = {int(i): crop for i, crop in zip(indices, crops, strict=True)}
        mel = mel_image(load_audio(path, _LIP_SAMPLE_RATE))
        mapping = mel.shape[1] / n_frames

        images = []
        for start in np.linspace(0, n_frames - length - 1, count).astype(int):
            start = int(start)
            frames = [by_frame.get(start + i) for i in range(length)]
            begin = int(np.round(start * mapping))
            end = int(np.round((start + length) * mapping))
            if any(frame is None for frame in frames) or end <= begin:
                continue
            sub_mel = cv2.resize(mel[:, begin:end], (_LIP_SIDE * length, _LIP_SIDE))
            images.append(np.concatenate((sub_mel, np.concatenate(frames, axis=1)), axis=0))
        if not images:
            raise PreprocessError("too_short", "no window with frames and audio")
        return np.stack(images)

    def close(self) -> None:
        """Flush the face cache."""
        self.faces.close()


class SyncedFaceLoader:
    """Turns a manifest row into AVAD's inputs: ``{"faces", "audio", "sample_rate"}``.

    ``faces`` is ``[n, size, size, 3]`` RGB uint8, a track at ``fps``. AVAD's dataset class
    assumes 25 fps video (its resampling command is written but commented out, and its audio
    windows are cut at 640 samples per frame), so other rates are mapped onto 25 here the way
    ffmpeg's ``-r 25`` does, repeating or skipping frames. Only the first ``frames`` count: the
    scoring code reads 50 windows of five.
    """

    def __init__(self, faces: FaceLoader, frames: int, fps: float, min_faces: int) -> None:
        self.faces = faces
        self.frames = frames
        self.fps = fps
        self.min_faces = min_faces

    def __call__(self, row: pd.Series) -> dict[str, Any]:
        path = dataset_dir(row["dataset"]) / row["rel_path"]
        indices, crops = face_track(self.faces, row, self.min_faces)
        _, fps = video_stats(path)
        if fps <= 0:
            raise PreprocessError("unreadable", f"{path}: no frame rate")
        wanted = (np.arange(self.frames) / self.fps * fps).astype(int)
        wanted = wanted[wanted <= indices[-1]]
        return {
            "faces": crops[np.searchsorted(indices, wanted)],
            "audio": load_audio(path),
            "sample_rate": 16000,
        }

    def close(self) -> None:
        """Flush the face cache."""
        self.faces.close()
