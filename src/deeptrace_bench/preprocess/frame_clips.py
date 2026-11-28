"""Whole-frame clips for detectors of generated video (DeMamba, D3), which read no face crop.

A text-to-video clip may have no face worth cropping, and these detectors were trained and
published on whole frames, so the shared face pipeline is skipped. Both upstreams prepare a frame
the same way: trim 10% off each side of its longer dimension, resize to 224 x 224 with OpenCV's
linear interpolation (albumentations' ``Resize`` default), then normalise with ImageNet's mean and
std (left to the adapter, as the colour order is). They differ in which frames they read:

- ``demamba`` (the checkpoint's ``eval_demamba_standalone.py``): every ``round(fps / target)``-th
  decoded frame, ``target`` 2 fps for clips of 4 s or more, else ``ceil(8 / duration)``, at most
  32 kept, then the centre run of ``clip_frames``. A shorter video gives fewer frames, which
  the adapter pads with zeros after normalising, as upstream does.
- ``rate`` (D3's ``utils/video2frame.py``): frames at ``rate`` per second over the first
  ``seconds`` (upstream: ``ffmpeg -t 3 -vf fps=8`` to JPEG, from a random start), the first
  ``clip_frames`` of them, or the first ``min_frames`` when fewer are there. Here the start is
  0, the frame nearest each time is taken and nothing goes through JPEG.

The result is one ``[1, frames, 224, 224, 3]`` uint8 RGB array per item, ``frames`` being
``clip_frames`` except for short videos.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import pandas as pd

from ..paths import dataset_dir
from . import PreprocessError
from .faces import read_video_frames

SIZE = 224


def demamba_indices(
    n_frames: int, fps: float, clip_frames: int = 8, max_frames: int = 32
) -> list[int]:
    """The frames DeMamba reads: its sampling stride, then the centre ``clip_frames`` kept."""
    if n_frames <= 0:
        return []
    fps = fps if fps > 0 else 8.0  # the standalone's fallback for a missing rate
    duration = n_frames / fps
    target = 2 if duration >= 4 else math.ceil(8 / max(duration, 1.0))
    stride = max(1, int(round(fps / target)))
    kept = list(range(0, n_frames, stride))[:max_frames]
    if len(kept) < clip_frames:
        return kept
    start = (len(kept) - clip_frames) // 2
    return kept[start : start + clip_frames]


def rate_indices(
    n_frames: int, fps: float, rate: float, seconds: float, max_frames: int
) -> list[int]:
    """Frames nearest ``0, 1/rate, 2/rate, ...`` in the first ``seconds``, up to ``max_frames``."""
    if n_frames <= 0 or fps <= 0:
        return []
    times = np.arange(0, min(seconds, n_frames / fps), 1.0 / rate)
    indices = np.unique(np.clip(np.round(times * fps).astype(int), 0, n_frames - 1))
    return [int(i) for i in indices[:max_frames]]


def trim_and_resize(rgb: np.ndarray, trim: float = 0.10, size: int = SIZE) -> np.ndarray:
    """Cut ``trim`` off each side of the longer dimension, then resize to ``size`` square."""
    h, w = rgb.shape[:2]
    if w > h:
        p = int(w * trim)
        rgb = rgb[:, p : w - p]
    else:
        p = int(h * trim)
        rgb = rgb[p : h - p, :]
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)


class FrameClipLoader:
    """Turns a manifest row into one clip of prepared whole frames."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.sampling = spec["sampling"]
        self.clip_frames = int(spec["clip_frames"])
        self.min_frames = int(spec.get("min_frames", 1))
        if self.sampling == "demamba":
            self.pick = lambda n, fps: demamba_indices(
                n, fps, self.clip_frames, int(spec.get("max_frames", 32))
            )
        elif self.sampling == "rate":
            self.pick = lambda n, fps: rate_indices(
                n, fps, float(spec["rate"]), float(spec["seconds"]), self.clip_frames
            )
        else:
            raise ValueError(f"unknown frame_clips sampling {self.sampling!r}")

    def __call__(self, row: pd.Series) -> np.ndarray:
        """The item's clip.

        Raises:
            PreprocessError: ``unreadable``, or ``too_short`` with fewer than ``min_frames``.
        """
        path = dataset_dir(row["dataset"]) / row["rel_path"]
        frames, _ = read_video_frames(path, 0, pick=self.pick)
        images = [trim_and_resize(rgb) for _, rgb in frames]
        if len(images) < self.min_frames:
            raise PreprocessError("too_short", f"{len(images)} frames, needs {self.min_frames}")
        if self.sampling == "rate" and len(images) < self.clip_frames:
            images = images[: self.min_frames]  # D3 reads 8 frames when it can't have 16
        return np.stack(images)[None]
