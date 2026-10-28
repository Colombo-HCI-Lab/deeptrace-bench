"""Aligned face clips for clip-level video detectors (FTCN), from consecutive frames.

FTCN aligns a whole clip at once: one similarity transform, fitted to the five landmarks of
every frame in the clip together, warps each frame onto its 224 x 224 template, so the face
holds still across the clip and only its own motion is left. Upstream's
``test_on_raw_video.py`` (no licence stated, so imported from the pinned checkout, never
copied) does it in four steps, followed here:

1. **Frames and faces**: the first ``max_frames`` frames, every one of them, with the shared
   detector's largest face per frame (``FaceLoader.detections``, cached as usual). Upstream
   reads up to 768 frames with RetinaFace and tracks faces across them; here the frames that
   have a face form one track, so a frame without one is skipped rather than ending the track.
2. **Context crops**: each face box widened by half its width and height on each side
   (upstream's ``get_crop_box``, imported), cut from the frame, with the five landmarks moved
   into the crop's coordinates.
3. **Clips**: windows of ``clip_frames`` consecutive faces, every ``clip_stride`` frames.
   Upstream slides one frame at a time; a stride keeps the cost down and the mean over clips,
   which is upstream's video score, close. A track shorter than one clip is mirror-padded at
   both ends exactly as upstream pads it.
4. **Alignment**: upstream's ``FasterCropAlignXRay`` on each window: the crops are placed on
   one canvas spanning all their boxes (pixels outside a frame's own crop stay black) and
   warped with the clip's shared transform. Its 68-landmark output isn't used, so the 68
   landmarks it also moves are passed as zeros.

The result is one ``[n_clips, clip_frames, size, size, 3]`` uint8 RGB array per item.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import PreprocessError
from .faces import FaceLoader

_TOOLS = "dtb_upstream_ftcn_tools"


def pad_indices(length: int, clip_frames: int) -> list[int]:
    """Frame indices of a track, mirror-padded at both ends when it is shorter than a clip.

    Upstream's padding in ``test_on_raw_video.py``, rewritten as a function: ``clip_frames - 1``
    indices of the track reflected without its end points are added before and after it.
    Upstream repeats the leading reflection only twice (it reuses the trailing one's length),
    so a track shorter than about a quarter of a clip fails its assertion there; here that is
    ``too_short``.

    Raises:
        PreprocessError: ``too_short``.
    """
    inner = list(range(length))
    if length >= clip_frames:
        return inner
    pad = clip_frames - 1
    post = inner[1:-1][::-1] + inner
    post = (post * (pad // len(post) + 1))[:pad]
    pre = inner + inner[1:-1][::-1]
    pre = (pre * (pad // len(post) + 1))[-pad:]
    if len(pre) != pad:
        raise PreprocessError("too_short", f"{length} faced frames can't be padded to a clip")
    return pre + inner + post


def clip_windows(length: int, clip_frames: int, stride: int) -> list[list[int]]:
    """Windows of ``clip_frames`` indices into a (padded) track, starting every ``stride``."""
    indices = pad_indices(length, clip_frames)
    starts = range(0, len(indices) - clip_frames + 1, stride)
    return [indices[s : s + clip_frames] for s in starts]


class FaceClipLoader:
    """Turns a manifest row into FTCN's aligned clips."""

    def __init__(
        self,
        faces: FaceLoader,
        upstream_dir: Path,
        clip_frames: int = 32,
        clip_stride: int = 16,
        size: int = 224,
        context: float = 0.5,
        min_frames: int = 8,
    ) -> None:
        if not faces.consecutive:
            raise ValueError("face clips need consecutive frames (input.frame_sampling)")
        self.faces = faces
        self.upstream_dir = Path(upstream_dir)
        self.clip_frames = clip_frames
        self.clip_stride = clip_stride
        self.size = size
        self.context = context
        self.min_frames = min_frames
        self._align: Any = None
        self._crop_box: Any = None

    def _setup(self) -> None:
        from ..models._upstream import load_module, load_package

        tools = self.upstream_dir / "test_tools"
        load_package(tools, _TOOLS)
        aligner = importlib.import_module(f"{_TOOLS}.faster_crop_align_xray")
        self._align = aligner.FasterCropAlignXRay(self.size)
        self._crop_box = load_module(tools / "utils.py", f"{_TOOLS}_utils").get_crop_box

    def _crops(self, images: dict[int, np.ndarray], records: list[dict]) -> list[tuple]:
        """Each faced frame's context crop and upstream's landmark tuple for it."""
        out = []
        for record in sorted(records, key=lambda r: r["frame_index"]):
            image = images.get(record["frame_index"])
            if image is None or record["kps"] is None:
                continue
            box = np.asarray(record["box"][:4], dtype=np.float64)
            kps = np.asarray(record["kps"], dtype=np.float64).reshape(5, 2)
            big = self._crop_box(image.shape[:2], box, scale=self.context)
            top_left = big[:2][None, :]
            info = (
                (box.reshape(2, 2) - top_left).reshape(-1),
                kps - top_left,
                np.zeros((68, 2)),
                big,
            )
            x1, y1, x2, y2 = big
            out.append((image[y1:y2, x1:x2], info))
        return out

    def __call__(self, row: pd.Series) -> np.ndarray:
        if self._align is None:
            self._setup()
        images, records = self.faces.detections(row)
        crops = self._crops(images, records)
        if len(crops) < self.min_frames:
            raise PreprocessError("no_face", f"{len(crops)} of {len(records)} frames have a face")
        clips = []
        for window in clip_windows(len(crops), self.clip_frames, self.clip_stride):
            frames = [crops[i][0] for i in window]
            landmarks = [crops[i][1] for i in window]
            _, aligned = self._align(landmarks, frames)
            clips.append(np.asarray(aligned, dtype=np.uint8))
        return np.stack(clips)
