"""Face detection and cropping for video models.

Design (not yet implemented):

1. **Detect once per dataset.** One detector (RetinaFace via ONNX, as GenD uses) runs over
   the sampled frames of every video and stores boxes, five-point landmarks and scores in
   ``DTB_ROOT/faces/<dataset>.parquet``. Detection is the slow step, and doing it once means
   every model sees the same faces.
2. **Crop per model.** Models were trained on different crops (DeepfakeBench aligns with
   81-point landmarks at 256 px, SBI crops at 380 px, GenD aligns at scale 1.3), so each
   model's config carries a ``CropSpec`` and crops are cut from the cached detections into
   ``DTB_CACHE``.
3. **Native mode for reproduction.** Reproducing a paper's numbers uses that model's own
   preprocessing; the shared detector is for cross-model comparison. Both are recorded in
   the run record so they are never mixed up.
4. **Failures are data.** A video with no face in enough frames raises
   ``PreprocessError("no_face")`` and is recorded per group, never dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel


@dataclass(frozen=True)
class FaceDetection:
    """One face found in one frame."""

    frame_index: int
    box: tuple[float, float, float, float]
    landmarks: np.ndarray
    score: float


class CropSpec(BaseModel):
    """How a model wants its faces cut."""

    size: int
    margin: float = 1.3
    align: Literal["none", "five_point"] = "none"
    normalize: Literal["imagenet", "clip", "none"] = "imagenet"


def sample_frame_indices(n_frames: int, k: int) -> list[int]:
    """Pick ``k`` frame indices spread evenly over a video of ``n_frames`` frames."""
    if n_frames <= 0:
        return []
    if n_frames <= k:
        return list(range(n_frames))
    return [int(i) for i in np.linspace(0, n_frames - 1, k).round()]


def detect_faces(video_path: Path, frame_indices: list[int]) -> list[FaceDetection]:
    """Detect faces in the given frames of a video. Not implemented yet; see module doc."""
    raise NotImplementedError("shared face detection is planned; see docs/evaluation.md")


def crop_face(frame: np.ndarray, detection: FaceDetection, spec: CropSpec) -> np.ndarray:
    """Cut one face from a frame per a model's crop spec. Not implemented yet."""
    raise NotImplementedError("per-model cropping is planned; see docs/evaluation.md")
