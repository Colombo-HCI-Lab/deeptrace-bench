"""Inputs for audio-visual models: face crops at a fixed frame rate plus the audio track.

An audio-visual detector (HAVIC) scores windows of consecutive frames together with the
audio of the same stretch, so its loader hands over both, plus what the model needs to line
them up: how many frames the rate gives over the whole video (``n_sampled``), since the
model keeps the audio in the same proportion as the frames it uses. Windowing and audio
features are the adapter's business; the loader only decodes.

Failures are the faces' (``no_face``, ``unreadable``) or the audio's (``no_audio``,
``empty_audio``), and become the item's status like any other.
"""

from __future__ import annotations

from typing import Any

import cv2
import pandas as pd

from ..paths import dataset_dir
from .audio import SAMPLE_RATE, load_audio
from .faces import FaceLoader, rate_frame_count


class AudioVideoLoader:
    """Turns a manifest row into ``{"faces", "audio", "sample_rate", "n_sampled"}``."""

    def __init__(self, faces: FaceLoader, sample_rate: int = SAMPLE_RATE) -> None:
        if faces.frame_rate is None:
            raise ValueError("an audio-visual model needs input.frame_rate for its frames")
        self.faces = faces
        self.sample_rate = sample_rate

    def __call__(self, row: pd.Series) -> dict[str, Any]:
        path = dataset_dir(row["dataset"]) / row["rel_path"]
        crops = self.faces(row)
        audio = load_audio(path, self.sample_rate)
        cap = cv2.VideoCapture(str(path))
        try:
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()
        assert self.faces.frame_rate is not None
        return {
            "faces": crops,
            "audio": audio,
            "sample_rate": self.sample_rate,
            "n_sampled": rate_frame_count(n_frames, fps, self.faces.frame_rate),
        }

    def close(self) -> None:
        """Flush the face cache."""
        self.faces.close()
