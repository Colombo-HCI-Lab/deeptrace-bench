"""Runs of face crops for video models that take a few frames at once (FakeSTormer).

FakeSTormer's test data is a folder of face crops per video, 32 frames spread evenly over it
(``package_utils/images_crop.py -n 32``). Its dataset sorts them by name and cuts the list into
runs of ``NUM_FRAMES`` crops that do not overlap, starting at the first and dropping what is
left over (``_sampler`` in ``datasets/common.py``); each run is one network input, and a
video's score is the mean over its runs. A run therefore spans several seconds of the video, not
consecutive frames.

``FrameGroupLoader`` does the same to the crops the shared ``FaceLoader`` returns (evenly spaced
frames that have a face, in order): one ``[n_groups, group_frames, h, w, 3]`` uint8 RGB array
per item.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import PreprocessError
from .faces import FaceLoader


def group_crops(crops: list[np.ndarray], group_frames: int) -> np.ndarray:
    """Cut ``crops`` into runs of ``group_frames`` that do not overlap, dropping the rest.

    Raises:
        PreprocessError: ``too_short`` if there are fewer crops than one run.
    """
    n_groups = len(crops) // group_frames
    if n_groups == 0:
        raise PreprocessError("too_short", f"{len(crops)} faced frames, a run takes {group_frames}")
    kept = np.stack(crops[: n_groups * group_frames])
    return kept.reshape(n_groups, group_frames, *kept.shape[1:])


class FrameGroupLoader:
    """Turns a manifest row into runs of face crops."""

    def __init__(self, faces: FaceLoader, group_frames: int = 4) -> None:
        self.faces = faces
        self.group_frames = group_frames

    def __call__(self, row: pd.Series) -> np.ndarray:
        """The runs of one item.

        Raises:
            PreprocessError: ``unreadable``, ``no_face`` or ``too_short``.
        """
        return group_crops(self.faces(row), self.group_frames)

    def close(self) -> None:
        """Write any cached detections not yet on disk."""
        self.faces.close()
