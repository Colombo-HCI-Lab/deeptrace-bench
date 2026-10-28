"""Whole-face crops for RealForensics, from consecutive frames.

RealForensics (``preprocessing/extract_faces.py``, MIT) cuts the whole face, not the mouth,
from frames warped to LRW's mean face, the way LipForensics cuts mouths. It differs in the
details, so it gets its own loader on the same first two steps:

1. **Frames, faces and landmarks**: as ``MouthLoader``: the first ``frames_per_clip``
   consecutive frames, the shared detector's largest face per frame, and FAN's 68 landmarks on
   each face box; a frame without a face takes the nearest earlier landmarks.
2. **Faces**: upstream's ``crop_patch``, imported from the checkout: each frame's landmarks
   averaged over a window of up to 6 frames either side and re-centred on the frame's own
   landmarks, the frame warped onto ``20words_mean_face.npy`` on eight stable points
   (28, 33, 36, 39, 42, 45, 48, 54) into 256 x 256, then a 150 x 150 patch centred on the mean
   of landmarks 15 to 67, in colour.

The result is one ``[n_frames, 150, 150, 3]`` uint8 RGB array per item; the adapter cuts it
into clips. A patch that runs off the warped frame is ``no_face``.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from . import PreprocessError
from .mouths import MouthLoader

# extract_faces.py's defaults (its load_args)
_CROP = 150
_START, _STOP = 15, 68
_WINDOW_MARGIN = 12


class TalkingFaceLoader(MouthLoader):
    """Turns a manifest row into RealForensics' face patches, one per frame."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._extract: Any = None
        self._args = argparse.Namespace(
            crop_width=_CROP,
            crop_height=_CROP,
            start_idx=_START,
            stop_idx=_STOP,
            window_margin=_WINDOW_MARGIN,
        )

    def _setup(self) -> None:
        import torch

        from ..models._upstream import load_module

        self._fan = torch.jit.load(str(self.fan_path), map_location=self.device).eval()
        prep = self.upstream_dir / "preprocessing"
        self._extract = load_module(prep / "extract_faces.py", "dtb_upstream_realforensics_prep")
        self._mean_face = np.load(prep / "20words_mean_face.npy")

    def _mouths(self, frames: list[np.ndarray], landmarks: list[np.ndarray]) -> list[np.ndarray]:
        """Upstream's ``crop_patch`` over the whole sequence."""
        assert self._extract is not None and self._mean_face is not None
        try:
            patches = self._extract.crop_patch(frames, landmarks, self._mean_face, self._args)
        except Exception as exc:  # upstream raises a bare Exception when off the frame
            raise PreprocessError("no_face", f"face patch off the frame: {exc}") from exc
        return list(np.asarray(patches, dtype=np.uint8))
