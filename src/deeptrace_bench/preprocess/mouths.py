"""Mouth crops for lip-based detectors (LipForensics), from consecutive frames.

LipForensics reads runs of consecutive grayscale mouth crops. Its own preprocessing
(``preprocessing/crop_mouths.py``, MIT) starts from face frames with 68 landmarks per frame,
which its README suggests getting from RetinaFace and FAN. Here:

1. **Frames and faces**: the first ``frames_per_clip`` frames, every one of them, with the
   shared detector's largest face per frame (``FaceLoader.detections``, cached as usual).
2. **Landmarks**: FAN (2DFAN4 from the ``face-alignment`` package, its TorchScript weights
   pinned in the model's config) on each face box, cropped and decoded with the package's own
   ``crop`` and ``get_preds_fromhm`` exactly as ``FaceAlignment.get_landmarks_from_image``
   does for a given box (``reference_scale`` 195, the box centre raised by 12% of its height).
   A frame without a face takes the nearest earlier landmarks (the first ones, before any).
3. **Mouths**: upstream's loop, with its ``warp_img``, ``apply_transform`` and ``cut_patch``
   imported from the checkout: landmarks smoothed over 12 frames, each frame warped to the
   LRW mean face (``20words_mean_face.npy``) on five stable points into 256 x 256, then a
   96 x 96 patch around the mean mouth landmark; frames after the last full window reuse the
   last transform. Each patch becomes grayscale the way upstream's loader converts it (PIL
   ``"L"``).

The result is one ``[n_frames, 96, 96]`` uint8 array per item; the adapter cuts it into
clips. Too few frames with a face is ``no_face``.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import PreprocessError
from .faces import FaceLoader

STABLE_POINTS = [33, 36, 39, 42, 45]
_STD_SIZE = (256, 256)
_REFERENCE_SCALE = 195.0  # face-alignment's value for its default (SFD) boxes


class MouthLoader:
    """Turns a manifest row into an array of grayscale mouth crops, one per frame."""

    def __init__(
        self,
        faces: FaceLoader,
        fan_path: Path,
        upstream_dir: Path,
        device: str = "cpu",
        min_frames: int = 25,
        window_margin: int = 12,
        crop_size: int = 96,
        mouth: tuple[int, int] = (48, 68),
    ) -> None:
        if not faces.consecutive:
            raise ValueError("mouth crops need consecutive frames (input.frame_sampling)")
        self.faces = faces
        self.fan_path = fan_path
        self.upstream_dir = Path(upstream_dir)
        self.device = device
        self.min_frames = min_frames
        self.window_margin = window_margin
        self.half = crop_size // 2
        self.mouth = slice(*mouth)
        self._fan: Any = None
        self._utils: Any = None
        self._mean_face: np.ndarray | None = None

    def _setup(self) -> None:
        import torch

        from ..models._upstream import load_module

        self._fan = torch.jit.load(str(self.fan_path), map_location=self.device).eval()
        prep = self.upstream_dir / "preprocessing"
        self._utils = load_module(prep / "utils.py", "dtb_upstream_lipforensics_prep")
        self._mean_face = np.load(prep / "20words_mean_face.npy")

    def landmarks(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        """FAN's 68 landmarks (image coordinates) for the face in ``box``."""
        import torch
        from face_alignment.utils import crop, get_preds_fromhm

        if self._fan is None:
            self._setup()
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        center = torch.tensor([x2 - (x2 - x1) / 2.0, y2 - (y2 - y1) / 2.0])
        center[1] = center[1] - (y2 - y1) * 0.12
        scale = (x2 - x1 + y2 - y1) / _REFERENCE_SCALE
        patch = crop(image, center, scale)
        inp = torch.from_numpy(patch.transpose((2, 0, 1))).float().div(255.0).unsqueeze(0)
        with torch.inference_mode():
            heatmaps = self._fan(inp.to(self.device)).float().cpu().numpy()
        _, points, _ = get_preds_fromhm(heatmaps, center.numpy(), scale)
        return np.asarray(points, dtype=np.float64).reshape(68, 2)

    def __call__(self, row: pd.Series) -> np.ndarray:
        images, records = self.faces.detections(row)
        records = sorted(records, key=lambda r: r["frame_index"])
        found = {
            r["frame_index"]: self.landmarks(images[r["frame_index"]], np.asarray(r["box"]))
            for r in records
            if r["box"] is not None and r["frame_index"] in images
        }
        if len(found) < self.min_frames:
            raise PreprocessError("no_face", f"{len(found)} of {len(records)} frames have a face")
        order = [r["frame_index"] for r in records if r["frame_index"] in images]
        first = found[min(found)]
        filled, last = [], first
        for index in order:
            last = found.get(index, last)
            filled.append(last)
        crops = self._mouths([images[i] for i in order], filled)
        return np.stack(crops)

    def _mouths(self, frames: list[np.ndarray], landmarks: list[np.ndarray]) -> list[np.ndarray]:
        """Upstream's ``crop_video_and_save`` loop, returning grayscale patches."""
        from PIL import Image

        utils, mean_face = self._utils, self._mean_face
        assert utils is not None and mean_face is not None

        def patch(frame: np.ndarray, marks: np.ndarray) -> np.ndarray:
            try:
                cut = utils.cut_patch(frame, marks[self.mouth], self.half, self.half)
            except Exception as exc:  # upstream raises a bare Exception when off the frame
                raise PreprocessError("no_face", f"mouth off the frame: {exc}") from exc
            return np.asarray(Image.fromarray(cut.astype(np.uint8)).convert("L"))

        out = []
        q_frames: deque = deque()
        q_marks: deque = deque()
        trans = None
        for frame, marks in zip(frames, landmarks, strict=True):
            q_frames.append(frame)
            q_marks.append(marks)
            if len(q_frames) == self.window_margin:
                smoothed = np.mean(q_marks, axis=0)
                cur_marks, cur_frame = q_marks.popleft(), q_frames.popleft()
                warped, trans = utils.warp_img(
                    smoothed[STABLE_POINTS, :], mean_face[STABLE_POINTS, :], cur_frame, _STD_SIZE
                )
                out.append(patch(warped, trans(cur_marks)))
        if trans is None:  # fewer frames than one smoothing window
            raise PreprocessError(
                "too_short", f"{len(frames)} frames, smoothing needs {self.window_margin}"
            )
        while q_frames:
            cur_frame, cur_marks = q_frames.popleft(), q_marks.popleft()
            warped = utils.apply_transform(trans, cur_frame, _STD_SIZE)
            out.append(patch(warped, trans(cur_marks)))
        return out

    def close(self) -> None:
        """Flush the face cache."""
        self.faces.close()
