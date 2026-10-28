"""RealForensics (Haliassos et al., CVPR 2022, ahaliassos/RealForensics, MIT): a CSN-101 video
network pretrained self-supervised on real talking faces (LRW, with their audio) and then
trained on FF++ to detect forgeries from video alone.

The network is upstream's own: ``csn_temporal_no_head(model_depth=101, input_channel=3)`` from
``stage2/models/backbones/csn.py`` (built from pytorchvideo's blocks) and the ``MeanLinear``
head of ``stage2/models/linear.py`` with ``norm_linear`` and scale 64, as
``conf/model/combined.yaml`` configures them. Those files import ``stage2.*`` by its top-level
name, so the checkout's root is on ``sys.path`` while they import and ``stage2`` leaves
``sys.modules`` afterwards.

Weights: ``realforensics_ff.pth``, a state dict with ``backbone.*`` and ``df_head.*`` keys,
which ``stage2/eval.py`` loads into the two halves; ``convert_checkpoint`` re-saves exactly
those, and the whole network then loads with every key matching.

Inputs come from ``preprocess.talking_faces.TalkingFaceLoader`` (FAN landmarks, upstream's
``crop_patch``, 150 x 150 RGB faces). As upstream's test data module: clips of 25 frames back
to back, the last one moved back to end at the last frame when frames are left over
(``UniformClipSampler`` with ``backpad_last``); each scaled to [0, 1], centre-cropped to 140,
resized to 112 (bilinear, no antialiasing, as torchvision 0.10 resized tensors) and
normalised with ImageNet's mean and std. Output: one logit per clip; P(fake) is its sigmoid.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..preprocess import PreprocessError
from ._upstream import scoped_modules
from .base import Detector, resolve_device

log = logging.getLogger(__name__)

_CROP, _SIZE = 140, 112
_MEAN = np.array([0.485, 0.456, 0.406])
_STD = np.array([0.229, 0.224, 0.225])
_BATCH = 4


def clip_starts(n_frames: int, clip: int) -> list[int]:
    """Clip starts of pytorchvideo's uniform sampler with ``backpad_last``."""
    if n_frames < clip:
        return []
    starts = list(range(0, n_frames - clip + 1, clip))
    if n_frames % clip:
        starts.append(n_frames - clip)
    return starts


@contextlib.contextmanager
def _upstream_imports(root: Path) -> Iterator[None]:
    sys.path.insert(0, str(root))
    try:
        with scoped_modules({}, purge=["stage2"]):
            yield
    finally:
        sys.path.remove(str(root))


def build_network(root: Path) -> Any:
    """Upstream's backbone and head, under ``backbone`` and ``df_head`` as in its learner."""
    import torch.nn as nn

    if not (root / "stage2" / "models" / "linear.py").is_file():
        raise FileNotFoundError(f"{root} is missing; run scripts/setup_models.py realforensics")
    with _upstream_imports(root):
        csn = importlib.import_module("stage2.models.backbones.csn")
        linear = importlib.import_module("stage2.models.linear")
        network = nn.Module()
        network.backbone = csn.csn_temporal_no_head(model_depth=101, input_channel=3)
        network.df_head = linear.MeanLinear(2048, out_dim=1, norm_linear=True, scale=64)
    return network


class RealForensicsDetector(Detector):
    """RealForensics on 25-frame clips of 150 x 150 face patches."""

    @property
    def source(self) -> str:
        return self.config.weights[0].name

    def convert_checkpoint(self) -> Path | None:
        """Keep the ``backbone`` and ``df_head`` weights, as ``eval.py`` does, as safetensors."""
        import torch

        if self.converted_is_current(self.source):
            return self.converted_path
        state = torch.load(self.weights_dir / self.source, map_location="cpu", weights_only=True)
        kept = {k: v for k, v in state.items() if k.startswith(("backbone.", "df_head."))}
        return self.save_converted(kept, self.source)

    def load(self, device: str) -> None:
        """Build upstream's network and load the converted weights, every key matching."""
        import torch

        assert self.upstream_dir is not None
        self.device = resolve_device(device)
        network = build_network(Path(self.upstream_dir))
        self.load_converted(network, self.source)
        self.model = network.eval().to(self.device)
        self._torch = torch
        shape = (1, 3, 1, 1, 1)
        self._mean = torch.tensor(_MEAN, dtype=torch.float32).view(shape).to(self.device)
        self._std = torch.tensor(_STD, dtype=torch.float32).view(shape).to(self.device)
        self.clip = int(self.config.input["clip_frames"])
        log.info("RealForensics loaded on %s", self.device)

    def score(self, inputs: np.ndarray) -> np.ndarray:  # type: ignore[override]
        """P(fake) per 25-frame clip of 150 x 150 RGB face patches."""
        torch = self._torch
        faces = np.asarray(inputs)
        starts = clip_starts(len(faces), self.clip)
        if not starts:
            raise PreprocessError(
                "too_short", f"{len(faces)} face frames, a clip needs {self.clip}"
            )
        top = int(round((faces.shape[1] - _CROP) / 2.0))
        left = int(round((faces.shape[2] - _CROP) / 2.0))
        scores = []
        for i in range(0, len(starts), _BATCH):
            clips = np.stack([faces[s : s + self.clip] for s in starts[i : i + _BATCH]])
            x = torch.from_numpy(clips).to(self.device).float() / 255  # B, T, H, W, C
            x = x[:, :, top : top + _CROP, left : left + _CROP].permute(0, 1, 4, 2, 3)
            b, t = x.shape[:2]
            x = torch.nn.functional.interpolate(
                x.reshape(b * t, 3, _CROP, _CROP),
                size=(_SIZE, _SIZE),
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )
            x = x.reshape(b, t, 3, _SIZE, _SIZE).permute(0, 2, 1, 3, 4)  # B, C, T, H, W
            x = (x - self._mean) / self._std
            with torch.inference_mode():
                logits = self.model.df_head(self.model.backbone(x))
            scores.append(torch.sigmoid(logits.float()).reshape(-1).cpu().numpy())
        return np.concatenate(scores)
