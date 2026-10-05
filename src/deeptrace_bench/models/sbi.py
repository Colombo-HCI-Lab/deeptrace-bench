"""SBI (Self-Blended Images, Shiohara and Yamasaki, CVPR 2022), an EfficientNet-B4 trained
on self-blends of FF++ real videos. Research-only licence: the code stays in ``third_party/``.

The adapter builds upstream's own ``Detector`` from ``src/inference/model.py`` in the pinned
checkout. Its constructor calls ``EfficientNet.from_pretrained("efficientnet-b4",
advprop=True, num_classes=2)``, which downloads ImageNet weights the checkpoint then
replaces; while it runs, ``from_pretrained`` is pointed at ``from_name`` with the same
arguments, which builds the identical network without the download (``advprop`` only picks
which ImageNet file to fetch).

Weights: ``FFc23.tar`` from the authors' Google Drive, a ``torch.save`` dict whose ``"model"``
entry is the state dict (keys ``net.*``); ``FFraw.tar`` is the raw-quality twin, kept for
completeness. ``convert_checkpoint`` re-saves the c23 one as safetensors.

Input, as ``src/inference/inference_video.py``: RetinaFace boxes widened by an eighth of
their width and height on each side, resized to 380 x 380 (``align: box``, margin 1.25),
RGB scaled to [0, 1] with no mean or std. Output: softmax over two logits, index 1 fake.
Upstream keeps every face at least half the largest one's area and takes the max per frame,
then the mean over frames; the shared pipeline keeps the largest face per frame, which is
the same on single-face videos. Its boxes come from SCRFD rather than RetinaFace, so parity
compares the network on identical crops, and the crop rule is checked separately.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ._upstream import load_module
from .base import Detector, resolve_device

log = logging.getLogger(__name__)

_MODULE_NAME = "dtb_upstream_sbi_model"
_BATCH = 16


class SBIDetector(Detector):
    """SBI EfficientNet-B4."""

    @property
    def source(self) -> str:
        return self.config.input.get("checkpoint", self.config.weights[0].name)

    def _build(self) -> Any:
        """Upstream's ``Detector``, built without fetching ImageNet weights."""
        from efficientnet_pytorch import EfficientNet

        assert self.upstream_dir is not None
        path = Path(self.upstream_dir) / "src" / "inference" / "model.py"
        try:
            module = load_module(path, _MODULE_NAME)
        except FileNotFoundError:
            msg = f"{path} is missing; run scripts/setup_models.py {self.config.id}"
            raise FileNotFoundError(msg) from None
        with _no_imagenet_download(EfficientNet):
            return module.Detector()

    def convert_checkpoint(self) -> Path | None:
        """Re-save the checkpoint's ``"model"`` state dict as safetensors."""
        import torch

        if self.converted_is_current(self.source):
            return self.converted_path
        saved = torch.load(self.weights_dir / self.source, map_location="cpu", weights_only=True)
        return self.save_converted(saved["model"], self.source)

    def load(self, device: str) -> None:
        """Build the network and load the converted weights, every key matching."""
        import torch

        self.device = resolve_device(device)
        model = self._build()
        self.load_converted(model, self.source)
        self.model = model.eval().to(self.device)
        self._torch = torch
        log.info("SBI loaded on %s", self.device)

    def score(self, inputs: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        """P(fake) per face crop (380 x 380 RGB uint8)."""
        torch = self._torch
        scores = []
        for start in range(0, len(inputs), _BATCH):
            batch = np.stack(
                [np.asarray(c).transpose(2, 0, 1) for c in inputs[start : start + _BATCH]]
            )
            x = torch.from_numpy(batch).to(self.device).float() / 255
            with torch.inference_mode():
                logits = self.model(x)
            scores.append(logits.softmax(dim=1)[:, 1].float().cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0)


@contextlib.contextmanager
def _no_imagenet_download(efficientnet: Any) -> Iterator[None]:
    """Make ``EfficientNet.from_pretrained`` build the same network without its weights."""
    original = efficientnet.__dict__["from_pretrained"]  # the classmethod itself

    def from_name(
        model_name: str, weights_path: Any = None, advprop: bool = False, **kwargs: Any
    ) -> Any:
        kwargs.pop("in_channels", None)
        return efficientnet.from_name(model_name, **kwargs)

    efficientnet.from_pretrained = staticmethod(from_name)
    try:
        yield
    finally:
        efficientnet.from_pretrained = original
