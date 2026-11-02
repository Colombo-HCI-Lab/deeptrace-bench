"""UnivFD (Ojha, Li and Lee, CVPR 2023, WisconsinAIVision/UniversalFakeDetect, MIT): one linear
layer on the frozen image features of CLIP ViT-L/14, fitted to ProGAN images.

Upstream's ``CLIPModel`` wraps OpenAI's ``clip`` package and only adds that layer, so the adapter
runs the same backbone through transformers, from the pinned ``openai/clip-vit-large-patch14``
snapshot GenD also uses, rather than download a second copy of the weights. The parity check
against upstream's own code and OpenAI's checkpoint is what shows the two compute the same thing.

Weights: ``fc_weights.pth``, the layer's state dict (``weight``, ``bias``), re-saved as
safetensors.

Input: 224 x 224 face crops from the shared pipeline. Upstream's test transform is a centre crop
to 224 and CLIP's normalisation, with no resize, on whole images; a face crop of that size is
the closest fit to what a frame detector sees here, and the crop is a no-op. Output: the
sigmoid of the layer's one logit, P(fake), per frame.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .base import Detector, resolve_device

log = logging.getLogger(__name__)

_SOURCE = "fc_weights.pth"
_SNAPSHOT = "clip-vit-large-patch14"
_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)
_BATCH = 32


class UnivFDDetector(Detector):
    """A linear probe on CLIP ViT-L/14 image features."""

    def convert_checkpoint(self) -> Path | None:
        """Re-save the layer as safetensors."""
        import torch

        if self.converted_is_current(_SOURCE):
            return self.converted_path
        state = torch.load(self.weights_dir / _SOURCE, map_location="cpu", weights_only=True)
        return self.save_converted(state, _SOURCE)

    def load(self, device: str) -> None:
        """Load the pinned CLIP backbone and the fitted layer."""
        import torch
        from transformers import CLIPVisionModelWithProjection

        self.device = resolve_device(device)
        fc = torch.nn.Linear(768, 1)
        self.load_converted(fc, _SOURCE)
        clip = CLIPVisionModelWithProjection.from_pretrained(
            self.weights_dir / _SNAPSHOT, attn_implementation="eager"
        )
        self.clip = clip.eval().to(self.device)
        self.fc = fc.eval().to(self.device)
        self._torch = torch
        shape = (1, 3, 1, 1)
        self._mean = torch.tensor(_MEAN).view(shape).to(self.device)
        self._std = torch.tensor(_STD).view(shape).to(self.device)
        log.info("UnivFD loaded on %s", self.device)

    def score(self, inputs: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        """P(fake) per 224 x 224 face crop (RGB uint8)."""
        torch = self._torch
        crops = np.asarray(inputs)
        scores = []
        for start in range(0, len(crops), _BATCH):
            batch = torch.from_numpy(crops[start : start + _BATCH]).to(self.device)
            x = batch.permute(0, 3, 1, 2).float().div(255).sub(self._mean).div(self._std)
            with torch.inference_mode():
                features = self.clip(pixel_values=x).image_embeds
                scores.append(self.fc(features).sigmoid().reshape(-1).float().cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0)
