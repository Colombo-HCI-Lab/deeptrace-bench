"""DeMamba (Chen et al., arXiv 2405.19707, GenVideo): a detector of AI-generated video, X-CLIP-B/16
frame features fine-tuned end to end and read across time by a bidirectional Mamba block.

The authors released code but no weights (DeMamba issue #16). The checkpoint here is
``Ahsanz/DeMamba-GenVideo100K`` (Apache-2.0), trained by the Counterfactual-Deepfake-Bench team
(WVU) on GenVideo-100K with the authors' code, and published with a self-contained evaluator,
``eval_demamba_standalone.py``. That file is the upstream: the adapter imports its
``XCLIP_DeMamba`` (pure-PyTorch Mamba, so no mamba-ssm) and its ``load_checkpoint`` from the
pinned Hub checkout, never a copy.

Weights: ``best_acc.pth`` (epoch 15 of 30), which holds the fine-tuned X-CLIP tower too, loaded
once through upstream's ``load_checkpoint`` (it strips ``module.`` and maps transformers 4 and 5
key names) and re-saved as safetensors with the model's own keys.

Input: one clip of up to 8 whole frames (``preprocess.frame_clips``, sampling ``demamba``), 224
px RGB uint8. As upstream: ImageNet mean and std on 0-255 values, missing frames padded with
zeros after normalising. Output: the sigmoid of the one logit, P(fake), per clip; upstream's
decision threshold is 0.5. Upstream's evaluator runs bf16 on GPU by default; here fp32.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from types import ModuleType

import numpy as np

from ._upstream import load_module
from .base import Detector, resolve_device

log = logging.getLogger(__name__)

_SOURCE = "best_acc.pth"
_MODULE_NAME = "dtb_upstream_demamba_standalone"
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_FRAMES = 8


class DeMambaDetector(Detector):
    """XCLIP_DeMamba with the GenVideo-100K checkpoint."""

    def _module(self) -> ModuleType:
        assert self.upstream_dir is not None
        return load_module(self.upstream_dir / "eval_demamba_standalone.py", _MODULE_NAME)

    def convert_checkpoint(self):
        """Load the checkpoint with upstream's loader and re-save the model's state dict."""
        if self.converted_is_current(_SOURCE):
            return self.converted_path
        upstream = self._module()
        model = upstream.XCLIP_DeMamba()
        upstream.load_checkpoint(model, str(self.weights_dir / _SOURCE))
        return self.save_converted(model.state_dict(), _SOURCE)

    def load(self, device: str) -> None:
        """Build upstream's model and load the converted weights, every key matching."""
        import torch

        self.device = resolve_device(device)
        model = self._module().XCLIP_DeMamba()
        self.load_converted(model, _SOURCE)
        self.model = model.eval().to(self.device)
        self._torch = torch
        shape = (1, 1, 3, 1, 1)
        self._mean = (torch.tensor(_MEAN) * 255).view(shape).to(self.device)
        self._std = (torch.tensor(_STD) * 255).view(shape).to(self.device)
        log.info("DeMamba loaded on %s", self.device)

    def score(self, inputs: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        """P(fake) per clip (``[frames, 224, 224, 3]`` RGB uint8 each)."""
        torch = self._torch
        scores = []
        for clip in inputs:
            x = torch.from_numpy(np.asarray(clip)).to(self.device)[None]
            x = x.permute(0, 1, 4, 2, 3).float().sub(self._mean).div(self._std)
            if x.shape[1] < _FRAMES:
                pad = x.new_zeros((1, _FRAMES - x.shape[1], *x.shape[2:]))
                x = torch.cat([x, pad], dim=1)
            with torch.inference_mode():
                scores.append(self.model(x).float()[:, 0].sigmoid().cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0)
