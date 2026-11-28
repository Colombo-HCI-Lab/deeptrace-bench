"""D3 (Zheng et al., ICCV 2025, arXiv 2508.00701, Zig-HS/D3, MIT): training-free detection of
AI-generated video from second-order features. Nothing is trained: frozen X-CLIP-B/16 embeds
16 frames, the L2 distances between consecutive embeddings are differenced once more, and the
standard deviation of those second differences is the statistic. Real video has the larger
spread (upstream's ``eval.py`` scores AP with real as the positive class).

The adapter runs upstream's ``D3_model.forward`` from the pinned checkout with the official
setting (``--encoder XCLIP-16 --loss l2``). Upstream's ``__init__`` loads
``microsoft/xclip-base-patch16`` from the Hub at whatever revision is served; here the model is
built without it and given the pinned local snapshot, checked to load every vision key.

Input: one clip of 16 whole frames at 8 per second from the first 3 s (8 when fewer than 16
exist), from ``preprocess.frame_clips`` (sampling ``rate``). Upstream reads its frames with
``cv2.imread`` and never converts them, so X-CLIP sees BGR with ImageNet's RGB mean and std
applied in that channel order; the adapter flips the RGB frames and does the same.
Output: ``1 / (1 + std)`` per clip, higher for fakes and monotone in upstream's statistic,
so every rank metric is upstream's; there is no official threshold.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from types import ModuleType

import numpy as np

from ._upstream import load_module
from .base import Detector, resolve_device

log = logging.getLogger(__name__)

_SNAPSHOT = "xclip-base-patch16"
_MODULE_NAME = "dtb_upstream_d3_model"
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class D3Detector(Detector):
    """D3 with the X-CLIP-B/16 encoder and the L2 distance."""

    def _module(self) -> ModuleType:
        assert self.upstream_dir is not None
        return load_module(self.upstream_dir / "models" / "D3_model.py", _MODULE_NAME)

    def load(self, device: str) -> None:
        """Build upstream's model around the pinned X-CLIP vision tower."""
        import torch
        from transformers import XCLIPVisionModel

        self.device = resolve_device(device)
        encoder, info = XCLIPVisionModel.from_pretrained(
            self.weights_dir / _SNAPSHOT, output_loading_info=True
        )
        missing = [k for k in info["missing_keys"] if not k.endswith("position_ids")]
        if missing:
            raise RuntimeError(f"X-CLIP vision tower left keys at random: {missing[:5]}")
        upstream = self._module().D3_model
        model = upstream.__new__(upstream)  # upstream's __init__ would fetch from the Hub
        torch.nn.Module.__init__(model)
        model.loss_type, model.encoder_type, model.encoder = "l2", "XCLIP-16", encoder
        self.model = model.eval().to(self.device)
        self._torch = torch
        shape = (1, 1, 3, 1, 1)
        self._mean = (torch.tensor(_MEAN) * 255).view(shape).to(self.device)
        self._std = (torch.tensor(_STD) * 255).view(shape).to(self.device)
        log.info("D3 loaded on %s", self.device)

    def score(self, inputs: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        """``1 / (1 + std)`` per clip (``[frames, 224, 224, 3]`` RGB uint8 each)."""
        torch = self._torch
        scores = []
        for clip in inputs:
            bgr = np.ascontiguousarray(np.asarray(clip)[..., ::-1])
            x = torch.from_numpy(bgr).to(self.device)[None]
            x = x.permute(0, 1, 4, 2, 3).float().sub(self._mean).div(self._std)
            with torch.inference_mode():
                _, _, std = self.model(x)
            scores.append((1.0 / (1.0 + std.float())).cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0)
