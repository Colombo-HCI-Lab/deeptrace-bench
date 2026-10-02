"""SBI (Self-Blended Images), EfficientNet-B4 trained on self-blends of FF++ real videos.

Porting steps:

1. Weights: ``FFc23.tar`` (and ``FFraw.tar``) from Google Drive; the author says each is about
   135 MB. Use the c23 checkpoint for comparison with the FF++ c23 models.
2. The network is a plain EfficientNet-B4 (``efficientnet_pytorch`` upstream); rebuild it
   with ``timm`` or keep ``efficientnet_pytorch`` and remap keys in ``convert_checkpoint``.
3. Input: RetinaFace crops at 380 px. Upstream ``src/inference/inference_video.py`` takes the
   max over faces in a frame, then the mean over frames; use it for the parity check.
4. Upstream pins torch 1.9 (CUDA 11.1); install the right ``retinaface_pytorch`` package for
   parity (SBI #55 and #29 show people installing the wrong one).

Licence: research use only; code stays in ``third_party/``.
"""

from __future__ import annotations

from ._stub import PendingDetector


class SBIDetector(PendingDetector):
    """SBI EfficientNet-B4."""
