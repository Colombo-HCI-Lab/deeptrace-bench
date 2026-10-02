"""Xception and EfficientNet-B4 from DeepfakeBench (FF++ c23 checkpoints, release v1.0.1).

Porting steps:

1. Load only ``training/detectors/xception_detector.py`` / ``efficientnetb4_detector.py`` and
   their backbone files from the checkout with ``importlib``; importing the ``detectors``
   package pulls in every detector and its dependencies.
2. Upstream pins Python 3.7 and torch 1.12 (no H100 support). Expect ``torch.load`` to need
   ``weights_only=False`` once for conversion, plus removed numpy aliases (``np.float``).
3. ``convert_checkpoint``: strip ``module.`` prefixes, save ``model.safetensors``.
4. Input: aligned face crops, 256 px, 32 frames per video (DeepfakeBench uses dlib's 81-point
   landmarks; we use the shared detector and a matching ``CropSpec`` for comparison runs,
   and DeepfakeBench's own pipeline for the parity check).
5. Output: softmax over two logits; P(fake) is index 1. Check direction on FF++ test.

Known issues: published weights don't reproduce the README table for some users
(DeepfakeBench #109, #159), and EfficientNet-B4 has an architecture mismatch report (#79).
"""

from __future__ import annotations

from ._stub import PendingDetector


class XceptionDetector(PendingDetector):
    """DeepfakeBench Xception."""


class EfficientNetB4Detector(PendingDetector):
    """DeepfakeBench EfficientNet-B4."""
