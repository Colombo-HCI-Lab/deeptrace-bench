"""AASIST and AASIST-L (clovaai/aasist), trained on ASVspoof 2019 LA train.

Porting steps (the easiest audio model):

1. Weights are committed in the upstream repo (``models/weights/AASIST.pth``, 1.3 MB, and
   ``AASIST-L.pth``, 0.4 MB). The variant comes from ``input.variant`` in the config.
2. ``models/AASIST.py`` is plain PyTorch and MIT-licensed; it can be loaded from the checkout
   or copied in. Drop the unmaintained ``torchcontrib`` import (only used for SWA training).
3. Input: 16 kHz mono windows of 64,600 samples (``preprocess.audio.segment``).
4. Score direction: the model outputs two logits. Upstream's ``evaluation.py`` uses the bona
   fide node, and a user reports the order looks inverted with the shipped weights (aasist
   #17, 2026-08-30). Settle it on ASVspoof 2019 LA eval (AUC must come out above 0.5)
   before any South Asian run.
"""

from __future__ import annotations

from ._stub import PendingDetector


class AASISTDetector(PendingDetector):
    """AASIST / AASIST-L."""
