"""XLS-R 300M + AASIST (Tak et al. 2022, TakHemlata/SSL_Anti-spoofing), trained on ASVspoof
2019 LA train.

Porting steps (the hardest wave 1 model):

1. Upstream pins Python 3.7, torch 1.8.1 and a vendored fairseq, none of which run on
   current GPUs. We drop fairseq: build the front end from ``facebook/wav2vec2-xls-r-300m``
   in transformers and remap the fairseq state-dict keys of the fine-tuned checkpoint in
   ``convert_checkpoint``. The AntiDeepfake maintainers warn the layer names differ, so the
   remap needs a key-by-key check, then the parity check against upstream.
2. Weights: ``LA_model.pth`` from the authors' Google Drive folder (also
   ``Best_LA_model_for_DF.pth``). Load with ``strict=False`` (upstream #1: a pre-activation
   block was removed in a cleanup) and list the missing keys explicitly.
3. Input: 16 kHz mono windows of 64,600 samples.
4. Pretraining: XLS-R saw Common Voice, BABEL, MLS, VoxPopuli and VoxLingua107, so evalsets
   whose real speech comes from those corpora get a ``pretrain-overlap`` warning.
"""

from __future__ import annotations

from ._stub import PendingDetector


class XLSRAASISTDetector(PendingDetector):
    """XLS-R + AASIST."""
