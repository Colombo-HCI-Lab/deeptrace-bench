"""GenD (WACV 2026), CLIP ViT-L/14 with only the LayerNorms tuned, trained on FF++ only.

Porting steps (the easiest model; upstream already runs on torch 2.8 and transformers 4.56):

1. Weights: ``yermandy/GenD_CLIP_L_14`` on Hugging Face (MIT, ungated, ``model.safetensors``).
   The PE and DINOv3 variants exist too; DINOv3's backbone is gated by Meta.
2. ``GenD.from_pretrained(...)`` plus ``model.feature_extractor.preprocess`` from the
   checkout. MIT, so the model file may be copied in if importing gets awkward.
3. Input: face crops from upstream ``detector.py`` (RetinaFace ONNX, landmark alignment at
   scale 1.3). There's no CLI that scores a video, so frames are sampled by the harness.
4. Training data: FF++ only. The "14 benchmarks" in the paper are its test suite, including
   FakeAVCeleb as a test set, which doesn't contaminate our FakeAVCeleb runs.
"""

from __future__ import annotations

from ._stub import PendingDetector


class GenDDetector(PendingDetector):
    """GenD CLIP-L/14."""
