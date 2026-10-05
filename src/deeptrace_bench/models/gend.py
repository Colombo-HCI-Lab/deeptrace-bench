"""GenD (WACV 2026): a foundation-model backbone with only the LayerNorms tuned, trained on
FF++ only. Two of the released backbones run here: CLIP ViT-L/14 (``gend``) and Perception
Encoder L/14 (``gend_pe_l``, built by timm); the DINOv3 one needs a gated Meta repo.

The adapter loads upstream's own model file, ``src/hf/modeling_gend.py`` (MIT), from the
pinned checkout rather than a copy, and builds it the way ``GenD.from_pretrained`` would,
with changes that don't alter the network:

- The CLIP backbone comes from the pinned local copy in the weights folder
  (``input.backbone_snapshot``), not from ``openai/clip-vit-large-patch14`` at whatever
  revision the Hub serves today. Upstream passes the backbone name straight to
  ``CLIPModel.from_pretrained``, so a local path works unchanged.
- The PE backbone is created by ``timm.create_model(..., pretrained=True)``, which would
  download ImageNet-free PE weights the checkpoint then replaces; while GenD is built, that
  call gets ``pretrained=False``. timm's built-in config for the model still sets the
  preprocessing (resize to 224, centre crop, normalisation), as upstream's does.
- The weights are loaded from ``model.safetensors`` with ``load_state_dict``, and every key
  must match.

Inputs are face crops (RGB uint8, any size) from the shared pipeline, aligned on five
landmarks at scale 1.3 and kept at native size, as GenD's README preprocesses. CLIP's
processor then resizes the shortest side to 224 (bicubic), centre-crops and normalises.
Output: softmax over two logits, index 1 is fake. A video's score is the mean of its frames',
which is also upstream's rule.

Training data: FF++ only. The paper's 14 benchmarks are test sets, FakeAVCeleb among them,
so they don't contaminate our runs. Parity with upstream is not checked yet.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator, Sequence
from types import ModuleType
from typing import Any

import numpy as np

from ._upstream import load_module
from .base import Detector, resolve_device

log = logging.getLogger(__name__)

# transformers looks the model class's module up in sys.modules, so the file loaded from the
# checkout is registered under a fixed name that can't collide with this repo's packages.
_MODULE_NAME = "dtb_upstream_gend_modeling"
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
_BATCH = 32


class GenDDetector(Detector):
    """GenD, CLIP-L/14 or PE-L/14 per the config."""

    def _module(self) -> ModuleType:
        assert self.upstream_dir is not None
        path = self.upstream_dir / "src" / "hf" / "modeling_gend.py"
        try:
            return load_module(path, _MODULE_NAME)
        except FileNotFoundError:
            msg = f"{path} is missing; run scripts/setup_models.py {self.config.id}"
            raise FileNotFoundError(msg) from None

    def load(self, device: str) -> None:
        """Build GenD on its pinned backbone and load its weights."""
        import torch
        from safetensors.torch import load_file

        self.device = resolve_device(device)
        module = self._module()
        config = module.GenDConfig.from_json_file(str(self.weights_dir / "config.json"))
        snapshot = self.config.input.get("backbone_snapshot")
        if snapshot:
            config.backbone = str(self.weights_dir / snapshot)
        with _timm_without_downloads():
            model = module.GenD(config)
        state = load_file(str(self.weights_dir / "model.safetensors"))
        result = model.load_state_dict(state, strict=False)
        if result.unexpected_keys or result.missing_keys:
            raise RuntimeError(
                f"GenD weights don't fit the network: missing {result.missing_keys[:5]}, "
                f"unexpected {result.unexpected_keys[:5]}"
            )
        processor = getattr(model.feature_extractor._preprocess, "image_processor", None)
        if processor is not None:  # CLIP; PE's transform comes from timm's own config
            self._check_processor(processor)
        self.model = model.eval().to(self.device)
        self._torch = torch
        log.info("GenD (%s) loaded on %s", config.backbone, self.device)

    @staticmethod
    def _check_processor(processor) -> None:  # noqa: ANN001
        # Upstream silently falls back to another model's processor if loading fails; make
        # sure the one in use is CLIP-L/14's.
        crop = processor.crop_size
        crop = crop.get("height") if isinstance(crop, dict) else crop
        if (
            crop != 224
            or not np.allclose(processor.image_mean, _CLIP_MEAN)
            or not np.allclose(processor.image_std, _CLIP_STD)
        ):
            raise RuntimeError(f"unexpected CLIP processor settings: {processor}")

    def score(self, inputs: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        """P(fake) per face crop."""
        from PIL import Image

        torch = self._torch
        preprocess = self.model.feature_extractor.preprocess
        scores = []
        for start in range(0, len(inputs), _BATCH):
            batch = [Image.fromarray(np.asarray(x)) for x in inputs[start : start + _BATCH]]
            pixels = torch.stack([preprocess(image) for image in batch]).to(self.device)
            with torch.inference_mode():
                logits = self.model(pixels)
            scores.append(logits.softmax(dim=-1)[:, 1].float().cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0)


@contextlib.contextmanager
def _timm_without_downloads() -> Iterator[None]:
    """Make ``timm.create_model`` build without pretrained weights while GenD is built."""
    import timm

    original = timm.create_model

    def create_model(*args: Any, **kwargs: Any) -> Any:
        kwargs["pretrained"] = False
        return original(*args, **kwargs)

    timm.create_model = create_model
    try:
        yield
    finally:
        timm.create_model = original
