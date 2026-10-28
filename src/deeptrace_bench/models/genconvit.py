"""GenConViT (Wodajo et al., 2023), its autoencoder (ED) branch: a ConvNeXt-Tiny that scores
each face and its autoencoder reconstruction together.

Upstream's MIT code is imported from the pinned checkout (``model/`` as a package, so
``genconvit_ed.py``'s relative imports resolve), never copied. The network is upstream's own
``GenConViTED``, built with ``pretrained=False`` so timm fetches no ImageNet weights (the
checkpoint replaces them). Upstream attaches a Swin-Tiny ``HybridEmbed`` as the ConvNeXt's
``patch_embed``, an attribute ConvNeXt's forward never calls, so the Swin weights sit in the
checkpoint without touching any score; they are still loaded strictly.

Only the ED branch is scored. Upstream's default runs it beside the VAE branch, but the VAE's
encoder samples its latent with ``torch.randn_like`` at test time too (``reparameterize`` has
no eval path), so its scores change from run to run; ``--net ed`` is upstream's own
deterministic option.

Weights: ``genconvit_ed_inference.pth`` from the Hugging Face repo (CC BY-NC 4.0), a plain
state dict saved with timm 0.6.5. timm 1.x renamed Swin's keys (the head became
``head.fc``, each stage's downsample moved to the next stage, the attention index and mask
became non-persistent); ``convert_checkpoint`` runs timm's own ``checkpoint_filter_fn`` for
Swin over both copies of the embedder, then re-saves as safetensors. The ConvNeXt keys are
unchanged.

Input, as upstream's ``pred_func.py``: 224 x 224 RGB face crops scaled to [0, 1] and
normalised with ImageNet's mean and std. Upstream crops dlib's tight boxes
(``face_recognition``) and resizes with ``INTER_AREA``; here the shared detector's box is
stretched to 224 (``align: box``, margin 1.0). Output: two logits whose class 0 is fake
(``ImageFolder``'s alphabetical order; upstream's ``real_or_fake`` reads index 0 as fake);
P(fake) is the sigmoid of logit 0, the value upstream averages over frames when it calls a
video fake.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ._upstream import load_package
from .base import Detector, resolve_device

log = logging.getLogger(__name__)

_PACKAGE = "dtb_upstream_genconvit_model"
# The two copies of the Swin embedder in the ED state dict.
_SWIN_PREFIXES = ("embedder.", "backbone.patch_embed.backbone.")
_MEAN = np.array([0.485, 0.456, 0.406])
_STD = np.array([0.229, 0.224, 0.225])
_BATCH = 32


def build_ed(root: Path) -> Any:
    """Upstream's ``GenConViTED`` from the checkout, without fetching ImageNet weights."""
    import yaml

    model_dir = root / "model"
    if not (model_dir / "genconvit_ed.py").is_file():
        raise FileNotFoundError(f"{model_dir} is missing; run scripts/setup_models.py genconvit")
    load_package(model_dir, _PACKAGE)
    module = importlib.import_module(f"{_PACKAGE}.genconvit_ed")
    config = yaml.safe_load((model_dir / "config.yaml").read_text())
    return module.GenConViTED(config, pretrained=False)


def convert_swin_keys(state: dict[str, Any], network: Any) -> dict[str, Any]:
    """The ED state dict with both Swin copies' keys moved to timm 1.x's layout."""
    from timm.models.swin_transformer import checkpoint_filter_fn

    swins = {
        "embedder.": network.embedder,
        "backbone.patch_embed.backbone.": network.backbone.patch_embed.backbone,
    }
    out = {k: v for k, v in state.items() if not k.startswith(_SWIN_PREFIXES)}
    for prefix, module in swins.items():
        part = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        out.update({prefix + k: v for k, v in checkpoint_filter_fn(part, module).items()})
    return out


class GenConViTDetector(Detector):
    """GenConViT's ED branch on 224 x 224 face crops."""

    @property
    def source(self) -> str:
        return self.config.weights[0].name

    def convert_checkpoint(self) -> Path | None:
        """Move the Swin keys to timm 1.x's layout and re-save as safetensors.

        The embedder is one module under two names, so its tensors appear twice; both copies
        must agree, and only the names ``safetensors.torch.load_model`` keeps are saved.
        """
        import torch
        from safetensors.torch import _remove_duplicate_names

        if self.converted_is_current(self.source):
            return self.converted_path
        assert self.upstream_dir is not None
        state = torch.load(self.weights_dir / self.source, map_location="cpu", weights_only=True)
        network = build_ed(Path(self.upstream_dir))
        state = convert_swin_keys(state, network)
        network.load_state_dict(state, strict=True)
        for kept, dropped in _remove_duplicate_names(network.state_dict()).items():
            for name in dropped:
                if not torch.equal(state[name], state[kept]):
                    raise ValueError(f"{self.source}: {name} and {kept} should be one tensor")
                del state[name]
        return self.save_converted(state, self.source)

    def load(self, device: str) -> None:
        """Build upstream's network and load the converted weights, every key matching."""
        import torch

        assert self.upstream_dir is not None
        self.device = resolve_device(device)
        network = build_ed(Path(self.upstream_dir))
        self.load_converted(network, self.source)
        self.model = network.eval().to(self.device)
        self._torch = torch
        shape = (1, 3, 1, 1)
        self._mean = torch.tensor(_MEAN, dtype=torch.float32).view(shape).to(self.device)
        self._std = torch.tensor(_STD, dtype=torch.float32).view(shape).to(self.device)
        log.info("GenConViT (ED) loaded on %s", self.device)

    def score(self, inputs: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        """P(fake) per face crop (224 x 224 RGB uint8)."""
        torch = self._torch
        scores = []
        for start in range(0, len(inputs), _BATCH):
            batch = np.stack([np.asarray(c) for c in inputs[start : start + _BATCH]])
            x = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).float() / 255
            x = (x - self._mean) / self._std
            with torch.inference_mode():
                logits = self.model(x)
            scores.append(torch.sigmoid(logits[:, 0]).float().cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0)
