"""LAA-Net (Nguyen et al., CVPR 2024), the Localized Artifact Attention Network: an EfficientNet-B4
with an enhanced feature pyramid whose heatmap and consistency heads train it to attend to
blending artefacts, trained on self-blended pseudo-fakes of real faces only.

Upstream is under the SnT academic licence (research use only), so its code stays in
``third_party/`` and is imported from the pinned checkout (``models/_snt.py``). The network is
built the way ``scripts/test.py`` builds it: ``build_model(cfg.MODEL, MODELS)`` from the
``MODEL`` section of ``configs/efn4_fpn_sbi_adv.yaml``, read from the checkout. Importing
``models`` needs only PyTorch, so nothing is stubbed. Weights are loaded strictly.

Weights: the SBI checkpoint from the README's Dropbox folder
(``PoseEfficientNet_EFN_hm10_EFPN_..._8SBI_..._model_best.pth``, epoch 46), a ``torch.save``
dict of ``epoch``, ``state_dict`` and ``optimizer`` whose tensors are float64 (upstream tests in
float64). It loads with ``weights_only=True`` and is re-saved as safetensors without the
optimizer. ``inception_block`` is also registered as ``deconv_4.0``, so the state dict lists
its tensors twice, sharing storage; the conversion stores each name its own copy (safetensors
refuses shared storage, and its ``load_model`` can't tell this sharing apart from a stray key),
and ``load`` fills the network with a strict ``load_state_dict``, as upstream's
``load_pretrained`` does.

Input, as upstream's test dataset (``datasets/face_forensic_sbi.py``, split ``test``) takes it.
Upstream's own preprocessing is RetinaFace boxes widened by 25% of the box on each side, cut
from the frame, resized to 256 x 256 and saved as PNG (``package_utils/images_crop.py``);
32 frames per video, evenly spaced. At test time each crop loses 5 pixels on every side,
is warped to 384 x 384 (``get_affine_transform``, bilinear), divided by 255 and normalised with
mean and std 0.5 (``_snt.InputWarp``). The shared loader reproduces the first step with the
shared detector's box in place of RetinaFace's (the ``box`` crop at margin 1.5, stretched to
256 x 256 like upstream's resize), and ``InputWarp`` the second. Differences from upstream:

- the box comes from SCRFD, not RetinaFace, so the crop differs by a few pixels on most faces;
- upstream picks the face with the largest area and highest score; here the largest box;
- evenly spaced frames are rounded to the nearest index (upstream truncates), and a frame with
  no face is dropped instead of ending the video's list;
- dlib landmarks are not used at test time upstream either, so nothing replaces them.

Output: the ``cls`` head's logit, averaged with that of the horizontally flipped image (the
config's ``TEST.flip_test: True``, as ``scripts/test.py`` does), then a sigmoid: P(fake), since
training labels fake as 1. Upstream averages the logits of a video's frames before the sigmoid;
here every frame gets its own sigmoid and the harness averages those, which orders videos
slightly differently. Upstream runs in float64; this adapter runs in float32, and parity
(``scripts/parity_reference/laa_net.py``, float64) shows what that changes.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ._snt import InputWarp, checkout_imports, read_config
from .base import Detector, resolve_device

log = logging.getLogger(__name__)

_CONFIG = "configs/efn4_fpn_sbi_adv.yaml"
# LAA-Net's top-level packages, imported from the checkout root
_UPSTREAM_NAMES = ["models", "register"]
_MARGIN = 5  # crop_by_margin in the test branch of datasets/face_forensic_sbi.py
_BATCH = 8


def build_network(root: Path) -> tuple[Any, dict[str, Any]]:
    """Upstream's network built from the checkout's config, and that config.

    Raises:
        FileNotFoundError: if the checkout is missing.
    """
    config = read_config(root, _CONFIG)
    with checkout_imports(root, _UPSTREAM_NAMES):
        from models import MODELS, build_model

        # building prints "Constructing the heatmap Decoder!"
        network = build_model(config["MODEL"], MODELS)
    return network, config


class LAANetDetector(Detector):
    """LAA-Net (EfficientNet-B4, SBI weights) on 256 x 256 face crops, one score per frame."""

    @property
    def source(self) -> str:
        return self.config.weights[0].name

    def convert_checkpoint(self) -> Path | None:
        """Re-save the state dict as safetensors, each name with a copy of its own."""
        import torch

        if self.converted_is_current(self.source):
            return self.converted_path
        saved = torch.load(self.weights_dir / self.source, map_location="cpu", weights_only=True)
        # inception_block is also deconv_4.0, which shares storage
        state = {key: tensor.clone() for key, tensor in saved["state_dict"].items()}
        return self.save_converted(state, self.source)

    def _load_weights(self, network: Any) -> None:
        """A strict ``load_state_dict`` of the converted file, refusing a stale conversion."""
        from safetensors.torch import load_file

        if not self.converted_path.exists():
            raise RuntimeError(
                f"{self.converted_path} is missing; run scripts/setup_models.py {self.config.id}"
            )
        if not self.converted_is_current(self.source):
            raise RuntimeError(
                f"{self.converted_path} isn't a current conversion of {self.source}; re-run "
                f"scripts/setup_models.py {self.config.id}"
            )
        network.load_state_dict(load_file(str(self.converted_path)), strict=True)

    def load(self, device: str) -> None:
        """Build upstream's network and load the converted weights, every key matching."""
        import torch

        assert self.upstream_dir is not None
        self.device = resolve_device(device)
        root = Path(self.upstream_dir)
        network, config = build_network(root)
        self._load_weights(network)
        self.model = network.eval().to(self.device)
        self._torch = torch
        self._flip = bool(config["TEST"]["flip_test"])
        normalize = config["DATASET"]["TRANSFORM"]["normalize"]
        self._warp = InputWarp(
            root,
            "laa_net",
            size=int(config["DATASET"]["IMAGE_SIZE"][0]),
            margin=_MARGIN,
            mean=normalize["mean"],
            std=normalize["std"],
        )
        log.info("LAA-Net loaded on %s", self.device)

    def score(self, inputs: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        """P(fake) per face crop (``[h, w, 3]`` RGB uint8 each, 256 x 256 from the loader)."""
        torch = self._torch
        crops = list(inputs)
        scores = []
        for start in range(0, len(crops), _BATCH):
            batch = np.stack([self._warp(c) for c in crops[start : start + _BATCH]])
            x = torch.from_numpy(batch).to(device=self.device, dtype=torch.float32)
            with torch.inference_mode():
                logit = self.model(x)[0]["cls"]
                if self._flip:
                    logit = (logit + self.model(x.flip(dims=(3,)))[0]["cls"]) / 2
            scores.append(torch.sigmoid(logit).reshape(-1).float().cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0)
