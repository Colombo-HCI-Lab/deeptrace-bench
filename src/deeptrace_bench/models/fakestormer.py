"""FakeSTormer (Nguyen et al., ICCV 2025), a spatio-temporal ViT-B/16 (divided space-time
attention, a temporal token) trained on video-level pseudo-fakes made from real videos only,
with heatmap and temporal-localisation heads beside the classifier.

Upstream is under the SnT academic licence (research use only), so its code stays in
``third_party/`` and is imported from the pinned checkout (``models/_snt.py``). The network is
built the way ``scripts/test.py`` builds it: ``build_model(cfg.MODEL, MODELS)`` from the
``MODEL`` section of ``configs/temporal/FakeSFormer_base_c23.yaml`` (a ``TopDownDetector`` with
a ``TimeViT`` backbone and a ``TopdownHeatmapSimpleHead``), read from the checkout. The model
takes ``NUM_FRAMES`` = 4 frames: the checkpoint's ``time_embed`` has 4 + 1 entries and the
temporal-localisation head 4 outputs, and with a temporal token ``TimeViT`` adds ``time_embed``
without resizing it, so any other length fails. Weights are loaded strictly.

Importing the network needs only PyTorch, einops and timm, but the repo's files import the
OpenMMLab stack at the top for training and initialisation helpers that inference never calls.
While they import, ``mmcv.cnn`` and ``mmcv.runner`` are stand-ins (``auto_fp16`` becomes a
no-op decorator; the others raise if called), ``models.utils`` is a stand-in for
``load_checkpoint``, and the packages ``models``, ``models.networks`` and its ``backbones``,
``heads`` and ``detectors`` folders are registered empty so their ``__init__`` files, which
import the Swin, ResNet and Xception variants and the neck, don't run. The three modules the
config needs are then imported by name.

Weights: the C23 checkpoint from the README's Dropbox folder
(``TopDownDetector_C23_ViTBase224_ST_hm100_..._normREAL_model_best.pth``, epoch 8), a
``torch.save`` dict of ``epoch``, ``state_dict`` and ``optimizer`` that loads with
``weights_only=True`` and is re-saved as safetensors without the optimizer. C23 (FF++ trained at
c23 compression) is the one used because the benchmark's videos are compressed.

Input, as upstream's test dataset (``datasets/fakesformer_sbi.py``, split ``test``) takes it.
Upstream's own preprocessing is RetinaFace boxes widened by 25% of the box on each side, cut
from the frame, resized to 256 x 256 and saved as PNG (``package_utils/images_crop.py``);
32 frames per video, evenly spaced; the sorted list is cut into 8 runs of 4 that do not overlap
(``preprocess/frame_groups.py``). At test time each frame loses 15 pixels on every side (the
dataset's ``margin_ = 15`` "for the others"; its comment gives 0 for DFW, 5 for DFDCP and 13
for DFD, per dataset), is warped to 224 x 224 (``get_affine_transform``, bilinear), divided by
255 and normalised with ImageNet's mean and std, in float64 and then cast to float32 as
``test.py`` does (``_snt.InputWarp``). Differences from upstream:

- the box comes from the shared SCRFD detector, not RetinaFace (the ``box`` crop at margin 1.5,
  stretched to 256 x 256 like upstream's resize), so the crop differs by a few pixels;
- evenly spaced frames are rounded to the nearest index (upstream truncates), and a frame with
  no face is dropped before the runs are cut, where upstream's list has only frames with a face
  anyway;
- the 15 pixel margin is upstream's default, not tuned per dataset.

Output: the ``cls`` head's logit (``flip_test`` is off in the config) through a sigmoid: P(fake),
one score per run of 4 frames. Upstream averages the logits of a video's runs before the sigmoid
(``bin_calculate_auc_ap_ar``); here every run gets its own sigmoid and the harness averages
those, which orders videos slightly differently.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ._snt import InputWarp, checkout_imports, read_config, stub_module, unused
from ._upstream import load_package
from .base import Detector, resolve_device

log = logging.getLogger(__name__)

_CONFIG = "configs/temporal/FakeSFormer_base_c23.yaml"
# FakeSTormer's top-level packages, imported from the checkout root
_UPSTREAM_NAMES = ["models", "register", "losses"]
_PACKAGES = {
    "models": "models",
    "models.networks": "models/networks",
    "models.networks.backbones": "models/networks/backbones",
    "models.networks.heads": "models/networks/heads",
    "models.networks.detectors": "models/networks/detectors",
}
_MARGIN = 15  # crop_by_margin in the test branch of datasets/fakesformer_sbi.py
_BATCH = 4


def _stubs() -> dict[str, Any]:
    return {
        "mmcv": stub_module("mmcv"),
        "mmcv.cnn": stub_module(
            "mmcv.cnn",
            build_upsample_layer=unused,
            build_conv_layer=unused,
            build_norm_layer=unused,
            constant_init=unused,
            normal_init=unused,
        ),
        "mmcv.runner": stub_module("mmcv.runner", auto_fp16=lambda *a, **k: lambda f: f),
        "models.utils": stub_module("models.utils", load_checkpoint=unused),
    }


def build_network(root: Path) -> tuple[Any, dict[str, Any]]:
    """Upstream's network built from the checkout's config, and that config.

    Raises:
        FileNotFoundError: if the checkout is missing.
    """
    config = read_config(root, _CONFIG)
    with checkout_imports(root, _UPSTREAM_NAMES, _stubs()):
        for name, folder in _PACKAGES.items():
            package = load_package(root / folder, name)
            # upstream's Registry calls inspect.getmodule, which fails on a module with no file
            package.__file__ = str(root / folder / "__init__.py")
        # importing registers the classes the config names
        import models.networks.backbones.vit  # noqa: F401
        import models.networks.detectors.topdown  # noqa: F401
        import models.networks.heads.hm_simple_head  # noqa: F401
        from models.builder import MODELS, build_model

        network = build_model(config["MODEL"], MODELS)
    return network, config


class FakeSTormerDetector(Detector):
    """FakeSTormer (C23) on runs of four 256 x 256 face crops, one score per run."""

    @property
    def source(self) -> str:
        return self.config.weights[0].name

    def convert_checkpoint(self) -> Path | None:
        """Re-save the state dict as safetensors."""
        import torch

        if self.converted_is_current(self.source):
            return self.converted_path
        saved = torch.load(self.weights_dir / self.source, map_location="cpu", weights_only=True)
        return self.save_converted(saved["state_dict"], self.source)

    def load(self, device: str) -> None:
        """Build upstream's network and load the converted weights, every key matching."""
        import torch

        assert self.upstream_dir is not None
        self.device = resolve_device(device)
        root = Path(self.upstream_dir)
        network, config = build_network(root)
        self.load_converted(network, self.source)
        self.model = network.eval().to(self.device)
        self._torch = torch
        if config["TEST"]["flip_test"]:
            raise NotImplementedError("the flip test is off in the config this adapter follows")
        normalize = config["DATASET"]["TRANSFORM"]["normalize"]
        self._warp = InputWarp(
            root,
            "fakestormer",
            size=int(config["DATASET"]["IMAGE_SIZE"][0]),
            margin=_MARGIN,
            mean=normalize["mean"],
            std=normalize["std"],
        )
        self.group_frames = int(config["MODEL"]["backbone"]["num_frames"])
        log.info("FakeSTormer loaded on %s", self.device)

    def score(self, inputs: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        """P(fake) per run (``[4, 256, 256, 3]`` RGB uint8 crops each)."""
        torch = self._torch
        runs = np.asarray(inputs)
        if runs.ndim != 5 or runs.shape[1] != self.group_frames:
            raise ValueError(f"expected runs of {self.group_frames} crops, got {runs.shape}")
        scores = []
        for start in range(0, len(runs), _BATCH):
            # [batch, 3, frames, h, w]: each frame warped, then channels before time
            batch = np.stack(
                [
                    np.stack([self._warp(frame) for frame in run]).transpose(1, 0, 2, 3)
                    for run in runs[start : start + _BATCH]
                ]
            )
            x = torch.from_numpy(batch).to(device=self.device, dtype=torch.float32)
            with torch.inference_mode():
                logit = self.model(x)[0]["cls"]
            scores.append(torch.sigmoid(logit).reshape(-1).float().cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0)
