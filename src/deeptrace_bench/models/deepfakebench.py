"""Detectors from DeepfakeBench (SCLBD/DeepfakeBench, CC BY-NC 4.0), release v1.0.1 weights.

One adapter serves every frame-level DeepfakeBench detector whose checkpoint the release
ships; ``input.detector`` names it (``xception``, ``efficientnetb4``, ``ucf``, ``f3net``,
``spsl``), which is also the name of its ``training/config/detector/<name>.yaml``.

Upstream code is never copied (the licence is non-commercial). The adapter imports the one
detector file it needs from the pinned checkout. DeepfakeBench's packages import every
detector, the training losses and tensorboard on the way in, so while the detector loads,
``scoped_modules`` puts stand-ins in their place:

- ``metrics``, ``networks`` and ``detectors`` become empty packages over the checkout's
  folders, carrying the real ``metrics.registry`` registries, so only the files actually
  needed are imported.
- ``loss`` (training losses), ``metrics.base_metrics_class`` (training metrics) and
  ``torch.utils.tensorboard`` (not installed) become stubs; inference never calls them.

All of these names are removed again afterwards. The detector is built by upstream's own
class from upstream's own yaml. Its constructor loads ImageNet weights from a local path
before the checkpoint replaces every weight; that one ``torch.load`` gets a stand-in
holding only the first convolution's shape, so F3Net's 12-channel and SPSL's 4-channel
first-layer swaps still run as upstream wrote them. A strict load of the converted
checkpoint then sets every weight.

Inference follows DeepfakeBench's test transform: RGB crops (here from the shared face
pipeline, 256 px), cubic resize to the yaml's ``resolution``, scaled to [0, 1] and
normalised with the yaml's mean and std (0.5 each for all five). Output: softmax over two
logits; label 1 is fake for every detector. UCF scores with its common-forgery head and
appends to ``prob`` / ``label`` lists on every call, which are cleared per batch.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import types
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ._upstream import load_package, scoped_modules
from .base import Detector, resolve_device

log = logging.getLogger(__name__)

# input.detector -> the backbone module it registers its network from
_BACKBONES = {
    "xception": "xception",
    "efficientnetb4": "efficientnetb4",
    "ucf": "xception",
    "f3net": "xception",
    "spsl": "xception",
}
_GENERIC = ["metrics", "networks", "detectors", "loss"]
_BATCH = 32


class DeepfakeBenchDetector(Detector):
    """A DeepfakeBench frame-level detector, chosen by ``input.detector``."""

    @property
    def name(self) -> str:
        return self.config.input["detector"]

    @property
    def source(self) -> str:
        return self.config.weights[0].name

    @property
    def training_dir(self) -> Path:
        assert self.upstream_dir is not None
        path = Path(self.upstream_dir) / "training"
        if not path.is_dir():
            msg = f"{path} is missing; run scripts/setup_models.py {self.config.id}"
            raise FileNotFoundError(msg)
        return path

    def detector_config(self) -> dict[str, Any]:
        """Upstream's yaml for this detector, with ImageNet initialisation switched off."""
        import yaml

        path = self.training_dir / "config" / "detector" / f"{self.name}.yaml"
        config = yaml.safe_load(path.read_text())
        config["pretrained"] = None
        return config

    def _build(self) -> Any:
        """Upstream's detector network with placeholder weights."""
        import torch

        config = self.detector_config()
        training = self.training_dir
        stubs = {
            "loss": _module("loss", LOSSFUNC=_AnyLoss()),
            "torch.utils.tensorboard": _module("torch.utils.tensorboard", SummaryWriter=object),
        }
        with scoped_modules(stubs, purge=_GENERIC):
            load_package(training / "metrics", "metrics")
            registry = importlib.import_module("metrics.registry")
            importlib.sys.modules["metrics.base_metrics_class"] = _module(
                "metrics.base_metrics_class", calculate_metrics_for_train=None
            )
            load_package(training / "networks", "networks").BACKBONE = registry.BACKBONE
            load_package(training / "detectors", "detectors").DETECTOR = registry.DETECTOR
            importlib.import_module(f"networks.{_BACKBONES[self.name]}")
            importlib.import_module(f"detectors.{self.name}_detector")
            with _imagenet_stand_in(torch):
                return registry.DETECTOR[self.name](config)

    def convert_checkpoint(self) -> Path | None:
        """Re-save the release checkpoint (a state dict) as safetensors."""
        import torch

        if self.converted_is_current(self.source):
            return self.converted_path
        state = torch.load(self.weights_dir / self.source, map_location="cpu", weights_only=True)
        return self.save_converted(state, self.source)

    def load(self, device: str) -> None:
        """Build the detector and load its converted weights, every key matching."""
        import torch

        self.device = resolve_device(device)
        model = self._build()
        self.load_converted(model)
        config = self.detector_config()
        self.resolution = int(config["resolution"])
        self.mean = np.asarray(config["mean"], dtype=np.float32)
        self.std = np.asarray(config["std"], dtype=np.float32)
        self.model = model.eval().to(self.device)
        self._torch = torch
        log.info("DeepfakeBench %s loaded on %s", self.name, self.device)

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        """One RGB uint8 crop to a normalised ``[3, r, r]`` float array, as upstream's test."""
        import cv2

        image = np.asarray(crop)
        if image.shape[:2] != (self.resolution, self.resolution):
            size = (self.resolution, self.resolution)
            image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
        scaled = image.astype(np.float32) / 255.0
        return ((scaled - self.mean) / self.std).transpose(2, 0, 1)

    def score(self, inputs: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        """P(fake) per face crop."""
        torch = self._torch
        scores = []
        for start in range(0, len(inputs), _BATCH):
            batch = np.stack([self.preprocess(c) for c in inputs[start : start + _BATCH]])
            x = torch.from_numpy(batch).to(self.device)
            data = {"image": x, "label": torch.zeros(len(x), dtype=torch.long, device=self.device)}
            with torch.inference_mode():
                logits = self.model(data, inference=True)["cls"]
            for attr in ("prob", "label"):
                if isinstance(getattr(self.model, attr, None), list):
                    getattr(self.model, attr).clear()
            scores.append(logits.softmax(dim=-1)[:, 1].float().cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0)


def _module(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


class _NoLoss:
    """Stands in for a training loss; inference never calls it."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class _AnyLoss:
    """``LOSSFUNC[name]`` for any name."""

    def __getitem__(self, name: str) -> type:
        return _NoLoss


@contextlib.contextmanager
def _imagenet_stand_in(torch: Any) -> Iterator[None]:
    """Make ``torch.load`` return just a first-convolution weight while a detector is built.

    Upstream's constructors load ImageNet Xception weights from a local file and copy the
    first convolution into wider ones (F3Net, SPSL). The checkpoint replaces every weight
    afterwards, so zeros of the right shape keep upstream's construction code unchanged.
    """
    original = torch.load

    def stand_in(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"conv1.weight": torch.zeros(32, 3, 3, 3)}

    torch.load = stand_in
    try:
        yield
    finally:
        torch.load = original
