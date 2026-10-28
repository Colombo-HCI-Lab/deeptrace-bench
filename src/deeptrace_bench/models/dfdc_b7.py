"""The DFDC winner (Selim Seferbekov, Kaggle Deepfake Detection Challenge 2020): seven
EfficientNet-B7 face classifiers trained on DFDC, averaged.

Upstream is MIT. The network is upstream's own ``DeepFakeClassifier`` from
``training/zoo/classifiers.py`` in the pinned checkout, with ``encoder="tf_efficientnet_b7_ns"``.
Its encoder table builds timm's ``tf_efficientnet_b7_ns`` with ``pretrained=True``; while the
file imports, ``timm.models.efficientnet`` is a stand-in whose EfficientNet constructors pass
``pretrained=False``, so nothing is downloaded (each checkpoint replaces every weight). timm 1.x
keeps the ``tf_efficientnet_b7_ns`` name and its key layout.

Weights: the seven checkpoints ``predict_submission.sh`` ensembles, each a ``torch.save`` dict
whose ``"state_dict"`` has ``module.``-prefixed keys, beside ``epoch`` and ``bce_best`` (a numpy
scalar, so the safe unpickler is told about numpy's scalar and dtype). Each is converted to its
own ``<name>.safetensors``, with the source's hash in the file's metadata, and loaded strictly.

Input, as ``kernel_utils.py``: MTCNN boxes widened by a third of their width and height on
each side (here the shared detector's box), resized so the longer side is 380 (``INTER_AREA``
down, ``INTER_CUBIC`` up) and centred on a black 380 x 380 square: the shared ``letterbox``
crop, margin 5/3, with upstream's two helpers. Then scaled to [0, 1] and normalised with
ImageNet's mean and std. Output: each
model's sigmoid; a frame's P(fake) is the mean over the seven. Upstream runs the models in
fp16 and summarises each model's frames with its "confident strategy" before averaging; here
everything is fp32, per the eval config, and frames are averaged by the shared rule.
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import os
import re
import types
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ._upstream import scoped_modules
from .base import Detector, resolve_device

log = logging.getLogger(__name__)

_ENCODER = "tf_efficientnet_b7_ns"
_EFFICIENTNETS = [f"tf_efficientnet_b{i}_ns" for i in range(2, 8)]
_MEAN = np.array([0.485, 0.456, 0.406])
_STD = np.array([0.229, 0.224, 0.225])
_BATCH = 16
_CONVERSION = "1"


@contextlib.contextmanager
def _no_imagenet_downloads() -> Iterator[None]:
    """A ``timm.models.efficientnet`` whose constructors never fetch pretrained weights."""
    import timm.models.efficientnet as real

    def offline(name: str) -> Any:
        build = getattr(real, name)
        return lambda pretrained=False, **kwargs: build(pretrained=False, **kwargs)

    stand_in = types.ModuleType("timm.models.efficientnet")
    stand_in.__dict__.update({name: offline(name) for name in _EFFICIENTNETS})
    with scoped_modules({"timm.models.efficientnet": stand_in}):
        yield


def build_classifier(root: Path) -> Any:
    """Upstream's ``DeepFakeClassifier`` with a B7 encoder, built without downloads."""
    path = root / "training" / "zoo" / "classifiers.py"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run scripts/setup_models.py dfdc_b7")
    name = "dtb_upstream_dfdc_classifiers"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with _no_imagenet_downloads():
        spec.loader.exec_module(module)
    return module.DeepFakeClassifier(encoder=_ENCODER)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    """The state dict of one checkpoint, ``module.`` stripped, with the safe unpickler."""
    import torch
    from numpy._core.multiarray import scalar

    allowed = [
        (scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        (type(np.dtype("float64")), "numpy.dtype[float64]"),
    ]
    with torch.serialization.safe_globals(allowed):
        saved = torch.load(path, map_location="cpu", weights_only=True)
    state = saved.get("state_dict", saved)
    return {re.sub("^module.", "", k): v for k, v in state.items()}


class DFDCWinnerDetector(Detector):
    """The seven-model EfficientNet-B7 ensemble, averaged per frame."""

    @property
    def sources(self) -> list[str]:
        return [w.name for w in self.config.weights]

    def _converted(self, source: str) -> Path:
        return self.weights_dir / f"{source}.safetensors"

    def _current(self, source: str) -> bool:
        from safetensors import safe_open

        path = self._converted(source)
        if not path.exists():
            return False
        with safe_open(str(path), "pt") as f:
            meta = f.metadata() or {}
        expected = (source, self._pinned_hash(source), _CONVERSION)
        found = (meta.get("source"), meta.get("source_sha256"), meta.get("conversion_version"))
        return found == expected

    def convert_checkpoint(self) -> Path | None:
        """Re-save each checkpoint's state dict as its own safetensors file."""
        from safetensors.torch import save_file

        for source in self.sources:
            if self._current(source):
                continue
            digest = self._pinned_hash(source)
            if digest is None:
                raise RuntimeError(
                    f"dfdc_b7/{source} has no recorded hash; run scripts/setup_models.py "
                    "dfdc_b7 --record first"
                )
            state = {
                k: v.contiguous() for k, v in _load_checkpoint(self.weights_dir / source).items()
            }
            part = self._converted(source).with_suffix(".part")
            metadata = {
                "source": source,
                "source_sha256": digest,
                "conversion_version": _CONVERSION,
            }
            save_file(state, str(part), metadata=metadata)
            os.replace(part, self._converted(source))
        return self._converted(self.sources[0])

    def load(self, device: str) -> None:
        """Build the network once per checkpoint and load each strictly."""
        import torch
        from safetensors.torch import load_model

        assert self.upstream_dir is not None
        self.device = resolve_device(device)
        self.models = []
        for source in self.sources:
            if not self._current(source):
                raise RuntimeError(
                    f"{self._converted(source)} is missing or stale; run "
                    "scripts/setup_models.py dfdc_b7"
                )
            network = build_classifier(Path(self.upstream_dir))
            load_model(network, str(self._converted(source)), strict=True)
            self.models.append(network.eval().to(self.device))
        self._torch = torch
        shape = (1, 3, 1, 1)
        self._mean = torch.tensor(_MEAN, dtype=torch.float32).view(shape).to(self.device)
        self._std = torch.tensor(_STD, dtype=torch.float32).view(shape).to(self.device)
        log.info("DFDC winner: %d models loaded on %s", len(self.models), self.device)

    def score(self, inputs: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        """P(fake) per 380 x 380 letterboxed face crop, the mean of the seven models' sigmoids."""
        torch = self._torch
        scores = []
        for start in range(0, len(inputs), _BATCH):
            batch = np.stack([np.asarray(c) for c in inputs[start : start + _BATCH]])
            x = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).float() / 255
            x = (x - self._mean) / self._std
            with torch.inference_mode():
                probs = [torch.sigmoid(m(x).reshape(-1)) for m in self.models]
            scores.append(torch.stack(probs).mean(0).float().cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0)
