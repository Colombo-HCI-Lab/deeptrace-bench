"""LipFD (Liu et al., NeurIPS 2024, AaronComo/LipFD): spots lip-sync fakes from images that stack
five video frames under the mel spectrogram of the same moment.

Upstream states no licence, so its code stays in ``third_party/`` and is imported from the pinned
checkout. The network is upstream's ``LipFD``: a 5 x 5 stride-5 convolution and CLIP ViT-L/14 read
the whole image, and a ResNet-50 reads crops of it at three scales, weighted region by region. Two
things differ from upstream's import, neither touching the network:

- ``clip.load`` would download OpenAI's checkpoint, which ``ckpt.pth`` then replaces in full. While
  the network is built it returns a CLIP built from the checkpoint's own ``encoder.*`` shapes.
- ``ftfy``, which only the text tokenizer uses, is a stand-in, and ``pkg_resources`` is reduced to
  the ``packaging`` it re-exports.

Weights: the ``model`` entry of ``ckpt.pth``, re-saved as safetensors without its optimizer state.

Input: ``preprocess.tracks.LipCompositeLoader``'s images, ``[n, 1000, 2500, 3]`` RGB uint8. Scoring
follows ``AVLip.__getitem__`` and ``validate.py``:

- Upstream reads its PNGs with ``cv2.imread``, so the network sees BGR, on the 0 to 255 scale. The
  ``Normalize`` in ``__getitem__`` is assigned to ``crops`` and overwritten on the next line, so
  nothing is normalised.
- ``crops[0]`` is ``img[:, 500:, i:i + 500] for i in range(5)``: five crops of the first frame one
  pixel apart, not the five frames, whatever the comment beside them intends. The released
  checkpoint was evaluated that way, so this keeps it; the other frames reach the model only
  through the CLIP branch.
- Resizing is torchvision's on tensors, which in upstream's torchvision 0.16 means no antialiasing.

Output: the sigmoid of the one logit, per image. Upstream reports images, not videos; the harness
averages them into one score per video.
"""

from __future__ import annotations

import importlib
import logging
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ._upstream import load_package, scoped_modules
from .base import Detector, resolve_device

log = logging.getLogger(__name__)

_SOURCE = "ckpt.pth"
_PACKAGE = "dtb_upstream_lipfd"
_ARCH = "ViT-L/14"
_ENCODER = "encoder."
_BATCH = 4
_SIDE, _CROP, _WHOLE = 500, 224, 1120
_SCALES = ((28, 196), (61, 163))  # crops[1], 0.65x, and crops[2], 0.45x


def _stubs() -> dict[str, types.ModuleType]:
    import packaging

    ftfy = types.ModuleType("ftfy")
    ftfy.fix_text = lambda text: text  # type: ignore[attr-defined]
    pkg_resources = types.ModuleType("pkg_resources")
    pkg_resources.packaging = packaging  # type: ignore[attr-defined]
    return {"ftfy": ftfy, "pkg_resources": pkg_resources}


def build_network(root: Path, state: dict[str, Any]) -> Any:
    """Upstream's ``LipFD``, its CLIP built from the shapes in ``state``, not downloaded."""
    load_package(root / "models", _PACKAGE)
    with scoped_modules(_stubs()):
        module = importlib.import_module(f"{_PACKAGE}.LipFD")
    clip = module.clip
    encoder = {k.removeprefix(_ENCODER): v for k, v in state.items() if k.startswith(_ENCODER)}
    original = clip.load
    clip.load = lambda name, device: (clip.build_model(encoder).float(), None)
    try:
        return module.LipFD(_ARCH)
    finally:
        clip.load = original


def network_inputs(images: Any) -> tuple[Any, list[list[Any]]]:
    """``AVLip.__getitem__`` on a float BGR batch ``[n, 3, 1000, 2500]``: the whole image at
    1120 and the three scales of crops."""
    from torch.nn.functional import interpolate

    def resize(x: Any, side: int) -> Any:
        return interpolate(x, size=(side, side), mode="bilinear", align_corners=False)

    crops: list[list[Any]] = [
        [resize(images[:, :, _SIDE:, i : i + _SIDE], _CROP) for i in range(5)],
        [],
        [],
    ]
    for scale, (low, high) in enumerate(_SCALES, start=1):
        crops[scale] = [resize(crop[:, :, low:high, low:high], _CROP) for crop in crops[0]]
    return resize(images, _WHOLE), crops


class LipFDDetector(Detector):
    """LipFD on stacked frames and mel spectrograms."""

    def convert_checkpoint(self) -> Path | None:
        """Keep the network's weights and re-save them as safetensors."""
        import torch

        if self.converted_is_current(_SOURCE):
            return self.converted_path
        checkpoint = torch.load(
            self.weights_dir / _SOURCE, map_location="cpu", weights_only=True, mmap=True
        )
        return self.save_converted(checkpoint["model"], _SOURCE)

    def load(self, device: str) -> None:
        """Build upstream's network and load the converted weights, every key matching."""
        import torch
        from safetensors.torch import load_file

        assert self.upstream_dir is not None
        self.device = resolve_device(device)
        state = load_file(str(self.converted_path))
        network = build_network(Path(self.upstream_dir), state)
        self.load_converted(network, _SOURCE)
        self.model = network.eval().to(self.device)
        self._torch = torch
        log.info("LipFD loaded on %s", self.device)

    def score(self, inputs: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        """P(fake) per image (``[1000, 2500, 3]`` RGB uint8 each)."""
        torch = self._torch
        images = np.asarray(inputs)
        scores = []
        for start in range(0, len(images), _BATCH):
            bgr = torch.from_numpy(images[start : start + _BATCH, :, :, ::-1].copy())
            x = bgr.to(self.device).permute(0, 3, 1, 2).float()
            whole, crops = network_inputs(x)
            with torch.inference_mode():
                features = self.model.get_features(whole)
                logits = self.model(crops, features)[0]
            scores.append(logits.sigmoid().reshape(-1).float().cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0)
