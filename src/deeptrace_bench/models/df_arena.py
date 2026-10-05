"""DF Arena 500M and 1B (Speech-Arena-2025), universal speech anti-spoofing models.

Each is a Hugging Face repo holding both code and weights: ``backbone.py`` (an XLS-R
``Wav2Vec2Model``, attention pooling over every hidden layer, and a conformer head from
``conformer.py``), ``modeling_antispoofing.py`` (the ``PreTrainedModel`` wrapper) and
``pytorch_model.bin``. The model card loads them with ``trust_remote_code``; this adapter
doesn't. The code comes from the pinned checkout (``upstream.host: hf``), loaded as a package
so its relative imports work, and two things change without altering the network:

- ``backbone.py`` builds its XLS-R from ``Wav2Vec2Config.from_pretrained("facebook/...")`` at
  whatever revision the Hub serves. The adapter rebinds that name to a shim that reads the
  pinned ``config.json`` snapshot in the weights folder and refuses any other repo, so
  loading never touches the network.
- The pickled ``pytorch_model.bin`` is converted once to safetensors and loaded with every
  key matching.

Input: 16 kHz mono windows of 64,600 samples. The card's feature extractor scores only a
clip's first 64,600 samples (tiling a shorter clip up to that length); the harness scores
every window and averages them (``configs/eval/default.yaml``), so the two agree on clips up
to about 4 s and differ on longer ones. Parity compares window by window and can't see this;
reproduced EERs can. ``forward`` unsqueezes a 1-D waveform, so windows go through one at a
time. Output: two logits with ``id2label {0: spoof, 1: bonafide}``; P(fake) is the softmax at
``label2id["spoof"]``. Training data per the card: ASVspoof 2019 and 2024, MLAAD, Codecfake,
LibriSeVoc, DFADD, CtrSVDD, SpoofCeleb, EnvSDD (see the model configs for the guard).
"""

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ._upstream import load_package
from .base import Detector, resolve_device

log = logging.getLogger(__name__)

SOURCE = "pytorch_model.bin"
# Old weight-norm names (fairseq and older transformers) and what torch's parametrizations
# call them now.
_RENAMES = {
    "pos_conv_embed.conv.weight_g": "pos_conv_embed.conv.parametrizations.weight.original0",
    "pos_conv_embed.conv.weight_v": "pos_conv_embed.conv.parametrizations.weight.original1",
}


class DFArenaDetector(Detector):
    """DF Arena 500M or 1B, whichever the config pins."""

    def _package(self) -> str:
        assert self.upstream_dir is not None
        name = f"dtb_upstream_{self.config.id}"
        try:
            load_package(self.upstream_dir, name)
        except FileNotFoundError:
            msg = f"{self.upstream_dir} is missing; run scripts/setup_models.py {self.config.id}"
            raise FileNotFoundError(msg) from None
        return name

    def _pinned_configs(self) -> dict[str, Path]:
        """Hub repo id -> pinned local snapshot, for every ``hf_snapshot`` weight."""
        return {
            w.repo: self.weights_dir / w.name
            for w in self.config.weights
            if w.kind == "hf_snapshot" and w.repo
        }

    def _build(self) -> tuple[Any, Any]:
        """The network (random weights) and its config, from the checkout's own classes."""
        from transformers import Wav2Vec2Config

        assert self.upstream_dir is not None
        package = self._package()
        pinned = self._pinned_configs()

        class _PinnedWav2Vec2Config:
            @staticmethod
            def from_pretrained(name: str, **kwargs: Any) -> Wav2Vec2Config:
                if name not in pinned:
                    raise RuntimeError(f"{name} isn't pinned in configs/models/{package}")
                return Wav2Vec2Config.from_pretrained(str(pinned[name]), **kwargs)

        backbone = importlib.import_module(f"{package}.backbone")
        backbone.Wav2Vec2Config = _PinnedWav2Vec2Config

        meta = json.loads((Path(self.upstream_dir) / "config.json").read_text())
        auto = meta["auto_map"]
        config_cls = _attr(package, auto["AutoConfig"])
        model_cls = _attr(package, auto["AutoModel"])
        config = config_cls.from_json_file(str(Path(self.upstream_dir) / "config.json"))
        return model_cls(config), config

    def convert_checkpoint(self) -> Path | None:
        """Re-save the pickled checkpoint as safetensors, with current weight-norm names."""
        import torch

        if self.converted_is_current(SOURCE):
            return self.converted_path
        state = torch.load(
            self.weights_dir / SOURCE, map_location="cpu", mmap=True, weights_only=True
        )
        renamed = {}
        for key, value in state.items():
            for old, new in _RENAMES.items():
                if key.endswith(old):
                    key = key.removesuffix(old) + new
            renamed[key] = value
        return self.save_converted(renamed, SOURCE)

    def load(self, device: str) -> None:
        """Build the network from the checkout and load the converted weights."""
        import torch

        self.device = resolve_device(device)
        model, config = self._build()
        self.load_converted(model, SOURCE)
        self.model = model.eval().to(self.device)
        self.fake_index = int(config.label2id["spoof"])
        self._torch = torch
        log.info("%s loaded on %s (fake index %d)", self.config.id, self.device, self.fake_index)

    def score(self, inputs: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        """P(fake) per window."""
        torch = self._torch
        scores = []
        for window in np.asarray(inputs, dtype=np.float32):
            x = torch.from_numpy(window).to(self.device)
            with torch.inference_mode():
                logits = self.model(x)["logits"].reshape(-1, 2)
            scores.append(float(logits.softmax(dim=-1)[0, self.fake_index]))
        return np.asarray(scores, dtype=np.float64)


def _attr(package: str, target: str) -> Any:
    """``module.Class`` inside the loaded package, as written in ``auto_map``."""
    module_name, _, attr = target.rpartition(".")
    return getattr(importlib.import_module(f"{package}.{module_name}"), attr)
