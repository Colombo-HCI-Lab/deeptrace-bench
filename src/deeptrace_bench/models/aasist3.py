"""AASIST3 (Borodin et al. 2024, lab260ru/AASIST3, CC BY-NC-ND 4.0): an XLS-R 300M front end
(transformers' ``Wav2Vec2Model``) with KAN layers in the AASIST back end, trained on ASVspoof
2019 LA, ASVspoof 5, MLAAD and M-AILABS.

The repo's README says the released weights are not the ones behind the paper's numbers, and
the Speech DF Arena measures them at 28 to 32% EER on In-the-Wild and ASVspoof 2021, so this
is a weak, multilingual-trained baseline, not a strong one. It is refused on MLAAD.

The adapter imports the checkout's ``model/`` folder as a package (its files import each other
relatively; the KAN layer is in-repo) and builds ``aasist3`` from the Hub repo's own
``config.json``. Its front end calls ``Wav2Vec2Config.from_pretrained`` on
``facebook/wav2vec2-large-xlsr-53`` at whatever revision the Hub serves; the adapter rebinds
that name to a pinned ``config.json`` snapshot, so nothing is fetched. ``model.safetensors``
already uses transformers' names and loads with every key matching once re-saved.

Input, as upstream's data loader: 16 kHz mono, pre-emphasis with coefficient 0.97, 64,600
samples; the front end then scales each window by its peak. Upstream pre-emphasises the whole
utterance before cutting a segment, the adapter each window, which differs only at a window's
first sample. Output: two logits; upstream labels spoof 0 and bona fide 1 and scores
``outputs[:, 1]``, so P(fake) is the softmax at index 0. The README's own example reads index
0 as bona fide instead, so the direction still needs its check (``input.fake_index``).
"""

from __future__ import annotations

import importlib
import json
import logging
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ._upstream import load_package
from .base import Detector, resolve_device

log = logging.getLogger(__name__)

_PACKAGE = "dtb_upstream_aasist3"
_BATCH = 16
PREEMPHASIS = 0.97


def preemphasis(windows: np.ndarray, coeff: float = PREEMPHASIS) -> np.ndarray:
    """``y[t] = x[t] - coeff * x[t - 1]`` per window, the first sample kept as it is."""
    out = np.array(windows, dtype=np.float32, copy=True)
    out[..., 1:] -= coeff * windows[..., :-1]
    return out


class AASIST3Detector(Detector):
    """AASIST3 with its KAN back end."""

    @property
    def source(self) -> str:
        return self.config.weights[0].name

    def _build(self) -> Any:
        from transformers import Wav2Vec2Config

        assert self.upstream_dir is not None
        folder = Path(self.upstream_dir) / "model"
        try:
            load_package(folder, _PACKAGE)
        except FileNotFoundError:
            msg = f"{folder} is missing; run scripts/setup_models.py {self.config.id}"
            raise FileNotFoundError(msg) from None
        pinned = self.weights_dir / self.config.input["backbone_config"]
        wav2vec = importlib.import_module(f"{_PACKAGE}.wav2vec")

        class _PinnedConfig:
            @staticmethod
            def from_pretrained(name: str, **kwargs: Any) -> Wav2Vec2Config:
                if name != "facebook/wav2vec2-large-xlsr-53":
                    raise RuntimeError(f"{name} isn't pinned in configs/models/{self.config.id}")
                return Wav2Vec2Config.from_pretrained(str(pinned))

        wav2vec.Wav2Vec2Config = _PinnedConfig
        full_model = importlib.import_module(f"{_PACKAGE}.full_model")
        settings = json.loads((self.weights_dir / self.config.input["model_config"]).read_text())
        settings["load_pretrained"] = False
        with warnings.catch_warnings():  # its constructor warns that the repo is deprecated
            warnings.simplefilter("ignore", DeprecationWarning)
            return full_model.aasist3(**settings)

    def convert_checkpoint(self) -> Path | None:
        """Re-save the Hub's safetensors under the converted name, with its source hash."""
        from safetensors.torch import load_file

        if self.converted_is_current(self.source):
            return self.converted_path
        return self.save_converted(load_file(str(self.weights_dir / self.source)), self.source)

    def load(self, device: str) -> None:
        """Build the network and load the converted weights, every key matching."""
        import torch

        self.device = resolve_device(device)
        model = self._build()
        self.load_converted(model, self.source)
        self.model = model.eval().to(self.device)
        self.fake_index = int(self.config.input.get("fake_index", 0))
        self._torch = torch
        log.info("AASIST3 loaded on %s (fake index %d)", self.device, self.fake_index)

    def score(self, inputs: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        """P(fake) per window."""
        torch = self._torch
        windows = preemphasis(np.asarray(inputs, dtype=np.float32))
        scores = []
        for start in range(0, len(windows), _BATCH):
            x = torch.from_numpy(windows[start : start + _BATCH]).to(self.device)
            with torch.inference_mode():
                logits = self.model(x)
            scores.append(logits.softmax(dim=-1)[:, self.fake_index].float().cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0)
