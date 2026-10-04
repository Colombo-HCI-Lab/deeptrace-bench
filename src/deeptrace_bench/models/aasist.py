"""AASIST and AASIST-L (clovaai/aasist, MIT), trained on ASVspoof 2019 LA train.

The adapter loads upstream's ``models/AASIST.py`` from the pinned checkout and builds
``Model`` from the checkout's own ``config/<variant>.conf`` (``model_config``), so the two
variants differ only in that file and their weights (``AASIST.pth``, ``AASIST-L.pth``,
committed upstream). Plain PyTorch; nothing needed patching. Checked 2026-10-05: both
checkpoints load with every key matching, no prefix.

Input: 16 kHz mono windows of 64,600 samples (``preprocess.audio.segment``), the length the
models were trained on. Output: two logits; upstream's data loader labels bona fide 1 and
spoof 0, and its evaluation scores the bona fide logit. So P(fake) is the softmax at
``input.fake_index`` (0). A user reports inverted outputs with the shipped weights (aasist
issue 17), so the direction is checked on ASVspoof 2019 LA eval, where AUC must come out
above 0.5.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import numpy as np

from ._upstream import load_module
from .base import Detector, resolve_device

log = logging.getLogger(__name__)

_MODULE_NAME = "dtb_upstream_aasist"
_BATCH = 32


class AASISTDetector(Detector):
    """AASIST or AASIST-L, picked by ``input.variant``."""

    @property
    def variant(self) -> str:
        return self.config.input["variant"]

    @property
    def source(self) -> str:
        """The downloaded checkpoint (the config's first weight)."""
        return self.config.weights[0].name

    def _module(self) -> ModuleType:
        assert self.upstream_dir is not None
        path = self.upstream_dir / "models" / "AASIST.py"
        try:
            return load_module(path, _MODULE_NAME)
        except FileNotFoundError:
            msg = f"{path} is missing; run scripts/setup_models.py {self.config.id}"
            raise FileNotFoundError(msg) from None

    def _model_config(self) -> dict:
        assert self.upstream_dir is not None
        conf = Path(self.upstream_dir) / "config" / f"{self.variant}.conf"
        return json.loads(conf.read_text())["model_config"]

    def convert_checkpoint(self) -> Path | None:
        """Re-save the upstream checkpoint (a plain state dict) as safetensors."""
        import torch

        if self.converted_is_current(self.source):
            return self.converted_path
        state = torch.load(self.weights_dir / self.source, map_location="cpu", weights_only=True)
        return self.save_converted(state, self.source)

    def load(self, device: str) -> None:
        """Build the variant's network and load its converted weights."""
        import torch

        self.device = resolve_device(device)
        model = self._module().Model(self._model_config())
        self.load_converted(model)
        self.model = model.eval().to(self.device)
        self.fake_index = int(self.config.input.get("fake_index", 0))
        self._torch = torch
        log.info("%s loaded on %s (fake index %d)", self.variant, self.device, self.fake_index)

    def score(self, inputs: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        """P(fake) per window."""
        torch = self._torch
        windows = np.asarray(inputs, dtype=np.float32)
        scores = []
        for start in range(0, len(windows), _BATCH):
            x = torch.from_numpy(windows[start : start + _BATCH]).to(self.device)
            with torch.inference_mode():
                _, logits = self.model(x)
            scores.append(logits.softmax(dim=-1)[:, self.fake_index].float().cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0)
