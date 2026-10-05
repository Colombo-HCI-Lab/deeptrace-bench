"""XLS-R 300M + AASIST (Tak et al. 2022, TakHemlata/SSL_Anti-spoofing, MIT), trained on
ASVspoof 2019 LA train.

Upstream pins Python 3.7, torch 1.8.1 and a vendored fairseq, none of which run here. The
network itself is plain PyTorch apart from its front end: ``model.py``'s ``SSLModel`` loads
fairseq's XLS-R 300M (``xlsr2_300m.pt``) and calls it with
``(x, mask=False, features_only=True)["x"]``, the last layer's output after the final layer
norm. The adapter imports ``model.py`` from the pinned checkout with a stand-in ``fairseq``
whose ``load_model_ensemble_and_task`` returns transformers' ``Wav2Vec2Model`` (built from
the pinned ``facebook/wav2vec2-xls-r-300m`` config, which describes the same network) behind
that call signature. Every other layer is upstream's own.

``LA_model.pth`` holds the whole fine-tuned network, XLS-R included, with fairseq's names
under ``ssl_model.model.``; ``convert_checkpoint`` renames those with ``_fairseq.fairseq_to_hf``
and drops the pretraining-only quantiser, then the converted file loads with every key
matching. The pretrained XLS-R weights are not needed: the checkpoint replaces all of them.

Input: 16 kHz mono windows of 64,600 samples, as upstream's evaluation loader cuts them.
Output: two logits; upstream labels bona fide 1 and scores ``batch_out[:, 1]``, so P(fake) is
the softmax at index 0 (``input.fake_index``). Pretraining: XLS-R saw Common Voice, BABEL,
MLS, VoxPopuli and VoxLingua107, so evalsets whose real speech comes from those corpora get a
``source_overlap`` verdict.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ._fairseq import check_config, fairseq_stand_in, fairseq_style_model, fairseq_to_hf
from ._upstream import load_module, scoped_modules
from .base import Detector, resolve_device

log = logging.getLogger(__name__)

_MODULE_NAME = "dtb_upstream_ssl_antispoofing"
_PREFIX = "ssl_model.model."
_BATCH = 16


class XLSRAASISTDetector(Detector):
    """XLS-R + AASIST, with the fairseq front end rebuilt in transformers."""

    conversion_version = 1

    @property
    def source(self) -> str:
        return self.config.input["checkpoint"]

    @property
    def backbone_config_dir(self) -> Path:
        return self.weights_dir / self.config.input["backbone_config"]

    def _backbone_config(self) -> Any:
        from transformers import Wav2Vec2Config

        return Wav2Vec2Config.from_pretrained(str(self.backbone_config_dir))

    def _build(self) -> Any:
        """Upstream's ``Model`` with a transformers XLS-R in place of fairseq's."""
        config = self._backbone_config()
        stubs = fairseq_stand_in(lambda _checkpoint: fairseq_style_model(config))
        assert self.upstream_dir is not None
        path = Path(self.upstream_dir) / "model.py"
        with scoped_modules(stubs):
            try:
                module = load_module(path, _MODULE_NAME)
            except FileNotFoundError:
                msg = f"{path} is missing; run scripts/setup_models.py {self.config.id}"
                raise FileNotFoundError(msg) from None
            # ``SSLModel`` looks ``fairseq`` up when built, and a module imported earlier
            # still holds that load's stand-in; point it at this one.
            module.fairseq = stubs["fairseq"]
            return module.Model(None, "cpu")

    def convert_checkpoint(self) -> Path | None:
        """Rename the fairseq front end's keys and save the whole network as safetensors."""
        import torch

        if self.converted_is_current(self.source):
            return self.converted_path
        state = torch.load(self.weights_dir / self.source, map_location="cpu", weights_only=True)
        converted = fairseq_to_hf(state, prefix=_PREFIX, new_prefix=_PREFIX)
        backbone = {
            k.removeprefix(_PREFIX): v for k, v in converted.items() if k.startswith(_PREFIX)
        }
        check_config(self._backbone_config(), backbone)
        return self.save_converted(converted, self.source)

    def load(self, device: str) -> None:
        """Build the network and load the converted weights, every key matching."""
        import torch

        self.device = resolve_device(device)
        model = self._build()
        self.load_converted(model, self.source)
        self.model = model.eval().to(self.device)
        self.fake_index = int(self.config.input.get("fake_index", 0))
        self._torch = torch
        log.info("XLS-R + AASIST loaded on %s (fake index %d)", self.device, self.fake_index)

    def score(self, inputs: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        """P(fake) per window."""
        torch = self._torch
        windows = np.asarray(inputs, dtype=np.float32)
        scores = []
        for start in range(0, len(windows), _BATCH):
            x = torch.from_numpy(windows[start : start + _BATCH]).to(self.device)
            with torch.inference_mode():
                logits = self.model(x)
            scores.append(logits.softmax(dim=-1)[:, self.fake_index].float().cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0)
