"""AntiDeepfake (Ge et al. 2025, "Post-training for Deepfake Speech Detection", NII Yamagishi
Lab): speech SSL models post-trained on 74k hours of real and fake speech in over 100
languages. Code BSD-3 (nii-yamagishilab/AntiDeepfake), checkpoints CC BY-NC-SA 4.0.

Each checkpoint is one Hugging Face repo with a ``model.safetensors`` holding upstream's
``models/W2V.py`` ``Model``: a fairseq wav2vec 2.0 front end (``m_ssl.model.*``), mean
pooling over frames and one linear layer (``proj_fc``). The adapter imports ``W2V.py`` from
the pinned GitHub checkout with stand-in ``fairseq`` modules (``_fairseq.fairseq_stand_in``):
upstream builds its front end from ``W2V_configs.py``'s fairseq settings for
``input.variant`` (``mms_300m``, ``w2v_small`` and so on), which become the equivalent
transformers ``Wav2Vec2Model``. ``convert_checkpoint`` renames the front end's keys and drops
the pretraining-only ones; the converted file then loads with every key matching.

Input: 16 kHz mono, each window normalised to zero mean and unit variance
(``layer_norm(wav, wav.shape)``, as upstream's test loader does to a whole utterance).
Upstream scores a whole utterance in one pass; the harness scores 4 s windows and averages
them (``configs/eval/default.yaml``), as for the other audio models, so long clips differ
from upstream's protocol. Output: two logits, upstream's labels are real 1 and fake 0, so
P(fake) is the softmax at index 0.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ._fairseq import (
    check_config,
    fairseq_stand_in,
    fairseq_style_model,
    fairseq_to_hf,
    hf_config_from_fairseq,
)
from ._upstream import load_module, load_package, scoped_modules
from .base import Detector, resolve_device

log = logging.getLogger(__name__)

_PREFIX = "m_ssl.model."
_PACKAGE = "models"  # upstream's own package name, removed again after the import
_BATCH = 8


class AntiDeepfakeDetector(Detector):
    """One AntiDeepfake checkpoint, the front end named by ``input.variant``."""

    @property
    def variant(self) -> str:
        return self.config.input["variant"]

    @property
    def source(self) -> str:
        """The downloaded checkpoint (the Hub's ``model.safetensors``, saved under its own name)."""
        return self.config.weights[0].name

    def _fairseq_settings(self) -> dict[str, Any]:
        configs = self._import("W2V_configs")
        return dict(configs.global_configs[self.variant])

    def _import(self, name: str) -> Any:
        """Import ``models/<name>.py`` from the checkout, with fairseq stood in."""
        assert self.upstream_dir is not None
        folder = Path(self.upstream_dir) / "models"
        stubs = fairseq_stand_in(
            lambda settings: fairseq_style_model(hf_config_from_fairseq(settings))
        )
        with scoped_modules(stubs, purge=[_PACKAGE]):
            try:
                load_package(folder, _PACKAGE)
                return load_module(folder / f"{name}.py", f"{_PACKAGE}.{name}")
            except FileNotFoundError:
                msg = f"{folder} is missing; run scripts/setup_models.py {self.config.id}"
                raise FileNotFoundError(msg) from None

    def _build(self) -> Any:
        """Upstream's ``Model`` for this variant, front end in transformers."""
        assert self.upstream_dir is not None
        folder = Path(self.upstream_dir) / "models"
        stubs = fairseq_stand_in(
            lambda settings: fairseq_style_model(hf_config_from_fairseq(settings))
        )
        with scoped_modules(stubs, purge=[_PACKAGE]):
            load_package(folder, _PACKAGE)
            w2v = load_module(folder / "W2V.py", f"{_PACKAGE}.W2V")
            return w2v.Model(self.variant)

    def convert_checkpoint(self) -> Path | None:
        """Rename the fairseq front end's keys and re-save."""
        from safetensors.torch import load_file

        if self.converted_is_current(self.source):
            return self.converted_path
        state = load_file(str(self.weights_dir / self.source))
        converted = fairseq_to_hf(state, prefix=_PREFIX, new_prefix=_PREFIX)
        backbone = {
            k.removeprefix(_PREFIX): v for k, v in converted.items() if k.startswith(_PREFIX)
        }
        check_config(hf_config_from_fairseq(self._fairseq_settings()), backbone)
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
        log.info("AntiDeepfake %s loaded on %s", self.variant, self.device)

    def score(self, inputs: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        """P(fake) per window."""
        torch = self._torch
        windows = np.asarray(inputs, dtype=np.float32)
        scores = []
        for start in range(0, len(windows), _BATCH):
            x = torch.from_numpy(windows[start : start + _BATCH]).to(self.device)
            with torch.inference_mode():
                x = torch.nn.functional.layer_norm(x, x.shape[-1:])
                logits = self.model(x)
            scores.append(logits.softmax(dim=-1)[:, self.fake_index].float().cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0)
