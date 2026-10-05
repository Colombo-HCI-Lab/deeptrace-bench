"""Detectors built on Tak et al.'s fairseq XLS-R wrapper: a fine-tuned XLS-R 300M front end
with a back end trained on ASVspoof 2019 LA.

Three upstreams share one layout, set by TakHemlata/SSL_Anti-spoofing (MIT): a ``model.py``
whose ``SSLModel`` loads fairseq's XLS-R 300M (``xlsr2_300m.pt``) with
``checkpoint_utils.load_model_ensemble_and_task`` and calls it with
``(x, mask=False, features_only=True)``, a ``Model(args, device)`` around it, and a checkpoint
of the whole fine-tuned network with the front end under ``ssl_model.model.``:

- **XLS-R + AASIST** (Tak et al., Odyssey 2022): the AASIST graph back end on the last
  layer's output (``["x"]``).
- **XLS-R + SLS** (Zhang et al., ACM MM 2024, QiShanZhang/SLSforASVspoof-2021-DF, no licence
  declared): attention over every layer's output (``["layer_results"]``) and two linear
  layers; returns log-probabilities, whose softmax is the probabilities themselves.
- XLSR-Mamba subclasses this adapter (``xlsr_mamba.py``).

None of them runs here as released (Python 3.7, torch 1.8, a vendored fairseq). The adapter
imports the checkout's ``model.py`` with stand-in ``fairseq`` modules
(``_fairseq.fairseq_stand_in``) that hand ``SSLModel`` transformers' ``Wav2Vec2Model``, built
from the pinned ``facebook/wav2vec2-xls-r-300m`` config (the same network) and answering
fairseq's call. Every other layer is upstream's own. ``convert_checkpoint`` renames the front
end's keys (``_fairseq.fairseq_to_hf``) and drops the pretraining-only quantiser; the
converted file then loads with every key matching. The pretrained XLS-R weights are not
needed: the checkpoint replaces all of them.

Input: 16 kHz mono windows of 64,600 samples, unnormalised, as the upstream loaders cut them
(SLS's first linear layer is sized for exactly that length). Output: two classes, upstream
labels bona fide 1 and scores ``batch_out[:, 1]``, so P(fake) is the softmax at index 0
(``input.fake_index``). Pretraining: XLS-R saw Common Voice, BABEL, MLS, VoxPopuli and
VoxLingua107, so evalsets whose real speech comes from those corpora get a ``source_overlap``
verdict.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ._fairseq import check_config, fairseq_stand_in, fairseq_style_model, fairseq_to_hf
from ._upstream import load_module, scoped_modules
from .base import Detector, resolve_device, unwrap_state

log = logging.getLogger(__name__)

_PREFIX = "ssl_model.model."
_BATCH = 16


class SSLXLSRDetector(Detector):
    """An upstream ``model.py`` on Tak et al.'s XLS-R wrapper, fairseq rebuilt in transformers."""

    conversion_version = 1

    @property
    def source(self) -> str:
        return self.config.input["checkpoint"]

    @property
    def backbone_config_dir(self) -> Path:
        return self.weights_dir / self.config.input["backbone_config"]

    @property
    def module_name(self) -> str:
        return f"dtb_upstream_{self.config.id}"

    def _backbone_config(self) -> Any:
        from transformers import Wav2Vec2Config

        return Wav2Vec2Config.from_pretrained(str(self.backbone_config_dir))

    def extra_stubs(self) -> dict[str, Any]:
        """Stand-ins beyond fairseq that a subclass's upstream needs while it imports."""
        return {}

    def model_args(self) -> Any:
        """What upstream's ``Model(args, device)`` gets as ``args``."""
        return None

    def _build(self) -> Any:
        """Upstream's ``Model`` with a transformers XLS-R in place of fairseq's."""
        config = self._backbone_config()
        stubs = fairseq_stand_in(lambda _checkpoint: fairseq_style_model(config))
        stubs |= self.extra_stubs()
        assert self.upstream_dir is not None
        path = Path(self.upstream_dir) / self.config.input.get("module", "model.py")
        with scoped_modules(stubs):
            try:
                module = load_module(path, self.module_name)
            except FileNotFoundError:
                msg = f"{path} is missing; run scripts/setup_models.py {self.config.id}"
                raise FileNotFoundError(msg) from None
            # ``SSLModel`` looks ``fairseq`` up when built, and a module imported earlier
            # still holds that load's stand-in; point it at this one.
            module.fairseq = stubs["fairseq"]
            return module.Model(self.model_args(), "cpu")

    def convert_checkpoint(self) -> Path | None:
        """Rename the fairseq front end's keys and save the whole network as safetensors."""
        if self.converted_is_current(self.source):
            return self.converted_path
        converted = fairseq_to_hf(self._read_source(), prefix=_PREFIX, new_prefix=_PREFIX)
        backbone = {
            k.removeprefix(_PREFIX): v for k, v in converted.items() if k.startswith(_PREFIX)
        }
        check_config(self._backbone_config(), backbone)
        return self.save_converted(converted, self.source)

    def _read_source(self) -> dict[str, Any]:
        """The checkpoint's state dict, without a ``DataParallel`` ``module.`` prefix (SLS)."""
        path = self.weights_dir / self.source
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(path))
        else:
            import torch

            state = torch.load(path, map_location="cpu", weights_only=True)
        return {k.removeprefix("module."): v for k, v in unwrap_state(state).items()}

    def load(self, device: str) -> None:
        """Build the network and load the converted weights, every key matching."""
        import torch

        self.device = resolve_device(device)
        model = self._build()
        self.load_converted(model, self.source)
        self.model = model.eval().to(self.device)
        self.fake_index = int(self.config.input.get("fake_index", 0))
        self._torch = torch
        log.info("%s loaded on %s (fake index %d)", self.config.id, self.device, self.fake_index)

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
