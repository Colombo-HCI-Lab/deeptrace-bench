"""The interface every detector adapter implements.

An adapter wraps one upstream model so the harness can treat them all the same:

- ``convert_checkpoint`` turns the downloaded checkpoint into ``model.safetensors`` with
  normalised keys (strip ``module.``, remap fairseq names to transformers names), once. After
  that, adapters load safetensors only and never unpickle files from Google Drive.
- ``load`` builds the network and loads the converted weights.
- ``score`` maps a batch of preprocessed inputs (audio windows or face crops of one item) to
  P(fake) per window or frame. Audio windows come as one ``[n, samples]`` array; face crops
  come as a list of ``[h, w, 3]`` uint8 RGB arrays, since a model with native-size crops gets
  crops of different sizes. The harness aggregates the scores into one per item, so the
  aggregation rule lives in one place (``configs/eval/default.yaml``).

Adapters must match upstream: before an adapter is used for results, its scores on about 200
items are checked against the upstream code run in its original environment (see
docs/models.md, "Parity check").
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..fetch import locked_hashes
from ..paths import weights_dir
from ..registry import ModelConfig, resolve
from ..upstream import checkout_dir

CONVERTED = "model.safetensors"
# Keys some trainers wrap a state dict in.
_WRAPPERS = ("state_dict", "model", "model_state_dict", "net")


class Detector(ABC):
    """Base class for detector adapters.

    ``conversion_version`` is recorded in every converted file; an adapter bumps it whenever
    its ``convert_checkpoint`` changes what it writes, so older conversions are redone.
    """

    conversion_version = 1

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.device = "cpu"

    @property
    def weights_dir(self) -> Path:
        """The model's weights directory in the store."""
        return weights_dir(self.config.id)

    @property
    def upstream_dir(self) -> Path | None:
        """The pinned upstream checkout, if the model has one."""
        return checkout_dir(self.config.upstream) if self.config.upstream else None

    @property
    def converted_path(self) -> Path:
        """Where ``convert_checkpoint`` writes the normalised weights."""
        return self.weights_dir / CONVERTED

    def convert_checkpoint(self) -> Path | None:
        """Write ``model.safetensors`` from the downloaded checkpoint. Default: nothing to do.

        ``scripts/setup_models.py`` calls this after the downloads pass their hash checks.
        An adapter that converts returns early when ``converted_is_current`` says the file
        already comes from today's download.

        Returns:
            The converted file, or None if the model loads its download directly.
        """
        return None

    def converted_is_current(self, source: str) -> bool:
        """True if the converted file was made from the download now pinned as ``source``,
        by the current conversion code."""
        if not self.converted_path.exists():
            return False
        expected = (source, self._pinned_hash(source), str(self.conversion_version))
        return _converted_from(self.converted_path) == expected

    def save_converted(self, state: Mapping[str, Any], source: str) -> Path:
        """Write ``state`` as the converted weights, recording which download it came from.

        A ``{"state_dict": ...}`` style wrapper is unwrapped and a ``module.`` prefix (from
        ``DataParallel``) stripped. The source file's hash from the lock goes into the
        safetensors metadata, so ``load_converted`` can tell a stale conversion apart.

        Raises:
            RuntimeError: if ``source`` has no hash in the lock yet.
            ValueError: if ``source`` is itself called ``model.safetensors``, which the
                conversion would overwrite; such a weight needs another local ``name``.
        """
        from safetensors.torch import save_file

        if source == CONVERTED:
            raise ValueError(
                f"{self.config.id}: the download {source} would be overwritten by its own "
                f"conversion; rename it (the weight's name) in configs/models/{self.config.id}.yaml"
            )
        digest = self._pinned_hash(source)
        if digest is None:
            raise RuntimeError(
                f"{self.config.id}/{source} has no recorded hash; run scripts/setup_models.py "
                f"{self.config.id} --record first"
            )
        state = unwrap_state(state)
        tensors = {k.removeprefix("module."): v.contiguous() for k, v in state.items()}
        part = self.converted_path.with_name(CONVERTED + ".part")
        metadata = {
            "source": source,
            "source_sha256": digest,
            "conversion_version": str(self.conversion_version),
        }
        save_file(tensors, str(part), metadata=metadata)
        os.replace(part, self.converted_path)
        return self.converted_path

    def load_converted(self, module: Any, source: str, strict: bool = True) -> None:
        """Load the weights converted from ``source`` into ``module``, every key matching.

        Raises:
            RuntimeError: if the file is missing, was converted from another file or another
                download of ``source``, or by older conversion code.
        """
        from safetensors.torch import load_model

        path = self.converted_path
        if not path.exists():
            raise RuntimeError(f"{path} is missing; run scripts/setup_models.py {self.config.id}")
        if not self.converted_is_current(source):
            found, _, version = _converted_from(path)
            raise RuntimeError(
                f"{path} isn't a current conversion of {source} (it came from {found}, "
                f"conversion version {version}); re-run scripts/setup_models.py {self.config.id}"
            )
        load_model(module, str(path), strict=strict)

    def _pinned_hash(self, source: str) -> str | None:
        return locked_hashes(self.config.id).get(source)

    @abstractmethod
    def load(self, device: str) -> None:
        """Build the network on ``device`` and load weights."""

    @abstractmethod
    def score(self, inputs: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        """Return P(fake) for each window or frame in ``inputs`` (``n`` of them)."""


def unwrap_state(state: Mapping[str, Any]) -> Mapping[str, Any]:
    """The tensors of a checkpoint, out of a ``{"state_dict": ...}`` style wrapper if any."""
    for key in _WRAPPERS:
        inner = state.get(key)
        if isinstance(inner, Mapping) and inner:
            return inner
    return state


def _converted_from(path: Path) -> tuple[str | None, str | None, str | None]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as fh:
        meta = fh.metadata() or {}
    return meta.get("source"), meta.get("source_sha256"), meta.get("conversion_version")


def resolve_device(device: str) -> str:
    """``auto`` becomes the best available device (cuda, then mps, then cpu); others pass."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_tf32(enabled: bool) -> None:
    """Allow or forbid TF32 on CUDA (cuDNN convolutions and matmuls) for this process.

    cuDNN allows TF32 convolutions by default on Ampere and newer GPUs. That moved DF Arena's
    scores up to 2.5e-2 from fp32 on CPU, where parity holds to 1e-3; in fp32 the gap was
    1.5e-4 (checked 2026-10-05). ``configs/eval/default.yaml`` (``compute.tf32``) decides.
    """
    import torch

    torch.backends.cudnn.allow_tf32 = enabled
    torch.backends.cuda.matmul.allow_tf32 = enabled


def load_detector(config: ModelConfig) -> Detector:
    """Instantiate a model's adapter from its config.

    Raises:
        ValueError: if the model has no adapter yet.
    """
    if config.adapter is None:
        raise ValueError(f"{config.id} has no adapter yet (status: {config.status})")
    cls = resolve(config.adapter)
    return cls(config)
