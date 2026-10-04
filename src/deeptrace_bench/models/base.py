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
    """Base class for detector adapters."""

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
        """True if the converted file was made from the download now pinned as ``source``."""
        if not self.converted_path.exists():
            return False
        return _converted_from(self.converted_path) == (source, self._pinned_hash(source))

    def save_converted(self, state: Mapping[str, Any], source: str) -> Path:
        """Write ``state`` as the converted weights, recording which download it came from.

        A ``{"state_dict": ...}`` style wrapper is unwrapped and a ``module.`` prefix (from
        ``DataParallel``) stripped. The source file's hash from the lock goes into the
        safetensors metadata, so ``load_converted`` can tell a stale conversion apart.

        Raises:
            RuntimeError: if ``source`` has no hash in the lock yet.
        """
        from safetensors.torch import save_file

        digest = self._pinned_hash(source)
        if digest is None:
            raise RuntimeError(
                f"{self.config.id}/{source} has no recorded hash; run scripts/setup_models.py "
                f"{self.config.id} --record first"
            )
        state = unwrap_state(state)
        tensors = {k.removeprefix("module."): v.contiguous() for k, v in state.items()}
        part = self.converted_path.with_name(CONVERTED + ".part")
        save_file(tensors, str(part), metadata={"source": source, "source_sha256": digest})
        os.replace(part, self.converted_path)
        return self.converted_path

    def load_converted(self, module: Any, strict: bool = True) -> None:
        """Load the converted weights into ``module``, every key matching when ``strict``.

        Raises:
            RuntimeError: if the file is missing or was converted from a different download
                than the one pinned now.
        """
        from safetensors.torch import load_model

        path = self.converted_path
        if not path.exists():
            raise RuntimeError(f"{path} is missing; run scripts/setup_models.py {self.config.id}")
        source, digest = _converted_from(path)
        if digest is None or digest != self._pinned_hash(source):
            raise RuntimeError(
                f"{path} was converted from another download of {source}; re-run "
                f"scripts/setup_models.py {self.config.id}"
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


def _converted_from(path: Path) -> tuple[str | None, str | None]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as fh:
        meta = fh.metadata() or {}
    return meta.get("source"), meta.get("source_sha256")


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


def load_detector(config: ModelConfig) -> Detector:
    """Instantiate a model's adapter from its config.

    Raises:
        ValueError: if the model has no adapter yet.
    """
    if config.adapter is None:
        raise ValueError(f"{config.id} has no adapter yet (status: {config.status})")
    cls = resolve(config.adapter)
    return cls(config)
