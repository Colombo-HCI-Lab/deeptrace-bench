"""The interface every detector adapter implements.

An adapter wraps one upstream model so the harness can treat them all the same:

- ``convert_checkpoint`` turns the downloaded checkpoint into ``model.safetensors`` with
  normalised keys (strip ``module.``, remap fairseq names to transformers names), once. After
  that, adapters load safetensors only and never unpickle files from Google Drive.
- ``load`` builds the network and loads the converted weights.
- ``score`` maps a batch of preprocessed inputs (audio windows or face crops of one item) to
  P(fake) per window or frame. The harness aggregates those into one score per item, so the
  aggregation rule lives in one place (``configs/eval/default.yaml``).

Adapters must match upstream: before an adapter is used for results, its scores on about 200
items are checked against the upstream code run in its original environment (see
docs/models.md, "Parity check").
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..paths import weights_dir
from ..registry import ModelConfig, resolve
from ..upstream import checkout_dir


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

    def convert_checkpoint(self) -> Path | None:
        """Write ``model.safetensors`` from the downloaded checkpoint. Default: nothing to do.

        Returns:
            The converted file, or None if the model loads its download directly.
        """
        return None

    @abstractmethod
    def load(self, device: str) -> None:
        """Build the network on ``device`` and load weights."""

    @abstractmethod
    def score(self, inputs: np.ndarray) -> np.ndarray:
        """Return P(fake) for each window or frame in ``inputs`` (shape ``[n, ...]``)."""


def load_detector(config: ModelConfig) -> Detector:
    """Instantiate a model's adapter from its config.

    Raises:
        ValueError: if the model has no adapter yet.
    """
    if config.adapter is None:
        raise ValueError(f"{config.id} has no adapter yet (status: {config.status})")
    cls = resolve(config.adapter)
    return cls(config)
