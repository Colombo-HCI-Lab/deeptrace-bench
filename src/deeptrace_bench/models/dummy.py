"""A deterministic stand-in detector for tests and dry runs. Never used for results."""

from __future__ import annotations

import numpy as np

from .base import Detector


class DummyDetector(Detector):
    """Scores each window by a squashed mean of its values, so outputs are reproducible."""

    def load(self, device: str) -> None:
        """Nothing to load."""
        self.device = device

    def score(self, inputs: np.ndarray) -> np.ndarray:
        """Return sigmoid(mean) per row of ``inputs``."""
        flat = inputs.reshape(len(inputs), -1).astype(np.float64)
        return 1.0 / (1.0 + np.exp(-flat.mean(axis=1)))
