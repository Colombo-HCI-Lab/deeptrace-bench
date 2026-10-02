"""Shared base for adapters that are documented but not written yet."""

from __future__ import annotations

import numpy as np

from .base import Detector


class PendingDetector(Detector):
    """Raises with a pointer to the porting notes until the adapter is written."""

    def load(self, device: str) -> None:
        """Not implemented yet."""
        raise NotImplementedError(
            f"the {self.config.id} adapter isn't written yet; porting steps are in its module "
            "docstring and docs/models.md"
        )

    def score(self, inputs: np.ndarray) -> np.ndarray:
        """Not implemented yet."""
        raise NotImplementedError(f"the {self.config.id} adapter isn't written yet")
