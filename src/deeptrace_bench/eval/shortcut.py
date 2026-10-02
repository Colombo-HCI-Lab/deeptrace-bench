"""Shortcut probe for cross-corpus evalsets (not yet implemented).

When real and fake items come from different corpora (MLAAD fakes paired with real speech
from elsewhere), a detector can score well by telling the corpora apart (microphone, noise
floor, loudness, silence, original sample rate) rather than by spotting synthesis.

The probe fits a logistic regression on trivial features only and reports its
cross-validated AUC next to the detectors'. If the probe alone scores high, the evalset
measures the corpus, not fakeness, and its detector numbers need that caveat. Every
``cross_corpus`` evalset gets a probe result before its numbers are reported.
"""

from __future__ import annotations

import numpy as np

FEATURES = [
    "duration_s",
    "rms_db",
    "silence_ratio",
    "spectral_rolloff_hz",
    "spectral_bandwidth_hz",
    "original_sample_rate",
]


def trivial_features(wave: np.ndarray, sample_rate: int, original_sample_rate: int) -> dict:
    """Compute the probe's features for one clip. Not implemented yet."""
    raise NotImplementedError("shortcut probe is planned; see docs/evaluation.md")


def probe_auc(features: np.ndarray, labels: np.ndarray, folds: int = 5, seed: int = 0) -> float:
    """Cross-validated AUC of logistic regression on trivial features. Not implemented yet."""
    raise NotImplementedError("shortcut probe is planned; see docs/evaluation.md")
