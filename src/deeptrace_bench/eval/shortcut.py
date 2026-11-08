"""Shortcut probe for cross-corpus evalsets.

When real and fake items come from different corpora (MLAAD fakes paired with real speech
from elsewhere), a detector can score well by telling the corpora apart (microphone, noise
floor, loudness, silence, original sample rate) rather than by spotting synthesis.

The probe fits a logistic regression on trivial features only and reports its
cross-validated AUC next to the detectors'. If the probe alone scores high, the evalset
measures the corpus, not fakeness, and its detector numbers need that caveat. Every
``cross_corpus`` evalset gets a probe result before its numbers are reported; a
``same_corpus`` one can be probed too, as a baseline for what "low" looks like.

Features come from the clip as every audio model hears it (16 kHz mono, ``load_audio``), plus
the file's own sample rate, which resampling hides:

- ``duration_s``: length in seconds.
- ``rms_db``: overall loudness, 20 log10 of the RMS (floored at -120 dB).
- ``silence_ratio``: the share of 25 ms frames (10 ms hop) more than 40 dB below the
  clip's loudest frame.
- ``spectral_rolloff_hz``: the mean, over non-silent frames, of the frequency under which
  85% of a frame's energy lies.
- ``spectral_bandwidth_hz``: the mean, over non-silent frames, of the energy-weighted spread
  around the frame's spectral centroid.
- ``original_sample_rate``: the file's sample rate before resampling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FEATURES = [
    "duration_s",
    "rms_db",
    "silence_ratio",
    "spectral_rolloff_hz",
    "spectral_bandwidth_hz",
    "original_sample_rate",
]
FRAME = 400  # 25 ms at 16 kHz
HOP = 160  # 10 ms
SILENCE_DB = 40.0
ROLLOFF = 0.85


def original_sample_rate(path: Path) -> int:
    """The sample rate stored in the file (its first audio stream for a container)."""
    from ..preprocess.audio import CONTAINER_SUFFIXES

    if Path(path).suffix.lower() in CONTAINER_SUFFIXES:
        import av

        with av.open(str(path)) as container:
            stream = next(s for s in container.streams if s.type == "audio")
            return int(stream.rate)
    import soundfile as sf

    return int(sf.info(str(path)).samplerate)


def trivial_features(wave: np.ndarray, sample_rate: int, original_sample_rate: int) -> dict:
    """The probe's features for one clip, ``wave`` being mono float at ``sample_rate``."""
    wave = np.asarray(wave, dtype=np.float64)
    duration = len(wave) / sample_rate
    rms = np.sqrt(np.mean(wave**2)) if len(wave) else 0.0
    rms_db = float(max(20 * np.log10(rms), -120.0)) if rms > 0 else -120.0

    if len(wave) < FRAME:
        wave = np.pad(wave, (0, FRAME - len(wave)))
    starts = np.arange(0, len(wave) - FRAME + 1, HOP)
    frames = np.stack([wave[s : s + FRAME] for s in starts]) * np.hanning(FRAME)
    energy = (frames**2).sum(axis=1)
    energy_db = 10 * np.log10(energy + 1e-12)
    loud = energy_db > energy_db.max() - SILENCE_DB
    silence_ratio = float(1 - loud.mean())

    power = np.abs(np.fft.rfft(frames[loud], axis=1)) ** 2
    freqs = np.fft.rfftfreq(FRAME, 1 / sample_rate)
    total = power.sum(axis=1)
    keep = total > 0
    if keep.any():
        power, total = power[keep], total[keep]
        cumulative = np.cumsum(power, axis=1)
        rolloff = freqs[np.argmax(cumulative >= ROLLOFF * total[:, None], axis=1)]
        centroid = (power * freqs).sum(axis=1) / total
        spread = (power * (freqs - centroid[:, None]) ** 2).sum(axis=1) / total
        rolloff_hz, bandwidth_hz = float(rolloff.mean()), float(np.sqrt(spread).mean())
    else:
        rolloff_hz = bandwidth_hz = 0.0
    return {
        "duration_s": float(duration),
        "rms_db": rms_db,
        "silence_ratio": silence_ratio,
        "spectral_rolloff_hz": rolloff_hz,
        "spectral_bandwidth_hz": bandwidth_hz,
        "original_sample_rate": float(original_sample_rate),
    }


def probe_scores(
    features: np.ndarray, labels: np.ndarray, folds: int = 5, seed: int = 0
) -> np.ndarray:
    """Out-of-fold P(fake) from a standardised logistic regression on ``features``."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    splits = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return cross_val_predict(model, features, labels, cv=splits, method="predict_proba")[:, 1]


def probe_auc(features: np.ndarray, labels: np.ndarray, folds: int = 5, seed: int = 0) -> float:
    """Cross-validated AUC of logistic regression on trivial features."""
    from .metrics import auc

    return auc(labels, probe_scores(features, labels, folds, seed))


def feature_aucs(features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Each feature alone as a score, as max(AUC, 1 - AUC): how far it separates the classes."""
    from .metrics import auc

    out = {}
    for j, name in enumerate(FEATURES):
        column = features[:, j]
        a = auc(labels, column) if np.ptp(column) > 0 else 0.5
        out[name] = float(max(a, 1 - a))
    return out
