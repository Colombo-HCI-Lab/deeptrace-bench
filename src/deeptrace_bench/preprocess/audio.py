"""Audio loading and fixed-length segmentation.

Every audio model gets the same input: 16 kHz mono, cut into 64,600-sample windows (about
4 s, the length AASIST and XLS-R + AASIST were trained on). A clip's score is the mean of its
window scores (see ``configs/eval/default.yaml``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from . import PreprocessError

SAMPLE_RATE = 16_000
SEGMENT_SAMPLES = 64_600


def load_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Read a file as float32 mono at ``sample_rate``.

    Raises:
        PreprocessError: ``unreadable`` if the file can't be decoded, ``empty_audio`` if it
            has no samples.
    """
    try:
        wave, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001 - any decode failure is the same outcome
        raise PreprocessError("unreadable", f"{path}: {exc}") from exc
    if wave.size == 0:
        raise PreprocessError("empty_audio", str(path))
    mono = wave.mean(axis=1)
    if sr != sample_rate:
        mono = soxr.resample(mono, sr, sample_rate).astype(np.float32)
    return mono


def segment(
    wave: np.ndarray,
    segment_samples: int = SEGMENT_SAMPLES,
    min_samples: int = SAMPLE_RATE,
) -> np.ndarray:
    """Cut a waveform into non-overlapping windows covering all of it.

    A clip shorter than one window is tiled up to one window, as AASIST's own ``pad`` does.
    A longer clip gets as many windows as fit, plus one aligned to the end if samples are
    left over, so the tail is scored too.

    Args:
        wave: 1-D float waveform.
        segment_samples: window length.
        min_samples: clips shorter than this are refused rather than tiled.

    Returns:
        Array of shape ``[n_windows, segment_samples]``.

    Raises:
        PreprocessError: ``too_short`` below ``min_samples``.
    """
    n = len(wave)
    if n < min_samples:
        raise PreprocessError("too_short", f"{n} samples < {min_samples}")
    if n <= segment_samples:
        reps = -(-segment_samples // n)
        return np.tile(wave, reps)[:segment_samples][None, :]
    starts = list(range(0, n - segment_samples + 1, segment_samples))
    if starts[-1] + segment_samples < n:
        starts.append(n - segment_samples)
    return np.stack([wave[s : s + segment_samples] for s in starts])
