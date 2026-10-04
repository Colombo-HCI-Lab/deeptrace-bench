"""Audio loading and fixed-length segmentation.

Every audio model gets the same input: 16 kHz mono, cut into 64,600-sample windows (about
4 s, the length AASIST and XLS-R + AASIST were trained on). A clip's score is the mean of its
window scores (see ``configs/eval/default.yaml``).

Audio files are read with soundfile. Video containers (an audio model scoring an audio-video
dataset hears its audio track) and m4a are decoded with PyAV, whose wheels bundle FFmpeg, so
no system binary is needed. Both paths mix channels down by averaging and resample with soxr,
so a clip reads the same whichever container it came in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from . import PreprocessError

SAMPLE_RATE = 16_000
SEGMENT_SAMPLES = 64_600
# Decoded with PyAV rather than soundfile.
CONTAINER_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4a"}


def load_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Read a file as float32 mono at ``sample_rate``.

    Raises:
        PreprocessError: ``unreadable`` if the file can't be decoded, ``no_audio`` for a
            video without an audio track, ``empty_audio`` if there are no samples.
    """
    try:
        if Path(path).suffix.lower() in CONTAINER_SUFFIXES:
            wave, sr = _decode_container(Path(path))
        else:
            wave, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except PreprocessError:
        raise
    except Exception as exc:  # noqa: BLE001 - any decode failure is the same outcome
        raise PreprocessError("unreadable", f"{path}: {exc}") from exc
    if wave.size == 0:
        raise PreprocessError("empty_audio", str(path))
    mono = wave.mean(axis=1)
    if sr != sample_rate:
        mono = soxr.resample(mono, sr, sample_rate).astype(np.float32)
    return mono


def _decode_container(path: Path) -> tuple[np.ndarray, int]:
    """The first audio stream as float32 ``[samples, channels]`` at its own rate."""
    import av

    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise PreprocessError("no_audio", str(path))
        # Planar float at the stream's own rate and layout; resampling stays with soxr.
        resampler = av.AudioResampler(format="fltp")
        chunks = []
        for frame in container.decode(stream):
            chunks += [out.to_ndarray() for out in resampler.resample(frame)]
        chunks += [out.to_ndarray() for out in resampler.resample(None)]
        rate = stream.rate
    if not chunks:
        return np.zeros((0, 1), np.float32), rate
    return np.concatenate(chunks, axis=1).T.astype(np.float32), rate


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
