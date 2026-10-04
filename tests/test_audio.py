"""Reading audio from audio files and from the audio track of video files."""

from __future__ import annotations

import av
import numpy as np
import pytest
import soundfile as sf

from deeptrace_bench.preprocess import PreprocessError
from deeptrace_bench.preprocess.audio import SAMPLE_RATE, load_audio

TONE_RATE = 22_050


def _tone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * TONE_RATE)) / TONE_RATE
    return (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _write_video(path, audio: np.ndarray | None) -> None:
    """A tiny synthetic mp4: ten black frames, plus a mono AAC track if given."""
    with av.open(str(path), "w") as out:
        video = out.add_stream("mpeg4", rate=10)
        video.width = video.height = 32
        video.pix_fmt = "yuv420p"
        sound = out.add_stream("aac", rate=TONE_RATE, layout="mono") if audio is not None else None
        for _ in range(10):
            frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), np.uint8), format="rgb24")
            out.mux(video.encode(frame))
        out.mux(video.encode())
        if sound is not None:
            size = sound.codec_context.frame_size
            for start in range(0, len(audio), size):
                chunk = audio[start : start + size][None, :]
                frame = av.AudioFrame.from_ndarray(chunk, format="flt", layout="mono")
                frame.sample_rate = TONE_RATE
                frame.pts = start
                out.mux(sound.encode(frame))
            out.mux(sound.encode())


def test_a_wav_is_read_mono_at_16k(tmp_path):
    path = tmp_path / "clip.wav"
    sf.write(path, np.stack([_tone(1.0), _tone(1.0)], axis=1), TONE_RATE)
    wave = load_audio(path)
    assert wave.dtype == np.float32 and wave.ndim == 1
    assert abs(len(wave) - SAMPLE_RATE) <= 1


def test_the_audio_track_of_a_video_is_read(tmp_path):
    path = tmp_path / "clip.mp4"
    _write_video(path, _tone(1.0))
    wave = load_audio(path)
    assert wave.dtype == np.float32 and wave.ndim == 1
    # AAC adds encoder padding, so allow a frame either way.
    assert abs(len(wave) - SAMPLE_RATE) < 2048
    assert 0.05 < np.abs(wave).max() < 0.5


def test_a_video_without_audio_is_a_status_not_a_crash(tmp_path):
    path = tmp_path / "silent.mp4"
    _write_video(path, None)
    with pytest.raises(PreprocessError) as info:
        load_audio(path)
    assert info.value.reason == "no_audio"
