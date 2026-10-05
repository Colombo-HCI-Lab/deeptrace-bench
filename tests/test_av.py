"""Audio-visual inputs: frames at a fixed rate, the audio track, and HAVIC's audio arithmetic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from deeptrace_bench.models.havic import _audio_segments
from deeptrace_bench.preprocess.av import AudioVideoLoader
from deeptrace_bench.preprocess.faces import (
    CropSpec,
    FaceLoader,
    rate_frame_count,
    rate_frame_indices,
)

from .test_audio import _tone, _write_video
from .test_faces import StubDetector


def test_frames_at_a_rate_start_at_zero_and_stop_at_the_cap():
    assert rate_frame_indices(100, fps=25.0, rate=5, max_frames=50) == list(range(0, 100, 5))
    assert rate_frame_indices(1000, fps=25.0, rate=5, max_frames=50)[-1] == 49 * 5
    assert len(rate_frame_indices(1000, fps=25.0, rate=5, max_frames=50)) == 50
    assert rate_frame_count(240, fps=24.0, rate=5) == 50  # 10 s, uncapped
    assert rate_frame_indices(0, fps=25.0, rate=5, max_frames=50) == []


def test_audio_segments_follow_the_windows():
    # 50 frames used of 50 sampled, windows of 16 every 2: 18 windows over all the audio
    cuts = _audio_segments(160_000, n_used=50, n_sampled=50, size=16, stride=2)
    assert len(cuts) == 18
    length = int(160_000 * 16 / 50)
    assert all(end - start == length for start, end in cuts)
    assert cuts[1][0] == int(2 / 16 * length) and cuts[-1][1] <= 160_000
    # frames used are a share of those sampled: the audio kept shrinks in proportion
    half = _audio_segments(160_000, n_used=24, n_sampled=48, size=16, stride=2)
    assert max(end for _, end in half) <= 80_000


def test_the_loader_returns_faces_audio_and_the_frame_count(tmp_path, monkeypatch):
    monkeypatch.setenv("DTB_ROOT", str(tmp_path))
    monkeypatch.setenv("DTB_NAMESPACE", "smoke")
    folder = tmp_path / "smoke" / "datasets" / "toy"
    folder.mkdir(parents=True)
    _write_video(folder / "clip.mp4", _tone(1.0))  # ten frames at 10 fps, one second
    faces = FaceLoader(
        crop=CropSpec(size=None, margin=1.0, align="box"),
        make_detector=StubDetector,
        detector_id="stub",
        settings={"detector": "stub", "frame_sampling": "rate"},
        frames_per_clip=50,
        frame_rate=5,
        min_face_frames=1,
    )
    loader = AudioVideoLoader(faces)
    row = pd.Series(
        {"item_id": "toy/clip", "dataset": "toy", "rel_path": "clip.mp4", "modality": "audio_video"}
    )
    out = loader(row)
    loader.close()
    assert out["n_sampled"] == 5 and len(out["faces"]) == 5  # frames 0, 2, 4, 6, 8
    assert out["sample_rate"] == 16_000 and abs(len(out["audio"]) - 16_000) < 2_000
    assert np.isfinite(out["audio"]).all()
