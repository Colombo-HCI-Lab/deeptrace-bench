"""HAVIC (CVPR 2026 Findings, tuffy-studio/HAVIC, MIT): an audio-visual detector, pretrained on
LRS2 and fine-tuned on a random 70% of FakeAVCeleb.

The adapter imports ``src/models/`` from the pinned checkout as a package and builds
``HAVIC_FT`` with its defaults, as ``evaluation/sliding_window_infer.py`` does. Two patches in
``patches/HAVIC/`` make it run: timm 1.x's ``Attention`` no longer takes ``qk_scale`` (always
None here, so the scale is the same), and ``HAVIC_FT.forward`` passes ``use_mask=False`` to
encoders that don't accept it (masking is already off with ``ids_keep=None``), which breaks
upstream's own inference at this commit.

Inputs (``preprocess.av.AudioVideoLoader``), following the inference script:

- Frames at 5 per second from the start, at most 50 (10 s); each frame's face box, uncropped
  (``align: box``, margin 1.0), resized to 224 x 224 bilinear and scaled to [0, 1].
  Upstream's boxes come from FaceX-Zoo's RetinaFace, ours from the shared SCRFD, and a frame
  without a face drops out here where upstream would skip the whole video.
- Windows of 16 frames, stride 2, the frame count trimmed so the windows tile it.
- Audio: the track at 16 kHz mono, kept in the same proportion as the frames used, cut into
  one segment per window (``_audio_segments``, upstream's arithmetic), each segment centred,
  128-bin Kaldi filterbanks (Hanning window, 10 ms shift, no dither) stretched to 1,024 frames
  and normalised with upstream's mean and std.

Output: one logit per window, 1 = fake; P(fake) is its sigmoid. Upstream takes the sigmoid of
the mean logit; the harness averages per-window probabilities, as for every other model.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..preprocess import PreprocessError
from ._upstream import load_package
from .base import Detector, resolve_device

log = logging.getLogger(__name__)

_PACKAGE = "dtb_upstream_havic_models"
_FBANK_MEAN, _FBANK_STD = -6.9960, 3.1205
_TARGET_FRAMES = 1024
_BATCH = 4


def _audio_segments(
    n_samples: int, n_used: int, n_sampled: int, size: int, stride: int
) -> list[tuple[int, int]]:
    """``(start, end)`` of each window's audio, as upstream's sliding-window inference cuts it.

    The audio kept is the share of the track the used frames cover; each window's segment is
    as long as that audio stretched over the windows' overlap pattern.
    """
    n_windows = (n_used - size) // stride + 1
    total = int((n_used / n_sampled) * n_samples)
    length = int((total * size) / (size + stride * (n_windows - 1)))
    step = int((stride / size) * length)
    segments = []
    for i in range(n_windows):
        start, end = i * step, i * step + length
        if end > total:
            end = total
            start = max(0, end - length)
        segments.append((int(start), int(end)))
    return segments


class HAVICDetector(Detector):
    """HAVIC, fine-tuned."""

    @property
    def source(self) -> str:
        return self.config.weights[0].name

    def _build(self) -> Any:
        assert self.upstream_dir is not None
        folder = Path(self.upstream_dir) / "src" / "models"
        try:
            load_package(folder, _PACKAGE)
        except FileNotFoundError:
            msg = f"{folder} is missing; run scripts/setup_models.py {self.config.id}"
            raise FileNotFoundError(msg) from None
        return importlib.import_module(f"{_PACKAGE}.HAVIC").HAVIC_FT()

    def convert_checkpoint(self) -> Path | None:
        """Re-save the fine-tuned state dict as safetensors."""
        import torch

        if self.converted_is_current(self.source):
            return self.converted_path
        state = torch.load(self.weights_dir / self.source, map_location="cpu", weights_only=True)
        return self.save_converted(state, self.source)

    def load(self, device: str) -> None:
        """Build HAVIC and load the converted weights, every key matching."""
        import torch

        self.device = resolve_device(device)
        model = self._build()
        self.load_converted(model, self.source)
        self.model = model.eval().to(self.device)
        self.window = int(self.config.input["window_frames"])
        self.stride = int(self.config.input["frame_stride"])
        self._torch = torch
        log.info("HAVIC loaded on %s", self.device)

    def _frames(self, crops: list[np.ndarray]) -> Any:
        from PIL import Image

        torch = self._torch
        resized = [
            np.asarray(Image.fromarray(np.asarray(c)).resize((224, 224), Image.BILINEAR))
            for c in crops
        ]
        return torch.from_numpy(np.stack(resized)).permute(0, 3, 1, 2).float() / 255

    def _fbank(self, segment: Any, sample_rate: int) -> Any:
        import torchaudio

        torch = self._torch
        fbank = torchaudio.compliance.kaldi.fbank(
            segment,
            htk_compat=True,
            sample_frequency=sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10,
        )
        fbank = (
            torch.nn.functional.interpolate(
                fbank.unsqueeze(0).transpose(1, 2),
                size=(_TARGET_FRAMES,),
                mode="linear",
                align_corners=False,
            )
            .transpose(1, 2)
            .squeeze(0)
        )
        return (fbank - _FBANK_MEAN) / _FBANK_STD

    def score(self, inputs: dict[str, Any]) -> np.ndarray:  # type: ignore[override]
        """P(fake) per window of 16 frames and the audio under them."""
        torch = self._torch
        crops = list(inputs["faces"])
        if len(crops) < self.window:
            raise PreprocessError(
                "too_short", f"{len(crops)} face frames, a window needs {self.window}"
            )
        remainder = (len(crops) - self.window) % self.stride
        if remainder:
            crops = crops[:-remainder]
        frames = self._frames(crops)
        starts = range(0, len(crops) - self.window + 1, self.stride)
        video = [frames[s : s + self.window].permute(1, 0, 2, 3) for s in starts]

        wave = torch.from_numpy(np.asarray(inputs["audio"], dtype=np.float32))[None]
        cuts = _audio_segments(
            wave.shape[1], len(crops), int(inputs["n_sampled"]), self.window, self.stride
        )
        audio = []
        for start, end in cuts:
            segment = wave[:, start:end]
            if segment.shape[1] < 400:  # under one 25 ms analysis frame
                raise PreprocessError("too_short", f"{segment.shape[1]} audio samples per window")
            audio.append(self._fbank(segment - segment.mean(), int(inputs["sample_rate"])))

        scores = []
        for i in range(0, len(video), _BATCH):
            v = torch.stack(video[i : i + _BATCH]).to(self.device)
            a = torch.stack(audio[i : i + _BATCH]).to(self.device)
            with torch.inference_mode():
                logits = self.model(audio=a, video=v, is_training=False)
            scores.append(torch.sigmoid(logits.float()).reshape(-1).cpu().numpy())
        return np.concatenate(scores)
