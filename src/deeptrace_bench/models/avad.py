"""AVAD (Feng, Chen and Owens, CVPR 2023, cfeng16/audio-visual-forensics, MIT): flags videos whose
audio-visual timing looks unlikely, having only ever seen real ones.

Two networks, both from the pinned checkout (``detect.py`` itself can't be imported: it parses the
command line and opens ``pca.pkl`` at import):

- **Sync model** (``sync_model.pth``): a 2D+3D ResNet-18 over five face frames, an audio encoder
  over five frames' worth of log filterbanks, and upstream's ``MP_AViT`` transformer, which gives
  one synchronisation value for that pair. Each of a video's windows is paired with 31 audio
  windows, shifted from 15 frames early to 15 late, so a window becomes a 31-long profile over
  delays.
- **Autoregressive model** (``dist_regressive_model.pth``): a small causal transformer that
  predicts each window's delay profile from the ones before it.

The score is upstream's: the KL divergence between the softmax of each actual profile and the
prediction, summed over delays and averaged over windows, higher for fake. Only the delay
distributions count (``lam`` 0, ``detect.py``'s default and its README's commands), so the feature
branch and its ``pca.pkl`` and ``avfeature_regressive_model.pth`` are pinned but never run. Being
unbounded, the score goes through a sigmoid to read as a probability; that is monotone, so AUC and
EER are what upstream's raw score gives.

The video frames are the same at every delay, so they go through the visual encoder once per
window rather than 31 times; batch norm runs in eval mode, so the embeddings are identical.

``wav2filterbanks`` is rebuilt here (MIT) for two calls current libraries no longer take:
``librosa.filters.mel``'s positional arguments and ``torch.stft`` without ``return_complex``.

Input: ``preprocess.tracks.SyncedFaceLoader``'s 25 fps face track and 16 kHz audio. Upstream reads
wav files as 16-bit integers without rescaling, and frames on the 0 to 255 scale without
normalising, so the adapter does the same. Output: one score per video.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..preprocess import PreprocessError
from ._upstream import load_module, load_package
from .base import Detector, resolve_device

log = logging.getLogger(__name__)

_SYNC = "sync_model.pth"
_REGRESSIVE = "dist_regressive_model.pth"
_SAMPLES_PER_FRAME = 640  # 16 kHz over 25 fps
_CLIP_FRAMES = 5
_DELAYS = 31
_MAX_SHIFT = 15
_MAX_WINDOWS = 50
_BATCH = 8
# detect.py's audio_opts
_SAMPLE_RATE, _N_FFT, _WIN, _HOP, _N_MELS = 16000, 512, 320, 160, 80


def build_sync_network(root: Path) -> Any:
    """``detect.py``'s ``network`` of visual encoder, audio encoder and transformer."""
    import torch.nn as nn

    load_package(root / "backbone", "dtb_upstream_avad_backbone")
    backbone = importlib.import_module("dtb_upstream_avad_backbone.select_backbone")
    audio = load_module(root / "audio_process.py", "dtb_upstream_avad_audio")
    model = load_module(root / "model.py", "dtb_upstream_avad_model")

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vis_enc = backbone.select_backbone(network="r18")[0]
            self.aud_enc = audio.AudioEncoder()
            self.transformer = model.MP_AViT(
                image_size=14,
                patch_size=0,
                num_classes=1,
                dim=512,
                depth=3,
                heads=4,
                mlp_dim=512,
                dim_head=128,
                dropout=0.1,
                emb_dropout=0.1,
                max_visual_len=5,
                max_audio_len=4,
            )

    return Network()


def build_regressive_network(root: Path) -> Any:
    """``detect.py``'s autoregressive model over delay profiles."""
    module = load_module(root / "transformer_component.py", "dtb_upstream_avad_transformer")
    return module.transformer_decoder(
        input_dim_old=_DELAYS,
        input_dim=256,
        compress_factor=1,
        num_heads=16,
        dropout_prob=0.1,
        max_len=_MAX_WINDOWS - 1,
        layers=2,
    )


class AVADDetector(Detector):
    """AVAD's delay-distribution score on a face track and its audio."""

    def _converted(self, source: str) -> Path:
        return self.weights_dir / f"{Path(source).stem}.safetensors"

    def _current(self, source: str) -> bool:
        from safetensors import safe_open

        path = self._converted(source)
        if not path.exists():
            return False
        with safe_open(str(path), "pt") as f:
            meta = f.metadata() or {}
        return meta.get("source_sha256") == self._pinned_hash(source)

    def convert_checkpoint(self) -> Path | None:
        """Re-save both state dicts as safetensors."""
        import os

        import torch
        from safetensors.torch import save_file

        for source in (_SYNC, _REGRESSIVE):
            if self._current(source):
                continue
            digest = self._pinned_hash(source)
            if digest is None:
                raise RuntimeError(f"avad/{source} has no recorded hash; run setup_models --record")
            state = torch.load(self.weights_dir / source, map_location="cpu", weights_only=True)
            part = self._converted(source).with_suffix(".part")
            tensors = {k: v.contiguous() for k, v in state.items()}
            save_file(tensors, str(part), metadata={"source": source, "source_sha256": digest})
            os.replace(part, self._converted(source))
        return self._converted(_SYNC)

    def load(self, device: str) -> None:
        """Build both networks from the checkout and load them strictly."""
        import librosa
        import torch
        from safetensors.torch import load_model

        assert self.upstream_dir is not None
        self.device = resolve_device(device)
        for source in (_SYNC, _REGRESSIVE):
            if not self._current(source):
                raise RuntimeError(f"{self._converted(source)} is stale; run setup_models.py avad")
        root = Path(self.upstream_dir)
        sync = build_sync_network(root)
        regressive = build_regressive_network(root)
        load_model(sync, str(self._converted(_SYNC)), strict=True)
        load_model(regressive, str(self._converted(_REGRESSIVE)), strict=True)
        self.sync = sync.eval().to(self.device)
        self.regressive = regressive.eval().to(self.device)
        mel = librosa.filters.mel(
            sr=_SAMPLE_RATE, n_fft=_N_FFT, n_mels=_N_MELS, fmin=0, fmax=_SAMPLE_RATE // 2
        )
        self._mel = torch.from_numpy(mel).float().to(self.device)
        self._window = torch.hann_window(_WIN).to(self.device)
        self._torch = torch
        log.info("AVAD loaded on %s", self.device)

    def filterbanks(self, wave: Any) -> Any:
        """``wav2filterbanks`` then upstream's reshaping: ``[b, 1, 80, 20]`` from ``[b, 3200]``."""
        torch = self._torch
        spectrum = torch.stft(
            wave,
            n_fft=_N_FFT,
            hop_length=_HOP,
            win_length=_WIN,
            window=self._window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        magnitude = spectrum[:, :, :-1].abs()
        return torch.log(torch.matmul(self._mel, magnitude) + 1e-20)[:, None]

    def delay_profiles(self, faces: np.ndarray, wave: np.ndarray, windows: int) -> Any:
        """The sync model's ``[windows, 31]`` values, as ``test2`` computes them."""
        torch = self._torch
        audio = torch.from_numpy(wave).to(self.device)
        video = torch.from_numpy(faces).to(self.device).permute(3, 0, 1, 2)  # c, t, h, w
        profiles = []
        for start in range(0, windows, _BATCH):
            starts = range(start, min(start + _BATCH, windows))
            clips = torch.stack([video[:, k : k + _CLIP_FRAMES] for k in starts])
            cuts = [
                audio[(k + d) * _SAMPLES_PER_FRAME : (k + d + _CLIP_FRAMES) * _SAMPLES_PER_FRAME]
                for k in starts
                for d in range(_DELAYS)
            ]
            with torch.inference_mode():
                visual = self.sync.vis_enc(clips).repeat_interleave(_DELAYS, dim=0)
                sound = self.sync.aud_enc(self.filterbanks(torch.stack(cuts)))
                values = self.sync.transformer(visual, sound)
            profiles.append(values.reshape(len(starts), _DELAYS))
        return torch.cat(profiles)

    def score(self, inputs: Mapping[str, Any]) -> np.ndarray:  # type: ignore[override]
        """One score per item: the sigmoid of the KL divergence from the predicted profiles."""
        torch = self._torch
        functional = torch.nn.functional
        faces = np.asarray(inputs["faces"]).astype(np.float32)
        wave = np.asarray(inputs["audio"])
        if int(inputs["sample_rate"]) != _SAMPLE_RATE:
            raise ValueError("AVAD needs 16 kHz audio")
        frames = min(len(faces), len(wave) // _SAMPLES_PER_FRAME)
        windows = min(frames - _CLIP_FRAMES + 1, _MAX_WINDOWS)
        if windows < 2:  # one window has nothing to predict it from
            raise PreprocessError("too_short", f"{frames} frames with both video and audio")
        wave = np.round(wave[: frames * _SAMPLES_PER_FRAME] * 32768).astype(np.float32)
        wave = np.pad(wave, _MAX_SHIFT * _SAMPLES_PER_FRAME)
        profiles = self.delay_profiles(faces[:frames], wave, windows)

        pad = _MAX_WINDOWS - windows
        sequence = functional.pad(profiles, (0, 0, 0, pad))[None]
        mask = torch.zeros(1, _MAX_WINDOWS, device=self.device)
        mask[:, windows:] = 1.0
        with torch.inference_mode():
            target = sequence[:, 1:].softmax(dim=2)
            logits = self.regressive(sequence[:, :-1].clone(), mask[:, :-1])
            divergence = functional.kl_div(logits.log_softmax(dim=2), target, reduction="none")
        score = divergence[0, : _MAX_WINDOWS - pad - 1].sum(dim=1).mean()
        return np.array([float(torch.sigmoid(score.double()))])
