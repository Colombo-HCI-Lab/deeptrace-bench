"""AVH-Align (Smeu et al., CVPR 2025, bit-ml/AVH-Align): a small fusion network that scores
how well a video's lips and its sound agree, on AV-HuBERT features, trained on real
AV-Deepfake1M videos only.

**Licence.** AV-HuBERT (facebookresearch/av_hubert) is under Meta's AV-HuBERT licence, which
bars use for "biometric processing" and reverse-engineering. Oshan accepted that risk for this
non-commercial research use on 2026-10-28 (``configs/models/avh_align.yaml``). AVH-Align itself
is CC BY-NC-SA 4.0. No AV-HuBERT code is in this repo or imported: fairseq doesn't install here,
so the encoder below is rebuilt from its published architecture with standard parts, and the
parity check against the real fairseq code is what shows it computes the same thing.

**The encoder** (AV-HuBERT large, the ``self_large_vox_433h`` checkpoint fine-tuned for lip
reading; AVH-Align uses its ``encoder.w2v_model`` and ``extract_finetune``):

- video: a 3D convolution stem (5 x 7 x 7, 64 channels, batch norm, PReLU, max pool) and a
  ResNet-18 trunk (basic blocks, PReLU) over each frame, then a linear projection to 1024;
- audio: 26 log filterbanks at 100 Hz stacked four at a time (104 values per video frame),
  projected to 1024;
- the two concatenated (2048), layer-normalised, projected to 1024 and passed through a
  24-layer pre-norm wav2vec 2.0 transformer with a convolutional position embedding:
  transformers' ``Wav2Vec2EncoderStableLayerNorm``, with fairseq's names mapped by
  ``_fairseq.hf_name``.

AVH-Align runs the encoder twice per video, once with only the video (the audio half of the
concatenation zeros) and once with only the audio, as ``deepfake_feature_extraction.py`` does.

**Scoring**, as ``eval.py``: both feature sequences L2-normalised per frame, upstream's own
``FusionModel`` (imported from the AVH-Align checkout) gives one alignment value per frame, and
the video's score is ``logsumexp(-value)`` over frames: one score per item, higher for fake.
The harness wants a probability, so the adapter reports that score's sigmoid; being monotone,
it leaves the AUC and EER exactly as upstream's raw score gives them.

**Inputs** come from ``preprocess.mouths.MouthLoader``, which runs the same mouth-ROI recipe
AV-HuBERT's ``preparation/align_mouth.py`` does (LRW mean face, stable points 33 to 45, 12-frame
smoothing, 96 x 96 patches over landmarks 48 to 67) on FAN landmarks rather than dlib's, over
the first 100 frames (the frames LipForensics reads, so they share one detection cache), plus
the audio track at 16 kHz. Each frame is scaled to [0, 1], centre-cropped to 88 and normalised
with mean 0.421 and std 0.165 (AV-HuBERT's task defaults). Upstream re-encodes the mouth crops
with ffmpeg before reading them back in grayscale, and doesn't resample video to 25 fps; neither
does this, so 24 fps MAVOS-DD video drifts against the 25 Hz audio features by one frame per
second, as it would upstream.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..preprocess import PreprocessError
from . import _fairseq
from ._upstream import load_module
from .base import Detector, resolve_device

log = logging.getLogger(__name__)

AVHUBERT = "self_large_vox_433h.pt"
FUSION = "AVH-Align_AV1M.pt"
_CROP = 88
_MEAN, _STD = 0.421, 0.165
_STACK = 4  # filterbank frames per video frame
_PREFIX = "encoder.w2v_model."
# AV-HuBERT parameters used only in pretraining or by the lip-reading decoder
_UNUSED = ("mask_emb", "final_proj.", "label_embs_concat")


def build_encoder(hparams: Mapping[str, Any]) -> Any:
    """AV-HuBERT's ``extract_finetune`` network, from the checkpoint's model settings."""
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Config
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2EncoderStableLayerNorm

    dim = int(hparams["encoder_embed_dim"])

    class BasicBlock(nn.Module):
        def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
            self.downsample = None
            if stride != 1 or inplanes != planes:
                self.downsample = nn.Sequential(
                    nn.Conv2d(inplanes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes)
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.relu1(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            residual = x if self.downsample is None else self.downsample(x)
            return self.relu2(out + residual)

    class Trunk(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            widths, inplanes = [64, 128, 256, 512], 64
            for i, planes in enumerate(widths, start=1):
                stride = 1 if i == 1 else 2
                blocks = [BasicBlock(inplanes, planes, stride), BasicBlock(planes, planes)]
                setattr(self, f"layer{i}", nn.Sequential(*blocks))
                inplanes = planes
            self.avgpool = nn.AdaptiveAvgPool2d(1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for i in range(1, 5):
                x = getattr(self, f"layer{i}")(x)
            return self.avgpool(x).flatten(1)

    class ResEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.frontend3D = nn.Sequential(
                nn.Conv3d(1, 64, (5, 7, 7), (1, 2, 2), (2, 3, 3), bias=False),
                nn.BatchNorm3d(64),
                nn.PReLU(num_parameters=64),
                nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)),
            )
            self.trunk = Trunk()

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # B, 1, T, H, W -> B, 512, T
            b = x.shape[0]
            x = self.frontend3D(x)
            t = x.shape[2]
            x = x.transpose(1, 2).reshape(b * t, *x.shape[1:2], *x.shape[3:])
            return self.trunk(x).view(b, t, -1).transpose(1, 2)

    class SubModel(nn.Module):
        def __init__(self, resnet: nn.Module | None, input_dim: int) -> None:
            super().__init__()
            self.resnet = resnet
            self.proj = nn.Linear(input_dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # -> B, dim, T
            if self.resnet is not None:
                x = self.resnet(x)
            return self.proj(x.transpose(1, 2)).transpose(1, 2)

    config = Wav2Vec2Config(
        hidden_size=dim,
        num_hidden_layers=int(hparams["encoder_layers"]),
        num_attention_heads=int(hparams["encoder_attention_heads"]),
        intermediate_size=int(hparams["encoder_ffn_embed_dim"]),
        hidden_act=hparams.get("activation_fn", "gelu"),
        do_stable_layer_norm=True,
        num_conv_pos_embeddings=int(hparams.get("conv_pos", 128)),
        num_conv_pos_embedding_groups=int(hparams.get("conv_pos_groups", 16)),
        layer_norm_eps=1e-5,
    )
    config._attn_implementation = "eager"

    class AVHubertEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_extractor_audio = SubModel(None, int(hparams["audio_feat_dim"]))
            self.feature_extractor_video = SubModel(ResEncoder(), 512)
            self.layer_norm = nn.LayerNorm(2 * dim)
            self.post_extract_proj = nn.Linear(2 * dim, dim)
            self.encoder = Wav2Vec2EncoderStableLayerNorm(config)

        def forward(self, audio: torch.Tensor | None, video: torch.Tensor | None) -> torch.Tensor:
            """``extract_finetune`` without masking: audio ``[B, 104, T]``, video
            ``[B, 1, T, 88, 88]``, either may be None; returns ``[B, T, dim]``."""
            if video is None:
                features_audio = self.feature_extractor_audio(audio)
                features_video = features_audio.new_zeros(features_audio.shape)
            elif audio is None:
                features_video = self.feature_extractor_video(video)
                features_audio = features_video.new_zeros(features_video.shape)
            else:
                features_video = self.feature_extractor_video(video)
                features_audio = self.feature_extractor_audio(audio)
            x = torch.cat([features_audio, features_video], dim=1).transpose(1, 2)
            x = self.post_extract_proj(self.layer_norm(x))
            return self.encoder(x).last_hidden_state

    if hparams.get("modality_fuse", "concat") != "concat" or hparams.get("sub_encoder_layers", 0):
        raise ValueError("only AV-HuBERT's concat fusion without sub-encoders is supported")
    return AVHubertEncoder()


def avhubert_state(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """The ``encoder.w2v_model`` weights of a fairseq AV-HuBERT checkpoint, renamed."""
    out: dict[str, Any] = {}
    for key, value in checkpoint["model"].items():
        if not key.startswith(_PREFIX):
            continue  # the lip-reading decoder
        name = key.removeprefix(_PREFIX)
        if name.startswith(_UNUSED):
            continue
        if name.startswith("encoder."):
            name = _fairseq.hf_name(name)
            if name is None:
                continue
        out[name] = value
    return out


def avhubert_hparams(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """The settings that shape the encoder, from the checkpoint's fairseq config."""
    cfg = checkpoint["cfg"]["model"]
    w2v = cfg["w2v_args"]["model"] if "w2v_args" in cfg else cfg
    # AVHubertConfig's defaults, for the settings the checkpoint leaves out
    defaults = {
        "activation_fn": "gelu",
        "conv_pos": 128,
        "conv_pos_groups": 16,
        "sub_encoder_layers": 0,
        "resnet_relu_type": "prelu",
        "modality_fuse": "concat",
    }
    keys = [
        "encoder_embed_dim",
        "encoder_layers",
        "encoder_attention_heads",
        "encoder_ffn_embed_dim",
        "activation_fn",
        "layer_norm_first",
        "conv_pos",
        "conv_pos_groups",
        "audio_feat_dim",
        "modality_fuse",
        "sub_encoder_layers",
        "resnet_relu_type",
    ]
    found = {k: w2v[k] for k in keys if w2v.get(k) is not None}
    hparams = {**defaults, **found}
    if hparams["resnet_relu_type"] != "prelu" or not hparams.get("layer_norm_first", False):
        raise ValueError(f"unexpected AV-HuBERT settings: {hparams}")
    return hparams


def load_fairseq_checkpoint(path: Path) -> dict[str, Any]:
    """A fairseq AV-HuBERT checkpoint through torch's safe unpickler.

    Its task state pickles fairseq's ``Dictionary`` and ``SentencepieceBPE`` and a
    ``SentencePieceProcessor`` (the lip-reading vocabulary), which fairseq isn't here to
    rebuild; inert stand-ins take their places. Every tensor loads as usual.
    """
    import torch

    class _Inert:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __setstate__(self, state: Any) -> None:
            self.state = state

    names = [
        "fairseq.data.dictionary.Dictionary",
        "fairseq.data.encoders.sentencepiece_bpe.SentencepieceBPE",
        "sentencepiece.SentencePieceProcessor",
    ]
    allowed = [(type(n.rsplit(".", 1)[1], (_Inert,), {}), n) for n in names]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def audio_features(wave: np.ndarray, sample_rate: int) -> np.ndarray:
    """``load_audio`` of ``deepfake_feature_extraction.py`` on a decoded wave: 26 log
    filterbanks, padded to a multiple of four frames, stacked four at a time and
    layer-normalised per stacked frame: ``[T, 104]``."""
    import torch
    from python_speech_features import logfbank

    feats = logfbank(wave, samplerate=sample_rate).astype(np.float32)
    if len(feats) % _STACK:
        pad = _STACK - len(feats) % _STACK
        feats = np.concatenate([feats, np.zeros((pad, feats.shape[1]), dtype=feats.dtype)])
    feats = feats.reshape(-1, _STACK * feats.shape[1])
    tensor = torch.from_numpy(feats)
    return torch.nn.functional.layer_norm(tensor, tensor.shape[1:]).numpy()


class AVHAlignDetector(Detector):
    """AVH-Align on AV-HuBERT large features of a video's mouth crops and audio."""

    def _converted(self, source: str) -> Path:
        return self.weights_dir / f"{Path(source).stem}.safetensors"

    def _current(self, source: str) -> bool:
        from safetensors import safe_open

        path = self._converted(source)
        if not path.exists():
            return False
        with safe_open(str(path), "pt") as f:
            meta = f.metadata() or {}
        return (meta.get("source"), meta.get("source_sha256")) == (
            source,
            self._pinned_hash(source),
        )

    def _save(self, state: Mapping[str, Any], source: str, extra: Mapping[str, str]) -> None:
        import os

        from safetensors.torch import save_file

        digest = self._pinned_hash(source)
        if digest is None:
            raise RuntimeError(
                f"avh_align/{source} has no recorded hash; run setup_models --record"
            )
        part = self._converted(source).with_suffix(".part")
        tensors = {k: v.contiguous() for k, v in state.items()}
        save_file(tensors, str(part), metadata={"source": source, "source_sha256": digest, **extra})
        os.replace(part, self._converted(source))

    def convert_checkpoint(self) -> Path | None:
        """Re-save AV-HuBERT's encoder (renamed, with its settings) and the fusion network."""
        import torch

        if not self._current(AVHUBERT):
            checkpoint = load_fairseq_checkpoint(self.weights_dir / AVHUBERT)
            hparams = avhubert_hparams(checkpoint)
            self._save(avhubert_state(checkpoint), AVHUBERT, {"hparams": json.dumps(hparams)})
        if not self._current(FUSION):
            import argparse

            # the training arguments sit beside the state dict as an argparse.Namespace
            with torch.serialization.safe_globals([argparse.Namespace]):
                fusion = torch.load(
                    self.weights_dir / FUSION, map_location="cpu", weights_only=True
                )
            self._save(fusion.get("state_dict", fusion), FUSION, {})
        return self._converted(AVHUBERT)

    def load(self, device: str) -> None:
        """Build the encoder and the fusion network and load both strictly."""
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_model

        assert self.upstream_dir is not None
        self.device = resolve_device(device)
        for source in (AVHUBERT, FUSION):
            if not self._current(source):
                raise RuntimeError(
                    f"{self._converted(source)} is stale; run setup_models.py avh_align"
                )
        with safe_open(str(self._converted(AVHUBERT)), "pt") as f:
            hparams = json.loads((f.metadata() or {})["hparams"])
        encoder = build_encoder(hparams)
        load_model(encoder, str(self._converted(AVHUBERT)), strict=True)
        upstream = load_module(Path(self.upstream_dir) / "model.py", "dtb_upstream_avh_align_model")
        fusion = upstream.FusionModel()
        load_model(fusion, str(self._converted(FUSION)), strict=True)
        self.encoder = encoder.eval().to(self.device)
        self.fusion = fusion.eval().to(self.device)
        self._torch = torch
        log.info("AVH-Align loaded on %s", self.device)

    def features(self, mouths: np.ndarray, wave: np.ndarray, sample_rate: int) -> tuple[Any, Any]:
        """AV-HuBERT's video-only and audio-only features, ``[T, dim]`` each."""
        torch = self._torch
        audio = audio_features(wave, sample_rate)
        top = int(round((mouths.shape[1] - _CROP) / 2.0))
        left = int(round((mouths.shape[2] - _CROP) / 2.0))
        frames = mouths[:, top : top + _CROP, left : left + _CROP].astype(np.float32) / 255.0
        frames = (frames - _MEAN) / _STD
        n = min(len(frames), len(audio))
        if n == 0:
            raise PreprocessError("too_short", "no frames with both video and audio")
        video = torch.from_numpy(frames[:n])[None, None].to(self.device)
        audio_t = torch.from_numpy(audio[:n]).T[None].to(self.device)
        with torch.inference_mode():
            f_audio = self.encoder(audio_t, None)[0]
            f_video = self.encoder(None, video)[0]
        return f_video, f_audio

    def score(self, inputs: Mapping[str, Any]) -> np.ndarray:  # type: ignore[override]
        """One score per item: the sigmoid of ``logsumexp(-alignment)`` over its frames."""
        torch = self._torch
        f_video, f_audio = self.features(
            np.asarray(inputs["mouths"]), np.asarray(inputs["audio"]), int(inputs["sample_rate"])
        )
        f_video = f_video / torch.linalg.norm(f_video, ord=2, dim=-1, keepdim=True)
        f_audio = f_audio / torch.linalg.norm(f_audio, ord=2, dim=-1, keepdim=True)
        with torch.inference_mode():
            output = self.fusion(f_video, f_audio)
        score = torch.logsumexp(-output, dim=0).reshape(-1)[0]
        return np.array([float(torch.sigmoid(score.double()))])
