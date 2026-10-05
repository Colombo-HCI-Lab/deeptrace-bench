"""LipForensics (Haliassos et al., CVPR 2021, ahaliassos/LipForensics, MIT): a lipreading
network (ResNet-18 front end with a 3D stem, multi-scale TCN back end) pretrained on LRW and
fine-tuned on FF++ to spot unnatural mouth movement. Relevant to lip-sync fakes.

The adapter imports ``models/`` from the pinned checkout as a package and builds ``Lipreading``
the way upstream's ``get_model`` does, from the checkout's ``lrw_resnet18_mstcn.json``, with
one output. ``lipforensics_ff.pth`` is ``{"model": state_dict}``; ``convert_checkpoint``
re-saves the state dict, which then loads with every key matching (upstream's own loader
moves it to ``cuda:0`` first, which this avoids).

Inputs come from ``preprocess.mouths.MouthLoader`` (consecutive frames, FAN landmarks, mean-
face warp, 96 x 96 grayscale mouths). As upstream's evaluation: non-overlapping clips of 25
frames, scaled to [0, 1], centre-cropped to 88 x 88 and normalised with mean 0.421 and std
0.165. Output: one logit per clip, label 1 fake; P(fake) is its sigmoid. Upstream averages a
video's clip logits before the AUC; the harness averages probabilities, as for every model.
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..preprocess import PreprocessError
from ._upstream import load_package
from .base import Detector, resolve_device

log = logging.getLogger(__name__)

_PACKAGE = "dtb_upstream_lipforensics_models"
_MEAN, _STD = 0.421, 0.165
_CROP = 88
_BATCH = 8


class LipForensicsDetector(Detector):
    """LipForensics on 25-frame mouth clips."""

    @property
    def source(self) -> str:
        return self.config.weights[0].name

    def _build(self) -> Any:
        assert self.upstream_dir is not None
        folder = Path(self.upstream_dir) / "models"
        try:
            load_package(folder, _PACKAGE)
        except FileNotFoundError:
            msg = f"{folder} is missing; run scripts/setup_models.py {self.config.id}"
            raise FileNotFoundError(msg) from None
        net = importlib.import_module(f"{_PACKAGE}.spatiotemporal_net")
        args = json.loads((folder / "configs" / "lrw_resnet18_mstcn.json").read_text())
        tcn_options = {
            "num_layers": args["tcn_num_layers"],
            "kernel_size": args["tcn_kernel_size"],
            "dropout": args["tcn_dropout"],
            "dwpw": args["tcn_dwpw"],
            "width_mult": args["tcn_width_mult"],
        }
        return net.Lipreading(num_classes=1, tcn_options=tcn_options, relu_type=args["relu_type"])

    def convert_checkpoint(self) -> Path | None:
        """Re-save the checkpoint's ``"model"`` state dict as safetensors."""
        import torch

        if self.converted_is_current(self.source):
            return self.converted_path
        saved = torch.load(self.weights_dir / self.source, map_location="cpu", weights_only=True)
        return self.save_converted(saved["model"], self.source)

    def load(self, device: str) -> None:
        """Build the network and load the converted weights, every key matching."""
        import torch

        self.device = resolve_device(device)
        model = self._build()
        self.load_converted(model, self.source)
        self.model = model.eval().to(self.device)
        self.clip = int(self.config.input["clip_frames"])
        self._torch = torch
        log.info("LipForensics loaded on %s", self.device)

    def score(self, inputs: np.ndarray) -> np.ndarray:  # type: ignore[override]
        """P(fake) per non-overlapping clip of ``clip_frames`` mouth crops."""
        torch = self._torch
        mouths = np.asarray(inputs)
        n_clips = len(mouths) // self.clip
        if n_clips == 0:
            raise PreprocessError(
                "too_short", f"{len(mouths)} mouth frames, a clip needs {self.clip}"
            )
        clips = mouths[: n_clips * self.clip].reshape(n_clips, self.clip, *mouths.shape[1:])
        top = int(round((clips.shape[2] - _CROP) / 2.0))
        left = int(round((clips.shape[3] - _CROP) / 2.0))
        clips = clips[:, :, top : top + _CROP, left : left + _CROP]
        x = (clips.astype(np.float32) / 255.0 - _MEAN) / _STD
        scores = []
        for start in range(0, n_clips, _BATCH):
            batch = torch.from_numpy(x[start : start + _BATCH]).unsqueeze(1).to(self.device)
            with torch.inference_mode():
                logits = self.model(batch, lengths=[self.clip] * len(batch))
            scores.append(torch.sigmoid(logits.float()).reshape(-1).cpu().numpy())
        return np.concatenate(scores)
