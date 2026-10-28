"""FTCN (Zheng et al., ICCV 2021), a fully temporal 3D ResNet-50 with a temporal transformer.

Upstream states no licence, so its code stays in ``third_party/`` and is imported from the
pinned checkout. The network is built the way ``test_on_raw_video.py`` builds it: upstream's
``config`` initialised from ``root_setting.yaml`` and ``setting/ftcn_tt.yaml``, then
``PluginLoader.get_classifier("i3d_temporal_var_fix_dropout_tt_cfg")``. Those files import
by top-level names (``config``, ``utils``, ``model``, ``slowfast``, ``trainer``) from the
checkout's root, so the root goes on ``sys.path`` while they import and the names are taken
out of ``sys.modules`` afterwards. Training-only imports become stand-ins for that time:

- ``termcolor``, ``simplejson``, ``psutil`` and ``matplotlib`` (logging, meters and plots);
- ``fvcore``: its ``CfgNode`` is yacs' (fvcore's is a thin subclass of it, and FTCN's own
  subclass overrides the one method fvcore adds that it calls), its ``Registry`` a dict, and
  its weight-init, file-io and FLOP helpers, which inference never calls, are stubs. The
  initial weights don't matter: a strict load replaces every one.

Two compatibility patches (``patches/FTCN/``): the network copies each ``nn.Conv3d``'s
constructor arguments by reading its signature, which on current PyTorch includes ``device``
and ``dtype`` that the module doesn't store; and ``get_crop_box`` used the removed
``np.int``.

Weights: ``ftcn_tt.pth``, a plain state dict (keys ``resnet.*``), re-saved as safetensors.

Input: aligned 224 x 224 RGB clips of 32 frames from ``preprocess.clips.FaceClipLoader``,
normalised with ImageNet's mean and std on the 0 to 255 scale, as upstream does. Output: the
sigmoid of the temporal transformer's one logit, P(fake), one score per clip.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import types
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ._upstream import scoped_modules
from .base import Detector, resolve_device

log = logging.getLogger(__name__)

_CLASSIFIER = "i3d_temporal_var_fix_dropout_tt_cfg"
_SETTING = "ftcn_tt.yaml"
# FTCN's top-level packages and modules, imported from the checkout root
_UPSTREAM_NAMES = ["config", "utils", "model", "slowfast", "trainer"]
_MEAN = np.array([0.485, 0.456, 0.406]) * 255
_STD = np.array([0.229, 0.224, 0.225]) * 255
_BATCH = 4


def _module(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


class _Registry:
    """fvcore's ``Registry``: names to classes, filled by its ``register`` decorator."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._items: dict[str, Any] = {}

    def register(self, obj: Any = None) -> Any:
        if obj is None:
            return self.register
        self._items[obj.__name__] = obj
        return obj

    def get(self, name: str) -> Any:
        return self._items[name]


def _stubs() -> dict[str, types.ModuleType]:
    from yacs.config import CfgNode

    def unused(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("a training-only FTCN helper was called")

    return {
        "termcolor": _module("termcolor", colored=lambda text, *a, **k: text),
        "simplejson": _module("simplejson", dumps=lambda obj, **kwargs: json.dumps(obj)),
        "psutil": _module("psutil", virtual_memory=unused),
        "matplotlib": _module("matplotlib"),
        "matplotlib.pyplot": _module("matplotlib.pyplot"),
        "fvcore": _module("fvcore"),
        "fvcore.common": _module("fvcore.common"),
        "fvcore.common.config": _module("fvcore.common.config", CfgNode=CfgNode),
        "fvcore.common.registry": _module("fvcore.common.registry", Registry=_Registry),
        "fvcore.common.file_io": _module("fvcore.common.file_io", PathManager=None),
        "fvcore.nn": _module("fvcore.nn"),
        "fvcore.nn.activation_count": _module(
            "fvcore.nn.activation_count", activation_count=unused
        ),
        "fvcore.nn.flop_count": _module("fvcore.nn.flop_count", flop_count=unused),
        "fvcore.nn.weight_init": _module("fvcore.nn.weight_init", c2_msra_fill=lambda m: None),
    }


@contextlib.contextmanager
def upstream_imports(root: Path) -> Iterator[None]:
    """Make FTCN's checkout importable by its own top-level names, for one ``with`` block."""
    stubs = _stubs()
    sys.path.insert(0, str(root))
    try:
        with scoped_modules(stubs, purge=_UPSTREAM_NAMES):
            yield
    finally:
        sys.path.remove(str(root))


def build_network(root: Path) -> Any:
    """Upstream's classifier network, built as ``test_on_raw_video.py`` builds it."""
    if not (root / "config.py").is_file():
        raise FileNotFoundError(f"{root} is missing; run scripts/setup_models.py ftcn")
    with upstream_imports(root):
        # The checkout's config.py; a fresh module each time, since the block purges it.
        from config import config as cfg
        from utils.plugin_loader import PluginLoader

        cfg.init_with_yaml()
        cfg.update_with_yaml(_SETTING)
        cfg.freeze()
        # Building logs (through upstream's "tensorpack" logger) and prints every layer it
        # replaces.
        upstream_log = logging.getLogger("tensorpack")
        level = upstream_log.level
        upstream_log.setLevel(logging.WARNING)
        try:
            with open(os.devnull, "w") as null, contextlib.redirect_stdout(null):
                classifier = PluginLoader.get_classifier(_CLASSIFIER)()
        finally:
            upstream_log.setLevel(level)
    return classifier.network


class FTCNDetector(Detector):
    """FTCN+TT on aligned 32-frame face clips."""

    @property
    def source(self) -> str:
        return self.config.weights[0].name

    def convert_checkpoint(self) -> Path | None:
        """Re-save the state dict as safetensors."""
        import torch

        if self.converted_is_current(self.source):
            return self.converted_path
        state = torch.load(self.weights_dir / self.source, map_location="cpu", weights_only=True)
        return self.save_converted(state, self.source)

    def load(self, device: str) -> None:
        """Build upstream's network and load the converted weights, every key matching."""
        import torch

        assert self.upstream_dir is not None
        self.device = resolve_device(device)
        network = build_network(Path(self.upstream_dir))
        self.load_converted(network, self.source)
        self.model = network.eval().to(self.device)
        self._torch = torch
        shape = (1, 3, 1, 1, 1)
        self._mean = torch.tensor(_MEAN, dtype=torch.float32).view(shape).to(self.device)
        self._std = torch.tensor(_STD, dtype=torch.float32).view(shape).to(self.device)
        log.info("FTCN loaded on %s", self.device)

    def score(self, inputs: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        """P(fake) per clip (``[clip_frames, 224, 224, 3]`` RGB uint8 each)."""
        torch = self._torch
        clips = np.asarray(inputs)
        scores = []
        for start in range(0, len(clips), _BATCH):
            batch = torch.from_numpy(clips[start : start + _BATCH]).to(self.device)
            x = batch.permute(0, 4, 1, 2, 3).float().sub(self._mean).div(self._std)
            with torch.inference_mode():
                out = self.model(x)["final_output"]
            scores.append(out.reshape(-1).float().cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0)
