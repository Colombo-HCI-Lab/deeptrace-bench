"""Reference scores for AVAD, running upstream's own ``detect.py`` scoring loop.

Runs in a virtualenv with Python 3.9, torch 1.9.0 on CPU (x86 Linux; its macOS arm64 build has no FFT), numpy<1.24, einops 0.4.1, librosa 0.9.1,
scipy and tqdm (docs/src/content/docs/models.md says how to make it), not in this repo's
environment, and doesn't import this package. It imports the checkout's ``detect.py``, builds the
sync and autoregressive models as its ``main`` does after checking both checkpoints against their
pinned sha256, and calls its ``test2`` with ``lam`` 0.

- ``detect.py``'s training and logging imports become stand-ins, and so does ``torchaudio``, which
  only ``wave2input`` uses, since neither is needed to score.
- The feature branch's result is multiplied by ``lam``, so at 0 it can't reach the score; the
  stand-in feature model and PCA below skip its cost.
- ``test2`` gets a list of exported items in place of its ``DataLoader``. Each goes in as
  ``FakeAVceleb.__getitem__`` leaves it: the frames ``[1, 3, T, 224, 224]`` as float32 on the 0 to
  255 scale, and the audio on the 16-bit scale, trimmed to whole frames and padded by 15 frames'
  worth of zeros on each side.

The sigmoid at the end is the adapter's mapping to a probability, applied to ``test2``'s scores.

Usage: <avad venv>/bin/python scripts/parity_reference/avad.py <run dir>
"""

import hashlib
import json
import os
import pickle
import platform
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock

import librosa  # noqa: F401  (before the stand-ins, which its imports would trip on)
import numpy as np
import sklearn.metrics  # noqa: F401
import torch

_STUBS = (
    "h5py",
    "loguru",
    "matplotlib",
    "matplotlib.pyplot",
    "pandas",
    "configargparse",
    "torch.utils.tensorboard",
    "torchaudio",
    "torchaudio.transforms",
    "torchaudio.functional",
    "fake_celeb_dataset",
    "deep_fake_data",
)

run = Path(sys.argv[1]).resolve()
spec = json.loads((run / "parity.json").read_text())
output = tempfile.mkdtemp()
for name in _STUBS:
    sys.modules[name] = MagicMock()
sys.path.insert(0, spec["checkout"])
os.chdir(spec["checkout"])
unpickle, pickle.load = pickle.load, lambda handle: None  # pca.pkl isn't used
sys.argv = ["detect.py", "--device", "cpu", "--output_dir", output, "--lam", "0"]
import detect  # noqa: E402  (the checkout's)

pickle.load = unpickle
detect.pca = types.SimpleNamespace(transform=lambda features: np.zeros((1, 31)))

weights = Path(spec["weights_dir"])


def checked(name):
    path = weights / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["weights"][name]:
        sys.exit(f"{path} doesn't match its pinned sha256")
    return torch.load(path, map_location="cpu")


class NoFeature(torch.nn.Module):
    def forward(self, video, audio, train=False):
        return torch.zeros(31, 31)


class NoFeatureRegression(torch.nn.Module):
    """The feature regressor's place: lam 0 multiplies its loss away, but its shape must match."""

    def forward(self, features, mask):
        return torch.zeros_like(features)


vis_enc, _ = detect.select_backbone(network="r18")
transformer = detect.MP_AViT(
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
sync = detect.network(vis_enc=vis_enc, aud_enc=detect.AudioEncoder(), transformer=transformer)
sync.load_state_dict(checked("sync_model.pth"))
regressive = detect.transformer_decoder(
    input_dim_old=31,
    input_dim=256,
    compress_factor=1,
    num_heads=16,
    dropout_prob=0.1,
    max_len=49,
    layers=2,
)
regressive.load_state_dict(checked("dist_regressive_model.pth"))

saved = np.load(run / "inputs.npz")
items = []
for index in range(len(spec["items"])):
    faces = saved[f"{index}__faces"].astype(np.float32)
    wave = saved[f"{index}__audio"]
    frames = min(len(faces), len(wave) // 640)
    audio = np.round(wave[: frames * 640] * 32768).astype(np.float32)
    audio = np.pad(audio, (15 * 640, 15 * 640), "constant", constant_values=(0, 0))
    items.append(
        {
            "video": torch.from_numpy(faces[:frames].transpose(3, 0, 1, 2))[None],
            "audio": torch.from_numpy(audio)[None],
            "sample": [str(index)],
        }
    )

detect.test2(
    sync, NoFeature(), items, dist_reg_model=regressive, avfeature_reg_model=NoFeatureRegression()
)
scores = np.load(Path(output) / "testing_scores.npy")
np.save(run / "reference.npy", 1 / (1 + np.exp(-scores.astype(np.float64))))
(run / "reference.json").write_text(
    json.dumps(
        {
            "method": "upstream detect.test2 at lam 0, video by video, sigmoid of its score",
            "env": {"python": platform.python_version(), "torch": torch.__version__},
        },
        indent=2,
    )
)
print(f"reference scores for {len(items)} items in {run}")
