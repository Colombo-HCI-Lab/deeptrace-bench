"""Reference scores for LipFD, with upstream's reader, network and pinned package versions.

Runs in a virtualenv with Python 3.10, torch 2.1.0 on CPU, torchvision 0.16.0 (upstream's
``requirements.txt``), numpy<2, ftfy, regex and tqdm (docs/src/content/docs/models.md says how to
make it), not in this repo's environment, and doesn't import this package. It builds
``build_model("CLIP:ViT-L/14")`` as ``validate.py`` does, loads ``state_dict["model"]`` from
``ckpt.pth`` strictly after checking its pinned sha256, and scores each exported image through
upstream's own ``AVLip.__getitem__`` and the two calls of ``validate``.

``AVLip`` reads PNGs by path with ``cv2.imread``. A stand-in ``cv2`` gives it the exported image
instead, flipped to BGR as ``imread`` would be; the PNG step is lossless, so nothing else changes.
``AVLip`` is made without its ``__init__``, whose file listing has nothing to list.

Usage: <clip venv>/bin/python scripts/parity_reference/lipfd.py <run dir>
"""

import hashlib
import json
import platform
import sys
import types
from pathlib import Path

import numpy as np
import torch

run = Path(sys.argv[1]).resolve()
spec = json.loads((run / "parity.json").read_text())
units = np.load(run / "inputs.npz")["units"]
sys.modules["cv2"] = types.SimpleNamespace(imread=lambda index: units[index][:, :, ::-1].copy())
sys.path.insert(0, spec["checkout"])
from data.datasets import AVLip  # noqa: E402  (the checkout's)
from models import build_model  # noqa: E402

checkpoint = Path(spec["weights_dir"]) / "ckpt.pth"
digest = hashlib.sha256()
with open(checkpoint, "rb") as handle:
    while chunk := handle.read(1 << 24):
        digest.update(chunk)
if digest.hexdigest() != spec["weights"][checkpoint.name]:
    sys.exit(f"{checkpoint} doesn't match its pinned sha256")
model = build_model("CLIP:ViT-L/14")
model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
model.eval()

dataset = AVLip.__new__(AVLip)
dataset.total_list = list(range(len(units)))
dataset.label_dict = dict.fromkeys(dataset.total_list, 0)
scores = []
with torch.no_grad():
    for index in range(len(units)):
        img, crops, _ = dataset[index]
        features = model.get_features(img[None])
        crops = [[t[None] for t in sublist] for sublist in crops]
        scores.extend(model(crops, features)[0].sigmoid().flatten().tolist())
        if index % 20 == 0:
            print(f"[{index + 1}/{len(units)}]", flush=True)
np.save(run / "reference.npy", np.asarray(scores, dtype=np.float64))
(run / "reference.json").write_text(
    json.dumps(
        {
            "method": "upstream AVLip.__getitem__ and build_model, ckpt.pth, image by image",
            "env": {"python": platform.python_version(), "torch": torch.__version__},
        },
        indent=2,
    )
)
print(f"reference scores for {len(units)} images in {run}")
