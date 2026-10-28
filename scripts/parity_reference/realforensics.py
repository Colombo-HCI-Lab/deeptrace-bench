"""Reference scores for RealForensics, in upstream's own PyTorch generation.

Runs in a virtualenv with upstream's ``environment.yml`` versions (Python 3.8, torch 1.9.0 on
CPU, torchvision 0.10.0, pytorchvideo 0.1.2, numpy<1.21, pillow<10; docs/src/content/docs/models.md
says how to make it), not in this repo's environment, and doesn't import this package. It
builds upstream's ``csn_temporal_no_head(model_depth=101)`` and ``MeanLinear`` head as
``conf/model/combined.yaml`` configures them, loads ``realforensics_ff.pth`` into them the way
``stage2/eval.py`` does (``backbone.*`` and ``df_head.*``, each with a strict
``load_state_dict``) after checking it against its pinned sha256, and scores each item's
exported 150 x 150 face patches with upstream's test transform: clips of 25 frames back to back,
the last moved back to end at the last frame (pytorchvideo's uniform sampler with
``backpad_last``), each ``x / 255``, torchvision ``CenterCrop(140)`` and ``Resize(112)`` on the
``C, T, H, W`` tensor, pytorchvideo ``Normalize`` with ImageNet's mean and std, then the
sigmoid of the head's logit.

``DTB_PATH_MAP="<from>=<to>,..."`` rewrites the path prefixes in ``parity.json`` when the run
directory was written on another machine.

Usage: <realforensics venv>/bin/python scripts/parity_reference/realforensics.py <run dir>
"""

import hashlib
import importlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from pytorchvideo.transforms import Normalize
from torchvision.transforms import CenterCrop, Compose, Resize


def mapped(path):
    for pair in filter(None, os.environ.get("DTB_PATH_MAP", "").split(",")):
        old, new = pair.split("=", 1)
        if path.startswith(old):
            return new + path[len(old) :]
    return path


run = Path(sys.argv[1]).resolve()
spec = json.loads((run / "parity.json").read_text())
root = Path(mapped(spec["checkout"]))
sys.path.insert(0, str(root))
csn = importlib.import_module("stage2.models.backbones.csn")
linear = importlib.import_module("stage2.models.linear")

backbone = csn.csn_temporal_no_head(model_depth=101, input_channel=3)
df_head = linear.MeanLinear(2048, out_dim=1, norm_linear=True, scale=64)
checkpoint = Path(mapped(spec["weights_dir"])) / "realforensics_ff.pth"
if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != spec["weights"][checkpoint.name]:
    sys.exit(f"{checkpoint} doesn't match its pinned sha256")
state_dict = torch.load(checkpoint, map_location="cpu")
backbone.load_state_dict(
    {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith("backbone")}
)
df_head.load_state_dict(
    {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith("df_head")}
)
backbone.eval()
df_head.eval()

transform = Compose(
    [
        CenterCrop(140),
        Resize(112),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
clip = int(spec["input"]["clip_frames"])
data = np.load(run / "inputs.npz")
units, item_index = data["units"], data["item_index"]
scores = []
with torch.no_grad():
    for i in range(int(item_index.max()) + 1):
        faces = units[item_index == i]
        starts = list(range(0, len(faces) - clip + 1, clip))
        if len(faces) % clip:
            starts.append(len(faces) - clip)
        for s in starts:
            video = torch.as_tensor(faces[s : s + clip]).permute(3, 0, 1, 2).float() / 255.0
            logit = df_head(backbone(transform(video).unsqueeze(0)))
            scores.append(float(torch.sigmoid(logit).reshape(-1)[0]))
np.save(run / "reference.npy", np.asarray(scores, dtype=np.float64))
(run / "reference.json").write_text(
    json.dumps(
        {
            "method": "upstream csn_temporal_no_head + MeanLinear, eval.py load, test transform",
            "env": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": __import__("torchvision").__version__,
            },
        },
        indent=2,
    )
)
print(f"reference scores for {len(scores)} clips in {run}")
