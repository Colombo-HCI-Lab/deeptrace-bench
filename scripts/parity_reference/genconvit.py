"""Reference scores for GenConViT's ED branch, in upstream's own timm generation.

Runs in a virtualenv with upstream's pinned timm (Python 3.9, torch 1.13.1, torchvision 0.14.1,
timm 0.6.5, pyyaml; docs/src/content/docs/models.md says how to make it), not in this repo's
environment, and doesn't import this package. It builds upstream's own ``GenConViT`` with
``net="ed"``, which loads the original ``genconvit_ed_inference.pth`` itself with a strict
``load_state_dict`` and no key conversion (timm 0.6.5 is the layout it was saved in), and
scores the exported crops as ``model/pred_func.py`` does: each frame scaled to [0, 1] and
normalised with ImageNet's mean and std, then the sigmoid of the logits; class 0 is fake.

``GenConViTED`` asks timm for ImageNet weights the checkpoint then replaces; ``create_model``
is told ``pretrained=False`` while the network is built, so nothing is fetched. Upstream opens
``weight/<name>.pth`` and ``model/config.yaml`` relative to the working directory, so the script
runs from the checkout and names the checkpoint by a path relative to its ``weight/`` folder,
after checking the file against its pinned sha256.

Usage: <genconvit venv>/bin/python scripts/parity_reference/genconvit.py <run dir>
"""

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from torchvision import transforms

run = Path(sys.argv[1]).resolve()
spec = json.loads((run / "parity.json").read_text())
root = Path(spec["checkout"])
checkpoint = Path(spec["weights_dir"]) / "genconvit_ed_inference.pth"
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if digest != spec["weights"][checkpoint.name]:
    sys.exit(f"{checkpoint} doesn't match its pinned sha256")

os.chdir(root)
sys.path.insert(0, str(root))
original = timm.create_model


def no_download(name, pretrained=False, **kwargs):
    return original(name, pretrained=False, **kwargs)


timm.create_model = no_download
import model.genconvit_ed as genconvit_ed  # noqa: E402  (the checkout's model package)

genconvit_ed.timm.create_model = no_download
from model.config import load_config  # noqa: E402
from model.genconvit import GenConViT  # noqa: E402

ed_name = os.path.relpath(checkpoint.with_suffix(""), root / "weight")
network = GenConViT(load_config(), ed=ed_name, vae=None, net="ed", fp16=False)
network.eval()
timm.create_model = original

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
units = np.load(run / "inputs.npz")["units"]
scores = []
with torch.no_grad():
    for start in range(0, len(units), 32):
        df = torch.tensor(units[start : start + 32]).float().permute((0, 3, 1, 2))
        for i in range(len(df)):
            df[i] = normalize(df[i] / 255.0)
        scores.append(torch.sigmoid(network(df))[:, 0].numpy())
np.save(run / "reference.npy", np.concatenate(scores).astype(np.float64))
(run / "reference.json").write_text(
    json.dumps(
        {
            "method": "upstream GenConViT(net='ed') and its own load, pred_func preprocessing",
            "env": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "timm": timm.__version__,
            },
        },
        indent=2,
    )
)
print(f"reference scores for {len(units)} crops in {run}")
