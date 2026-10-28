"""Reference scores for FTCN, in upstream's own PyTorch generation.

Runs in a virtualenv with upstream's stack (Python 3.8, torch 1.4.0 on CPU, fvcore, termcolor,
simplejson, psutil, matplotlib, einops; docs/src/content/docs/models.md says how to make it),
not in this repo's environment, and doesn't import this package. It builds the classifier the
way ``test_on_raw_video.py`` does (upstream's ``config`` from ``root_setting.yaml`` and
``setting/ftcn_tt.yaml``, then ``PluginLoader.get_classifier``), loads the original
``ftcn_tt.pth`` with upstream's own ``load`` after checking every key matches, and scores each
exported clip as that script does: one clip at a time, ImageNet mean and std on the 0 to 255
scale, ``float(output["final_output"])``.

The checkout carries this repo's two compatibility patches; on torch 1.4 the ``nn.Conv3d``
one changes nothing (its signature has no ``device`` or ``dtype``), and the ``np.int`` one is
in ``get_crop_box``, which scoring exported clips doesn't call.

torch 1.4 has no ``weights_only``; the checkpoint is checked against its pinned sha256 before
upstream loads it.

The run directory may be written on one machine and read on another (a Linux x86 worker for
torch 1.4); ``DTB_PATH_MAP="<from>=<to>,..."`` rewrites the path prefixes recorded in
``parity.json``.

Usage: <ftcn venv>/bin/python scripts/parity_reference/ftcn.py <run dir>
"""

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch


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
from config import config as cfg  # noqa: E402  (the checkout's config.py)
from utils.plugin_loader import PluginLoader  # noqa: E402

cfg.init_with_yaml()
cfg.update_with_yaml("ftcn_tt.yaml")
cfg.freeze()
classifier = PluginLoader.get_classifier(cfg.classifier_type)()
classifier.eval()

checkpoint = Path(mapped(spec["weights_dir"])) / "ftcn_tt.pth"
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if digest != spec["weights"][checkpoint.name]:
    sys.exit(f"{checkpoint} doesn't match its pinned sha256")
saved = {
    k[7:] if k.startswith("module") else k: v
    for k, v in torch.load(checkpoint, map_location="cpu").items()
}
expected = classifier.network.state_dict()
if set(saved) != set(expected) or any(saved[k].shape != expected[k].shape for k in saved):
    sys.exit("ftcn_tt.pth doesn't match the network key for key")
classifier.load(str(checkpoint))

mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(1, 3, 1, 1, 1)
std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(1, 3, 1, 1, 1)
units = np.load(run / "inputs.npz")["units"]
scores = []
with torch.no_grad():
    for clip in units:
        images = torch.as_tensor(clip, dtype=torch.float32).permute(3, 0, 1, 2)
        images = images.unsqueeze(0).sub(mean).div(std)
        scores.append(float(classifier(images)["final_output"]))
np.save(run / "reference.npy", np.asarray(scores, dtype=np.float64))
(run / "reference.json").write_text(
    json.dumps(
        {
            "method": "upstream PluginLoader classifier and load(), ftcn_tt.pth, clip by clip",
            "env": {"python": platform.python_version(), "torch": torch.__version__},
        },
        indent=2,
    )
)
print(f"reference scores for {len(units)} clips in {run}")
