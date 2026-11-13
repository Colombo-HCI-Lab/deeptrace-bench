"""Reference scores for LipForensics, in upstream's own PyTorch generation.

Runs in a virtualenv with upstream's ``requirements.txt`` versions (Python 3.9, torch 1.9.0 and
torchvision 0.10.0 on CPU, numpy<1.24), not in this repo's environment, and doesn't import this
package. It builds the network with upstream's ``get_model`` from the checkout, then loads
``lipforensics_ff.pth``'s ``"model"`` state dict strictly, as ``get_model`` does but onto the CPU
(``get_model`` maps the checkpoint to a CUDA device). Each item's exported 96 x 96 grayscale mouth
crops are cut into non-overlapping clips of ``input.clip_frames`` and go through ``evaluate.py``'s
own transform (``ToTensorVideo``, torchvision ``CenterCrop((88, 88))``, ``NormalizeVideo`` with
0.421 and 0.165), as ``ForensicsClips.__getitem__`` feeds it; P(fake) is the sigmoid of each clip's
logit. The mouth crops themselves come from the adapter, so this checks the network, weights and
transform, not upstream's landmark pipeline.

Usage: <lipforensics venv>/bin/python scripts/parity_reference/lipforensics.py <run dir>
"""

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.transforms import CenterCrop, Compose

run = Path(sys.argv[1]).resolve()
spec = json.loads((run / "parity.json").read_text())
checkout = Path(spec["checkout"])
os.chdir(checkout)  # get_model reads its config by a relative path
sys.path.insert(0, str(checkout))
from data.transforms import NormalizeVideo, ToTensorVideo  # noqa: E402
from models.spatiotemporal_net import get_model  # noqa: E402

checkpoint = Path(spec["weights_dir"]) / "lipforensics_ff.pth"
if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != spec["weights"][checkpoint.name]:
    sys.exit(f"{checkpoint} doesn't match its pinned sha256")
model = get_model(device="cpu")
model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])
model.eval()

transform = Compose([ToTensorVideo(), CenterCrop((88, 88)), NormalizeVideo((0.421,), (0.165,))])
clip = int(spec["input"]["clip_frames"])
data = np.load(run / "inputs.npz")
units, item_index = data["units"], data["item_index"]
scores = []
with torch.no_grad():
    for i in range(int(item_index.max()) + 1):
        mouths = units[item_index == i]
        clips = [mouths[s : s + clip] for s in range(0, len(mouths) - clip + 1, clip)]
        batch = torch.stack([transform(torch.from_numpy(c).unsqueeze(-1)) for c in clips])
        logits = model(batch, lengths=[clip] * len(clips))
        scores.append(torch.sigmoid(logits).reshape(-1).numpy())
np.save(run / "reference.npy", np.concatenate(scores).astype(np.float64))
(run / "reference.json").write_text(
    json.dumps(
        {
            "method": "upstream get_model and lipforensics_ff.pth strict, evaluate.py's transform",
            "env": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
            },
        },
        indent=2,
    )
)
print(f"reference scores for {sum(len(s) for s in scores)} clips in {run}")
