"""Reference scores for UnivFD, with upstream's own code and OpenAI's CLIP checkpoint.

Runs in a virtualenv with Python 3.10, torch 2.1.0 on CPU, torchvision 0.16.0, numpy<2, ftfy, regex
and tqdm (docs/src/content/docs/models.md says how to make it), not in this repo's environment, and
doesn't import this package. It builds ``get_model("CLIP:ViT-L/14")`` as ``validate.py`` does, which
downloads OpenAI's ``ViT-L-14.pt`` once, loads ``fc_weights.pth`` into ``model.fc`` after checking
its pinned sha256, and scores each exported crop with ``RealFakeDataset``'s transform (centre crop
to 224, ``ToTensor``, CLIP's normalisation) and ``model(x).sigmoid()``, as ``validate`` does.

Usage: <clip venv>/bin/python scripts/parity_reference/univfd.py <run dir>
"""

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

run = Path(sys.argv[1]).resolve()
spec = json.loads((run / "parity.json").read_text())
sys.path.insert(0, spec["checkout"])
from models import get_model  # noqa: E402  (the checkout's models package)

checkpoint = Path(spec["weights_dir"]) / "fc_weights.pth"
if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != spec["weights"][checkpoint.name]:
    sys.exit(f"{checkpoint} doesn't match its pinned sha256")
model = get_model("CLIP:ViT-L/14")
model.fc.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
model.eval()

transform = transforms.Compose(
    [
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ]
)
units = np.load(run / "inputs.npz")["units"]
scores = []
with torch.no_grad():
    for start in range(0, len(units), 32):
        batch = torch.stack([transform(Image.fromarray(u)) for u in units[start : start + 32]])
        scores.extend(model(batch).sigmoid().flatten().tolist())
        print(f"[{min(start + 32, len(units))}/{len(units)}]", flush=True)
np.save(run / "reference.npy", np.asarray(scores, dtype=np.float64))
(run / "reference.json").write_text(
    json.dumps(
        {
            "method": "upstream get_model and fc_weights.pth, OpenAI ViT-L-14.pt, RealFakeDataset's"
            " transform",
            "env": {"python": platform.python_version(), "torch": torch.__version__},
        },
        indent=2,
    )
)
print(f"reference scores for {len(units)} crops in {run}")
