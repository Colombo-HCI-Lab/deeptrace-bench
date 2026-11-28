"""Reference scores for D3, with upstream's own model and transform.

Runs in the ``xclip`` virtualenv (Python 3.11 with upstream's requirements.txt: torch 2.8.0 on
CPU, transformers 4.57.0, timm, albumentations, opencv), not in this repo's environment, and
doesn't import this package. It builds ``D3_model(encoder_type="XCLIP-16", loss_type="l2")`` as
``eval.py`` does, which loads ``microsoft/xclip-base-patch16`` from the Hub, and turns each
exported clip into upstream's input the way ``data/datasets.py`` does once a frame is read: BGR
(``cv2.imread`` order), ``set_preprocessing(None, None)`` (a 224 resize, a no-op on these 224 px
frames, and ImageNet normalisation in that channel order). The reference score is upstream's
statistic mapped as the adapter maps it, ``1 / (1 + std)``.

Usage: <xclip venv>/bin/python scripts/parity_reference/d3.py <run dir>
"""

import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

run = Path(sys.argv[1]).resolve()
spec = json.loads((run / "parity.json").read_text())
sys.path.insert(0, spec["checkout"])
from data.datasets import set_preprocessing  # noqa: E402  (the checkout's packages)
from models.D3_model import D3_model  # noqa: E402

model = D3_model(encoder_type="XCLIP-16", loss_type="l2").eval()
trans = set_preprocessing(None, None)

z = np.load(run / "inputs.npz")
units = z["units"] if "units" in z else [z[f"unit_{j}"] for j in range(int(z["n_units"]))]
scores = []
with torch.no_grad():
    for j, clip in enumerate(units):
        frames = [
            trans(image=np.ascontiguousarray(f[..., ::-1]))["image"].transpose(2, 0, 1)
            for f in clip
        ]
        x = torch.tensor(np.stack(frames))[None]
        _, _, std = model(x)
        scores.append(float(1.0 / (1.0 + std[0])))
        print(f"[{j + 1}/{len(units)}]", flush=True)
np.save(run / "reference.npy", np.asarray(scores, dtype=np.float64))
(run / "reference.json").write_text(
    json.dumps(
        {
            "method": "upstream D3_model (XCLIP-16, l2) and data/datasets.py's set_preprocessing on"
            " BGR frames; 1 / (1 + std)",
            "env": {"python": platform.python_version(), "torch": torch.__version__},
        },
        indent=2,
    )
)
print(f"reference scores for {len(units)} clips in {run}")
