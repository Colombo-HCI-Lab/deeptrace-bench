"""Reference scores for AASIST / AASIST-L: upstream's code path, not the adapter's.

Builds the model the way upstream's ``main.py`` does (``import_module("models.<arch>")`` from
the checkout root, ``config/<variant>.conf``), loads the original ``.pth`` with
``torch.load``, and runs it on the exported windows in eval mode. Upstream labels bona fide
1, so P(fake) is the softmax at index 0. Runs in this repo's environment: the model file is
plain PyTorch that upstream's own (older) environment would run unchanged.

Usage: uv run python scripts/parity_reference/aasist.py <run dir>
"""

import json
import platform
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

run = Path(sys.argv[1])
spec = json.loads((run / "parity.json").read_text())
checkout = Path(spec["checkout"])
variant = spec["input"]["variant"]
config = json.loads((checkout / "config" / f"{variant}.conf").read_text())["model_config"]

sys.path.insert(0, str(checkout))
model = import_module(f"models.{config['architecture']}").Model(config)
checkpoint = Path(spec["weights_dir"]) / f"{variant}.pth"
model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
model.eval()

units = np.load(run / "inputs.npz")["units"]
scores = []
with torch.no_grad():
    for start in range(0, len(units), 24):  # upstream's batch size
        _, out = model(torch.from_numpy(units[start : start + 24]).float())
        scores.append(torch.softmax(out, dim=1)[:, 0].numpy())
np.save(run / "reference.npy", np.concatenate(scores).astype(np.float64))
(run / "reference.json").write_text(
    json.dumps(
        {
            "method": "upstream models/AASIST.py and the original .pth, built as main.py does",
            "env": {"python": platform.python_version(), "torch": torch.__version__},
        },
        indent=2,
    )
)
print(f"reference scores for {len(units)} windows in {run}")
