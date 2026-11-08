"""Reference scores for GenD (CLIP or PE backbone): upstream's README path, not the adapter's.

Runs in this repo's environment, which has upstream's torch (2.8.0) and transformers (4.56.2) and
a newer timm (1.0.30 against 1.0.20), but goes through ``GenD.from_pretrained`` on the weights
folder, as upstream's README does, instead of the adapter's own build. That builds the backbone
the way upstream does, from ``openai/clip-vit-large-patch14`` on the Hub or timm's pretrained PE
weights, before the checkpoint replaces them, so it checks the adapter's pinned snapshot and its
timm patch too.
Each exported crop (native size) goes through ``model.feature_extractor.preprocess`` as a PIL
image, and P(fake) is the softmax at index 1.

Usage: uv run python scripts/parity_reference/gend.py <run dir>
"""

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import transformers
from PIL import Image

run = Path(sys.argv[1]).resolve()
spec = json.loads((run / "parity.json").read_text())
weights = Path(spec["weights_dir"])
checkpoint = weights / "model.safetensors"
if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != spec["weights"][checkpoint.name]:
    sys.exit(f"{checkpoint} doesn't match its pinned sha256")

sys.path.insert(0, spec["checkout"])
from src.hf.modeling_gend import GenD  # noqa: E402  (the checkout's model file)

model = GenD.from_pretrained(str(weights)).eval()

z = np.load(run / "inputs.npz")
n_units = int(z["n_units"]) if "n_units" in z else 0
units = [z[f"unit_{j}"] for j in range(n_units)] if n_units else list(z["units"])
scores = []
with torch.inference_mode():
    for start in range(0, len(units), 32):
        batch = [Image.fromarray(u) for u in units[start : start + 32]]
        pixels = torch.stack([model.feature_extractor.preprocess(img) for img in batch])
        scores.append(model(pixels).softmax(dim=-1)[:, 1].numpy())
np.save(run / "reference.npy", np.concatenate(scores).astype(np.float64))
(run / "reference.json").write_text(
    json.dumps(
        {
            "method": "upstream GenD.from_pretrained on the weights folder, its own preprocess",
            "env": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "timm": timm.__version__,
            },
        },
        indent=2,
    )
)
print(f"reference scores for {len(units)} crops in {run}")
