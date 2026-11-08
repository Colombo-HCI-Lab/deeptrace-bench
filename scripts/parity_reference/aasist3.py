"""Reference scores for AASIST3: upstream's own loading path and preprocessing, not the adapter's.

Runs in a virtualenv with upstream's pinned stack (Python 3.12, torch 2.6.0 and torchaudio 2.6.0 on
CPU, transformers 4.52.4, huggingface-hub 0.32.4; the parity stage's ``aasist3`` stack), and doesn't
import this package. It loads the model the way upstream's README does, ``aasist3.from_pretrained``
on a folder holding the Hub's ``config.json`` and ``model.safetensors`` (here under their pinned
local names, linked into a scratch folder), which also fetches the front end's config from
``facebook/wav2vec2-large-xlsr-53`` unpinned, as upstream does. Each exported window gets upstream's
``torchaudio.functional.preemphasis`` (0.97), and P(fake) is the softmax at index 0, since
upstream's datasets label spoof 0.

Usage: <aasist3 venv>/bin/python scripts/parity_reference/aasist3.py <run dir>
"""

import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio
import transformers

run = Path(sys.argv[1]).resolve()
spec = json.loads((run / "parity.json").read_text())
weights = Path(spec["weights_dir"])
checkpoint = weights / "aasist3.safetensors"
if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != spec["weights"][checkpoint.name]:
    sys.exit(f"{checkpoint} doesn't match its pinned sha256")

sys.path.insert(0, spec["checkout"])
from model import aasist3  # noqa: E402  (the checkout's model package)

with tempfile.TemporaryDirectory() as scratch:
    folder = Path(scratch) / "aasist3"
    folder.mkdir()
    (folder / "model.safetensors").symlink_to(checkpoint)
    (folder / "config.json").symlink_to(weights / "aasist3_config.json")
    os.chdir(scratch)  # the front end caches its config under "weights/", relative to here
    model = aasist3.from_pretrained(str(folder))
model.eval()

z = np.load(run / "inputs.npz")
units = torch.from_numpy(z["units"]).float()
scores = []
with torch.inference_mode():
    for start in range(0, len(units), 16):
        x = torchaudio.functional.preemphasis(units[start : start + 16])
        scores.append(model(x).softmax(dim=-1)[:, 0].numpy())
np.save(run / "reference.npy", np.concatenate(scores).astype(np.float64))
(run / "reference.json").write_text(
    json.dumps(
        {
            "method": "upstream aasist3.from_pretrained on the Hub files, torchaudio preemphasis",
            "env": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
        },
        indent=2,
    )
)
print(f"reference scores for {len(units)} windows in {run}")
