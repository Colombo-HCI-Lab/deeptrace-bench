"""Reference scores for DF Arena: the model card's own pipeline, remote code and all.

Loads the model exactly as its card says (``pipeline("antispoofing", trust_remote_code=True)``)
at the pinned revision: the Hub's code, the original ``pytorch_model.bin`` and whatever XLS-R
config the Hub serves, which the adapter pins instead. Each exported window goes through the
card's feature extractor and model one at a time; P(fake) is the pipeline's ``spoof`` score.
Needs the network.

Usage: uv run python scripts/parity_reference/df_arena.py <run dir>
"""

import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import pipeline

run = Path(sys.argv[1])
spec = json.loads((run / "parity.json").read_text())
upstream = spec["upstream"]
pipe = pipeline(
    "antispoofing",
    model=upstream["repo"],
    revision=upstream["commit"],
    trust_remote_code=True,
    device="cpu",
)
units = np.load(run / "inputs.npz")["units"]
scores = [pipe(window)["all_scores"]["spoof"] for window in units]
np.save(run / "reference.npy", np.asarray(scores, dtype=np.float64))
(run / "reference.json").write_text(
    json.dumps(
        {
            "method": "the model card's pipeline with trust_remote_code at the pinned revision",
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
