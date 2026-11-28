"""Reference scores for DeMamba, with the checkpoint's own ``eval_demamba_standalone.py``.

Runs in the ``xclip`` virtualenv (Python 3.11, torch 2.8.0 on CPU, transformers 4.57.0, opencv,
pandas, scikit-learn, tqdm; the evaluator's own requirements), not in this repo's environment,
and doesn't import this package. It builds ``XCLIP_DeMamba`` and loads ``best_acc.pth`` with the
evaluator's ``load_checkpoint`` after checking the file's pinned sha256, then scores each exported
clip as ``main`` does in fp32: the evaluator's normalisation (ImageNet mean and std on 0-255
values) on the already cropped and resized RGB frames, zero frames appended after normalising
when a clip is short (``make_clips``), and ``sigmoid`` of the one logit.

Usage: <xclip venv>/bin/python scripts/parity_reference/demamba.py <run dir>
"""

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

run = Path(sys.argv[1]).resolve()
spec = json.loads((run / "parity.json").read_text())
sys.path.insert(0, spec["checkout"])
import eval_demamba_standalone as upstream  # noqa: E402  (the checkout's evaluator)

checkpoint = Path(spec["weights_dir"]) / "best_acc.pth"
if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != spec["weights"][checkpoint.name]:
    sys.exit(f"{checkpoint} doesn't match its pinned sha256")
model = upstream.XCLIP_DeMamba()
upstream.load_checkpoint(model, str(checkpoint))
model.eval()

z = np.load(run / "inputs.npz")
units = z["units"] if "units" in z else [z[f"unit_{j}"] for j in range(int(z["n_units"]))]
scores = []
with torch.no_grad():
    for j, clip in enumerate(units):
        frames = [
            ((f.astype(np.float32) - upstream.MEAN) / upstream.STD).transpose(2, 0, 1) for f in clip
        ]
        x = torch.from_numpy(upstream.make_clips(frames, 1))
        scores.append(float(model(x).float()[:, 0].sigmoid()[0]))
        print(f"[{j + 1}/{len(units)}]", flush=True)
np.save(run / "reference.npy", np.asarray(scores, dtype=np.float64))
(run / "reference.json").write_text(
    json.dumps(
        {
            "method": "eval_demamba_standalone.py's XCLIP_DeMamba and load_checkpoint, its"
            " normalisation and make_clips, fp32",
            "env": {"python": platform.python_version(), "torch": torch.__version__},
        },
        indent=2,
    )
)
print(f"reference scores for {len(units)} clips in {run}")
