"""Reference scores for Tak et al.'s XLS-R wrapper models (XLS-R + AASIST, XLS-R + SLS), with real
fairseq.

Runs in a throwaway Python 3.9 virtualenv with upstream's own stack (torch 1.8.1 and the fairseq
commit the checkout vendors, ``fairseq-a54021305d6b...``; fairseq 0.12.2 rejects ``xlsr2_300m.pt``'s
config; docs/src/content/docs/adding_a_model.md says how to make it), not in this repo's
environment, and doesn't import this package. It imports the checkout's own ``model.py``, whose
``SSLModel`` loads fairseq's XLS-R 300M from the relative path ``xlsr2_300m.pt``, so the script runs
from the virtualenv's root, where that file is kept. The original checkpoint (``input.checkpoint``,
not the converted safetensors) then replaces every weight with every key matching, as upstream's
``main_SSL_LA.py`` loads it (``DataParallel``'s ``module.`` prefix stripped, which SLS's checkpoint
carries).

Upstream labels bona fide 1, so P(fake) is the softmax at ``input.fake_index`` (0). On CPU in
fp32 the front end is never moved, so ``SSLModel.extract_feat`` doesn't switch it back to
training mode as it does when it moves the model.

Usage: <fairseq venv>/bin/python scripts/parity_reference/ssl_xlsr.py <run dir>
"""

import importlib.util
import inspect
import json
import os
import platform
import sys
from pathlib import Path

import fairseq
import numpy as np
import torch

run = Path(sys.argv[1]).resolve()
spec = json.loads((run / "parity.json").read_text())
checkout = Path(spec["checkout"])
inputs = spec["input"]
module_path = checkout / inputs.get("module", "model.py")

venv_root = Path(sys.prefix)
if not (venv_root / "xlsr2_300m.pt").exists():
    sys.exit(f"put fairseq's xlsr2_300m.pt in {venv_root}")
os.chdir(venv_root)

module_spec = importlib.util.spec_from_file_location("upstream_model", module_path)
upstream = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(upstream)
model = upstream.Model(None, "cpu")

checkpoint = Path(spec["weights_dir"]) / inputs["checkpoint"]
# torch 1.8, upstream's own, predates ``weights_only``
safe = {"weights_only": True} if "weights_only" in inspect.signature(torch.load).parameters else {}
state = torch.load(checkpoint, map_location="cpu", **safe)
state = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state.items()}
model.load_state_dict(state)
model.eval()

fake_index = int(inputs.get("fake_index", 0))
units = np.load(run / "inputs.npz")["units"]
scores = []
with torch.no_grad():
    for start in range(0, len(units), 8):
        out = model(torch.from_numpy(units[start : start + 8]).float())
        scores.append(torch.softmax(out, dim=-1)[:, fake_index].numpy())
np.save(run / "reference.npy", np.concatenate(scores).astype(np.float64))
(run / "reference.json").write_text(
    json.dumps(
        {
            "method": f"upstream {module_path.name} with fairseq's xlsr2_300m.pt and the "
            f"original {inputs['checkpoint']}",
            "env": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "fairseq": fairseq.__version__,
            },
        },
        indent=2,
    )
)
print(f"reference scores for {len(units)} windows in {run}")
