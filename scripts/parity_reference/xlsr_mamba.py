"""Reference scores for XLSR-Mamba with real fairseq and mamba-ssm's CUDA kernels, on a GPU.

Runs in a throwaway Python 3.10 virtualenv with upstream's own stack (requirements.txt: torch 2.2.1
for CUDA 11.8, mamba-ssm 1.1.4 and causal-conv1d 1.1.3.post1, fairseq at the commit the README
names, ``a54021305d6b3c``), not in this repo's environment, and doesn't import this package. It
imports the checkout's own ``model.py``, whose ``SSLModel`` loads fairseq's XLS-R 300M from the
relative path ``xlsr2_300m.pt``, so the script runs from the virtualenv's root, where that file is
kept. The released safetensors checkpoint then replaces every weight, strictly.

The adapter swaps the kernels for pure-PyTorch stand-ins; this is the check against the kernels
themselves. The model is moved to the GPU before the first call, as upstream's ``main.py`` does,
so ``SSLModel.extract_feat`` doesn't switch the front end back to training mode. TF32 is off, as
the adapter's ``compute.tf32`` keeps it. Upstream scores ``batch_out[:, 1]``, bona fide, so P(fake)
is the softmax at ``input.fake_index`` (0).

Usage: <mamba venv>/bin/python scripts/parity_reference/xlsr_mamba.py <run dir>
"""

import importlib.util
import json
import os
import platform
import sys
import types
from pathlib import Path

import fairseq
import mamba_ssm
import numpy as np
import torch
from safetensors.torch import load_file

run = Path(sys.argv[1]).resolve()
spec = json.loads((run / "parity.json").read_text())
checkout = Path(spec["checkout"])
inputs = spec["input"]

if not torch.cuda.is_available():
    sys.exit("mamba-ssm's kernels need a CUDA GPU")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

venv_root = Path(sys.prefix)
if not (venv_root / "xlsr2_300m.pt").exists():
    sys.exit(f"put fairseq's xlsr2_300m.pt in {venv_root}")
os.chdir(venv_root)
sys.path.insert(0, str(checkout))  # model.py imports mamba_blocks as a top-level module

module_spec = importlib.util.spec_from_file_location("upstream_model", checkout / "model.py")
upstream = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(upstream)
args = types.SimpleNamespace(emb_size=int(inputs["emb_size"]), num_encoders=int(inputs["num_encoders"]))
model = upstream.Model(args, "cuda")
model.load_state_dict(load_file(Path(spec["weights_dir"]) / inputs["checkpoint"]), strict=True)
model.to("cuda")
model.eval()

fake_index = int(inputs.get("fake_index", 0))
units = np.load(run / "inputs.npz")["units"]
scores = []
with torch.no_grad():
    for start in range(0, len(units), 8):
        batch = torch.from_numpy(units[start : start + 8]).float().cuda()
        scores.append(torch.softmax(model(batch), dim=-1)[:, fake_index].cpu().numpy())
np.save(run / "reference.npy", np.concatenate(scores).astype(np.float64))
(run / "reference.json").write_text(
    json.dumps(
        {
            "method": "upstream model.py with fairseq's xlsr2_300m.pt, mamba-ssm's CUDA kernels "
            f"and the released {inputs['checkpoint']}",
            "env": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "fairseq": fairseq.__version__,
                "mamba_ssm": mamba_ssm.__version__,
                "gpu": torch.cuda.get_device_name(0),
            },
        },
        indent=2,
    )
)
print(f"reference scores for {len(units)} windows in {run}")
