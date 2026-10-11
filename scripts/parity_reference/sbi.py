"""Reference scores for SBI, in an old PyTorch generation.

Runs in the DeepfakeBench reference virtualenv (Python 3.9, torch 1.13, efficientnet-pytorch
0.7.1; docs/adding_a_model.md says how to make it), not in this repo's environment, and doesn't
import this package. It builds upstream's own ``Detector`` from ``src/inference/model.py`` in
the checkout, loads the original ``FFc23.tar`` (its ``"model"`` entry, not the converted
safetensors) with every key matching, and scores the exported crops as
``src/inference/inference_video.py`` does: RGB scaled to [0, 1], softmax index 1.

``Detector`` calls ``EfficientNet.from_pretrained``, which downloads ImageNet weights the
checkpoint then replaces; it is pointed at ``from_name`` while the network is built, as in the
adapter, so nothing is fetched and the network is the same.

torch 1.13's ``weights_only`` unpickler refuses ``FFc23.tar`` (torch 2.8's, which the adapter
uses, accepts it), so the file is first checked against its pinned sha256 and then loaded in
full: the bytes are the ones the adapter's safe load already read.

Usage: $DTB_ROOT/parity/envs/dfb/bin/python scripts/parity_reference/sbi.py <run dir>
"""

import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from efficientnet_pytorch import EfficientNet

run = Path(sys.argv[1]).resolve()
spec = json.loads((run / "parity.json").read_text())
module_path = Path(spec["checkout"]) / "src" / "inference" / "model.py"
module_spec = importlib.util.spec_from_file_location("sbi_model", module_path)
upstream = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(upstream)

original = EfficientNet.__dict__["from_pretrained"]
EfficientNet.from_pretrained = staticmethod(
    lambda name, weights_path=None, advprop=False, **kw: EfficientNet.from_name(name, **kw)
)
try:
    model = upstream.Detector()
finally:
    EfficientNet.from_pretrained = original

checkpoint = Path(spec["weights_dir"]) / spec["input"].get("checkpoint", "FFc23.tar")
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if digest != spec["weights"][checkpoint.name]:
    sys.exit(f"{checkpoint} doesn't match its pinned sha256")
model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
model.eval()

units = np.load(run / "inputs.npz")["units"]
scores = []
with torch.no_grad():
    for start in range(0, len(units), 16):
        x = torch.from_numpy(units[start : start + 16].transpose(0, 3, 1, 2)).float() / 255
        scores.append(model(x).softmax(1)[:, 1].numpy())
np.save(run / "reference.npy", np.concatenate(scores).astype(np.float64))
(run / "reference.json").write_text(
    json.dumps(
        {
            "method": f"upstream src/inference/model.py Detector and {checkpoint.name}",
            "env": {"python": platform.python_version(), "torch": torch.__version__},
        },
        indent=2,
    )
)
print(f"reference scores for {len(units)} crops in {run}")
