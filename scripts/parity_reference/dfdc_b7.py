"""Reference scores for the DFDC winner's seven-model ensemble, in upstream's own stack.

Runs in a virtualenv with upstream's Dockerfile versions (Python 3.8, torch 1.10.0 on CPU,
torchvision 0.11.1, timm 0.4.12; docs/src/content/docs/models.md says how to make it), not in
this repo's environment, and doesn't import this package. For each checkpoint it builds
upstream's own ``DeepFakeClassifier(encoder="tf_efficientnet_b7_ns")`` and loads it as
``predict_folder.py`` does (``"state_dict"``, ``module.`` stripped, ``strict=True``), after
checking the file against its pinned sha256. The exported crops are already letterboxed to
380 x 380 by the shared loader, so it scores them as ``predict_on_video`` does from there: each
frame scaled to [0, 1] and normalised with ImageNet's mean and std, every model's sigmoid, then
the mean over models. Upstream runs the models in fp16 on CUDA; this runs fp32 on CPU, as the
adapter does.

``classifiers.py`` builds its encoders with ``pretrained=True``, which would download ImageNet
weights the checkpoints then replace; the EfficientNet constructors are told
``pretrained=False`` while it imports, so nothing is fetched.

``DTB_PATH_MAP="<from>=<to>,..."`` rewrites the path prefixes in ``parity.json`` when the run
directory was written on another machine.

Usage: <dfdc venv>/bin/python scripts/parity_reference/dfdc_b7.py <run dir>
"""

import hashlib
import importlib.util
import json
import os
import platform
import re
import sys
from pathlib import Path

import numpy as np
import timm
import timm.models.efficientnet as efficientnet
import torch
from torchvision.transforms import Normalize


def mapped(path):
    for pair in filter(None, os.environ.get("DTB_PATH_MAP", "").split(",")):
        old, new = pair.split("=", 1)
        if path.startswith(old):
            return new + path[len(old) :]
    return path


run = Path(sys.argv[1]).resolve()
spec = json.loads((run / "parity.json").read_text())
root = Path(mapped(spec["checkout"]))
weights_dir = Path(mapped(spec["weights_dir"]))

for i in range(2, 8):
    name = f"tf_efficientnet_b{i}_ns"
    build = getattr(efficientnet, name)
    setattr(efficientnet, name, lambda pretrained=False, _b=build, **kw: _b(pretrained=False, **kw))
path = root / "training" / "zoo" / "classifiers.py"
module_spec = importlib.util.spec_from_file_location("dfdc_classifiers", path)
classifiers = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(classifiers)

models = []
for name in [w for w in spec["weights"] if not w.endswith(".safetensors")]:
    checkpoint = weights_dir / name
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != spec["weights"][name]:
        sys.exit(f"{checkpoint} doesn't match its pinned sha256")
    model = classifiers.DeepFakeClassifier(encoder="tf_efficientnet_b7_ns")
    state = torch.load(checkpoint, map_location="cpu")
    state = state.get("state_dict", state)
    model.load_state_dict({re.sub("^module.", "", k): v for k, v in state.items()}, strict=True)
    models.append(model.eval())

normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
units = np.load(run / "inputs.npz")["units"]
scores = []
with torch.no_grad():
    for start in range(0, len(units), 16):
        x = torch.tensor(units[start : start + 16]).float().permute((0, 3, 1, 2))
        for i in range(len(x)):
            x[i] = normalize(x[i] / 255.0)
        preds = [torch.sigmoid(m(x).reshape(-1)) for m in models]
        scores.append(torch.stack(preds).mean(0).numpy())
np.save(run / "reference.npy", np.concatenate(scores).astype(np.float64))
(run / "reference.json").write_text(
    json.dumps(
        {
            "method": f"upstream DeepFakeClassifier x{len(models)}, predict_folder load, fp32",
            "env": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "timm": timm.__version__,
            },
        },
        indent=2,
    )
)
print(f"reference scores for {len(units)} crops ({len(models)} models) in {run}")
