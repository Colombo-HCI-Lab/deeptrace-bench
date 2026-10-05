"""Reference scores for DeepfakeBench detectors, in DeepfakeBench's own PyTorch generation.

Runs in a throwaway Python 3.9 virtualenv with torch 1.13 (docs/models.md says how to make
it), not in this repo's environment, and doesn't import this package: only the
standard-library import helper (``models/_upstream.py``), loaded by path. It builds the
detector from the checkout the same way the adapter must, loads the original release
``.pth`` (not the converted safetensors) with every key matching, and applies
DeepfakeBench's test transform: cubic resize to the yaml's resolution, torchvision
``ToTensor`` and ``Normalize``. So parity covers the weights, the numerics of an old torch
and the transform, but not upstream's package imports, which pull in every detector.

Usage: $DTB_ROOT/parity/envs/dfb/bin/python scripts/parity_reference/deepfakebench.py <run dir>
"""

import importlib
import importlib.util
import json
import platform
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import yaml

run = Path(sys.argv[1])
spec = json.loads((run / "parity.json").read_text())
helper_spec = importlib.util.spec_from_file_location("dtb_parity_helper", spec["helper"])
helper = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(helper)

training = Path(spec["checkout"]) / "training"
name = spec["input"]["detector"]
config = yaml.safe_load((training / "config" / "detector" / f"{name}.yaml").read_text())
config["pretrained"] = None


def stub(module_name, **attrs):
    module = types.ModuleType(module_name)
    module.__dict__.update(attrs)
    return module


class AnyLoss:
    def __getitem__(self, key):
        return lambda *args, **kwargs: None


stubs = {
    "loss": stub("loss", LOSSFUNC=AnyLoss()),
    "torch.utils.tensorboard": stub("torch.utils.tensorboard", SummaryWriter=object),
    # FFD reads templates its released map type never uses (as the adapter does)
    "imageio": stub("imageio", imread=lambda path: np.zeros((19, 19, 3), dtype=np.uint8)),
    "loralib": stub("loralib"),
}
with helper.scoped_modules(stubs, purge=["metrics", "networks", "detectors", "loss"]):
    helper.load_package(training / "metrics", "metrics")
    registry = importlib.import_module("metrics.registry")
    sys.modules["metrics.base_metrics_class"] = stub(
        "metrics.base_metrics_class", calculate_metrics_for_train=None
    )
    helper.load_package(training / "networks", "networks").BACKBONE = registry.BACKBONE
    helper.load_package(training / "detectors", "detectors").DETECTOR = registry.DETECTOR
    importlib.import_module(
        "networks." + ("efficientnetb4" if name == "efficientnetb4" else "xception")
    )
    detector = importlib.import_module(f"detectors.{name}_detector")
    if name == "recce":  # timm's xception(pretrained=True) would download what the .pth replaces
        import functools

        detector.encoder_params["xception"]["init_op"] = functools.partial(
            detector.xception, pretrained=False
        )
    original_load = torch.load
    torch.load = lambda *args, **kwargs: {"conv1.weight": torch.zeros(32, 3, 3, 3)}
    try:
        model = registry.DETECTOR[name](config)
    finally:
        torch.load = original_load

checkpoint = next(n for n in spec["weights"] if n.endswith(".pth"))
state = torch.load(Path(spec["weights_dir"]) / checkpoint, map_location="cpu", weights_only=True)
state = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state.items()}
model.load_state_dict(state, strict=True)
model.eval()

size = int(config["resolution"])
transform = T.Compose([T.ToTensor(), T.Normalize(mean=config["mean"], std=config["std"])])
units = np.load(run / "inputs.npz")["units"]
scores = []
with torch.no_grad():
    for start in range(0, len(units), 32):
        crops = units[start : start + 32]
        x = torch.stack(
            [transform(cv2.resize(c, (size, size), interpolation=cv2.INTER_CUBIC)) for c in crops]
        )
        logits = model({"image": x, "label": torch.zeros(len(x), dtype=torch.long)}, inference=True)
        for attr in ("prob", "label"):
            if isinstance(getattr(model, attr, None), list):
                getattr(model, attr).clear()
        scores.append(torch.softmax(logits["cls"], dim=1)[:, 1].numpy())
np.save(run / "reference.npy", np.concatenate(scores).astype(np.float64))
(run / "reference.json").write_text(
    json.dumps(
        {
            "method": "upstream detector, original release .pth, DeepfakeBench test transform",
            "env": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "opencv": cv2.__version__,
            },
        },
        indent=2,
    )
)
print(f"reference scores for {len(units)} crops in {run}")
