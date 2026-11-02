"""Reference scores for FakeSTormer, in upstream's own stack, run on the adapter's exact crops.

Runs in a virtualenv with upstream's libraries, not in this repo's environment, and doesn't
import this package. It shares the virtualenv of ``laa_net.py``, whose docstring says how to make
it: Python 3.8, torch 1.13.1 and torchvision 0.14.1 on CPU (upstream's ``dockerfiles/Dockerfile``
is PyTorch 1.8.0, which has no wheel for Apple silicon), numpy 1.23.3, opencv-python-headless
4.6.0.66, albumentations 1.1.0, python-box 7.1.1, timm 0.6.13, mmcv 1.6.1 without ops (the
README's pin, built from its sdist), mmengine 0.10.7 and ptflops. One package upstream imports
and its Dockerfile and README don't list: ``plotly`` (``package_utils/utils.py``, reached through
``logs/logger.py`` from the Swin backbone), installed as 7.1.0 with torch, torchvision, numpy,
timm and opencv pinned beside it so uv doesn't upgrade them.

It follows ``scripts/test.py`` on ``configs/temporal/FakeSFormer_base_c23.yaml``: ``load_config``,
``from models import *`` (the whole package, as upstream does, nothing stubbed),
``build_model(cfg.MODEL, MODELS)`` in float32 (``.to(torch.float)``), ``load_pretrained``'s strict
load of the original ``.pth``, ``final_transform`` (``ToTensor`` and ``Normalize`` with ImageNet's
mean and std), and for each of a run's four frames ``crop_by_margin`` 15, ``get_center_scale``,
``get_affine_transform`` and ``cv2.warpAffine`` to 224 x 224, the dataset's float64 stack
transposed to ``[3, 4, 224, 224]``, then ``.to(torch.float)`` and the model. The score is the
sigmoid of ``outputs[0]["cls"]`` (``bin_calculate_auc_ap_ar`` applies it after upstream's
per-video mean, which is one score per run here). ``TEST.flip_test`` is off and
``no_shot_preds`` is 1, so each run is one forward pass.

Differences from running ``test.py``: the data loader (it reads saved PNG crops; this reads the
exported arrays) and the weights' route: ``load_pretrained``'s plain ``torch.load`` needs a GPU
for a checkpoint saved from one and unpickles a download, so the tensors come from the adapter's
safetensors conversion of the checkpoint instead (``load_converted``), after checking that it was
made from the pinned download. The strict ``load_state_dict`` is upstream's.

The run directory may be written on one machine and read on another;
``DTB_PATH_MAP="<from>=<to>,..."`` rewrites the path prefixes recorded in ``parity.json``.

Usage: <snt venv>/bin/python scripts/parity_reference/fakestormer.py <run dir>
"""

import json
import logging
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

BATCH = 4


def mapped(path):
    for pair in filter(None, os.environ.get("DTB_PATH_MAP", "").split(",")):
        old, new = pair.split("=", 1)
        if path.startswith(old):
            return new + path[len(old) :]
    return path


def load_converted(checkpoint, spec):
    """The checkpoint's tensors, from the adapter's safetensors conversion of it.

    torch 1.13's ``weights_only`` unpickler can't read the checkpoint (its optimizer state holds
    floats), and the file is a download that nothing vouches for, so it is not unpickled here.
    The conversion (``Detector.save_converted``) is a tensor-for-tensor copy made with
    ``weights_only=True`` and records the sha256 of the download it came from, which must be
    the pinned one.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = checkpoint.parent / "model.safetensors"
    with safe_open(str(path), "pt") as f:
        meta = f.metadata() or {}
    if (meta.get("source"), meta.get("source_sha256")) != (
        checkpoint.name,
        spec["weights"][checkpoint.name],
    ):
        sys.exit(f"{path} isn't a conversion of the pinned {checkpoint.name}")
    return load_file(str(path))


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
run = Path(sys.argv[1]).resolve()
spec = json.loads((run / "parity.json").read_text())
root = Path(mapped(spec["checkout"]))
os.chdir(root)  # test.py runs from the checkout root and adds it to sys.path
sys.path.insert(0, str(root))
from configs.get_config import load_config  # noqa: E402  (the checkout's own modules)
from models import MODELS, build_model  # noqa: E402
from package_utils.image_utils import crop_by_margin  # noqa: E402
from package_utils.transform import (  # noqa: E402
    final_transform,
    get_affine_transform,
    get_center_scale,
)

cfg = load_config("configs/temporal/FakeSFormer_base_c23.yaml")
checkpoint = Path(mapped(spec["weights_dir"])) / "fakestormer.pth"
state = load_converted(checkpoint, spec)

model = build_model(cfg.MODEL, MODELS).to(torch.float)
model.load_state_dict(state, strict=True)
model.eval()

size = cfg.DATASET.IMAGE_SIZE
aspect_ratio = size[1] * 1.0 / size[0]
transforms = final_transform(cfg.DATASET)
assert not cfg.TEST.flip_test and (cfg.TEST.no_shot_preds or 1) == 1


def prepare(run_frames):
    """One run of frames as the dataset returns it: ``[3, frames, h, w]`` float64."""
    inputs = []
    for image in run_frames:
        img = crop_by_margin(image, margin=[15, 15])
        c, s = get_center_scale(img.shape[:2], aspect_ratio, 200)
        trans = get_affine_transform(c, s, 0, size)
        warped = cv2.warpAffine(img, trans, (int(size[0]), int(size[1])), flags=cv2.INTER_LINEAR)
        inputs.append(transforms(warped / 255))
    stacked = torch.tensor(np.array([[j.numpy() for j in i] for i in inputs]))
    return stacked.transpose(0, 1)


units = np.load(run / "inputs.npz")["units"]
scores = []
start = time.time()
with torch.no_grad():
    for i in range(0, len(units), BATCH):
        inputs = torch.stack([prepare(u) for u in units[i : i + BATCH]]).to(dtype=torch.float)
        outputs = model(inputs)
        scores.extend(outputs[0]["cls"].sigmoid().reshape(-1).tolist())
        done = i + len(inputs)
        rate = done / (time.time() - start)
        logging.info(
            "[progress] reference %d/%d runs %.2f it/s eta %.0fs",
            done,
            len(units),
            rate,
            (len(units) - done) / rate,
        )
np.save(run / "reference.npy", np.asarray(scores, dtype=np.float64))
(run / "reference.json").write_text(
    json.dumps(
        {
            "method": (
                "upstream test.py path: build_model, strict load of the original .pth, "
                "crop_by_margin 15 + affine warp + Normalize per frame, runs of 4, sigmoid of "
                "the cls logit"
            ),
            "env": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "mmcv": __import__("mmcv").__version__,
            },
        },
        indent=2,
    )
)
print(f"reference scores for {len(units)} runs in {run}")
