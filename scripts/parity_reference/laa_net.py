"""Reference scores for LAA-Net, in upstream's own stack, run on the adapter's exact crops.

Runs in a virtualenv with upstream's libraries, not in this repo's environment, and doesn't
import this package. Upstream's ``dockerfiles/Dockerfile`` is PyTorch 1.8.0 (CUDA 11.1), Python
3.8, numpy 1.23.3, opencv-python 4.5.1.48, Pillow 9.3.0, scikit-image 0.19.3, scipy 1.9.3,
albumentations 1.1.0 and python-box 7.1.1. PyTorch 1.8 has no wheel for Apple silicon (the first
arm64 build is 1.12), so the closest stack that installs on a Mac is the same Python and numpy
with torch 1.13.1 and torchvision 0.14.1 on CPU and opencv-python-headless 4.6.0.66. FakeSTormer's
reference runs in the same virtualenv:

    uv venv --python 3.8 $DTB_ROOT/parity/envs/snt
    uv pip install --python $DTB_ROOT/parity/envs/snt/bin/python torch==1.13.1 \\
      torchvision==0.14.1 numpy==1.23.3 opencv-python-headless==4.6.0.66 Pillow==9.3.0 \\
      scikit-image==0.19.3 scipy==1.9.3 albumentations==1.1.0 python-box==7.1.1 pyyaml \\
      simplejson natsort==8.4.0 "scikit-learn<1.3" einops timm==0.6.13 mmengine ptflops \\
      addict yapf packaging safetensors plotly
    # mmcv 1.6.1 (FakeSTormer's pin) has no wheel here; its sdist builds without ops
    MMCV_WITH_OPS=0 uv pip install --python $DTB_ROOT/parity/envs/snt/bin/python \\
      --no-deps --no-build-isolation mmcv==1.6.1

(uv upgrades torch whenever timm or mmengine is installed without the pins beside it; check
``torch.__version__`` afterwards.)

It follows ``scripts/test.py`` for the checkpoint's own config (``configs/efn4_fpn_sbi_adv.yaml``):
``load_config``, ``build_model(cfg.MODEL, MODELS)`` converted to float64, ``load_pretrained``'s
strict load of the original ``.pth``, ``final_transform`` (``ToTensor`` and ``Normalize`` with
mean and std 0.5), and for each crop ``crop_by_margin`` 5, ``get_center_scale``,
``get_affine_transform`` and ``cv2.warpAffine`` to 384 x 384, then the model on the batch and on
``inputs.flip(dims=(3,))``, with the two ``cls`` outputs averaged (``TEST.flip_test: True``).
Each crop's score is the sigmoid of that (``bin_calculate_auc_ap_ar`` applies it after
upstream's per-video mean, which is one score per crop here).

Differences from running ``test.py``: the data loader (it reads saved PNG crops; this reads the
exported arrays) and the weights' route: ``load_pretrained``'s plain ``torch.load`` needs a GPU
for a checkpoint saved from one and unpickles a download, so the tensors come from the adapter's
safetensors conversion of the checkpoint instead (``load_converted``), after checking that it was
made from the pinned download. The strict ``load_state_dict`` is upstream's.

The run directory may be written on one machine and read on another;
``DTB_PATH_MAP="<from>=<to>,..."`` rewrites the path prefixes recorded in ``parity.json``.

Usage: <snt venv>/bin/python scripts/parity_reference/laa_net.py <run dir>
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

BATCH = 8


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

cfg = load_config("configs/efn4_fpn_sbi_adv.yaml")
checkpoint = Path(mapped(spec["weights_dir"])) / "laa_net_sbi.pth"
state = load_converted(checkpoint, spec)

model = build_model(cfg.MODEL, MODELS).to(torch.float64)
model.load_state_dict(state, strict=True)
model.eval()

size = cfg.DATASET.IMAGE_SIZE
aspect_ratio = size[1] * 1.0 / size[0]
transforms = final_transform(cfg.DATASET)
flip_test = cfg.TEST.flip_test


def prepare(image):
    img = crop_by_margin(image, margin=[5, 5])
    c, s = get_center_scale(img.shape[:2], aspect_ratio, 200)
    trans = get_affine_transform(c, s, 0, size)
    warped = cv2.warpAffine(img, trans, (int(size[0]), int(size[1])), flags=cv2.INTER_LINEAR)
    return transforms(warped / 255).to(torch.float64)


units = np.load(run / "inputs.npz")["units"]
scores = []
start = time.time()
with torch.no_grad():
    for i in range(0, len(units), BATCH):
        inputs = torch.stack([prepare(u) for u in units[i : i + BATCH]])
        outputs = model(inputs)[0]
        cls = outputs["cls"]
        if flip_test:
            cls = (cls + model(inputs.flip(dims=(3,)))[0]["cls"]) / 2
        scores.extend(cls.sigmoid().reshape(-1).tolist())
        done = i + len(inputs)
        rate = done / (time.time() - start)
        logging.info(
            "[progress] reference %d/%d crops %.2f it/s eta %.0fs",
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
                "upstream test.py path: build_model, strict load of the original .pth in "
                "float64, crop_by_margin 5 + affine warp + Normalize, flip test, sigmoid of "
                "the cls logit"
            ),
            "env": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "opencv": cv2.__version__,
            },
        },
        indent=2,
    )
)
print(f"reference scores for {len(units)} crops in {run}")
