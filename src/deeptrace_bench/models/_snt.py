"""What LAA-Net and FakeSTormer share: both are University of Luxembourg (SnT) repos from the
same authors, licensed for research use only, so neither is copied here. Each adapter imports
its network from the pinned checkout, and this module holds the two pieces both need.

**Importing the checkout.** Both repos import by top-level names (``models``, ``register``,
``losses``) from the checkout's root and, in ``builder.py``, append the working directory to
``sys.path``. ``checkout_imports`` puts the root first on ``sys.path`` for one ``with`` block,
removes those names afterwards and puts ``sys.path`` back, so nothing of the checkout outlives
the import.

**The test-time warp.** Both test scripts cut a saved face crop with ``crop_by_margin``, move it
to the network's input size with ``get_center_scale`` and ``get_affine_transform`` through
``cv2.warpAffine`` (bilinear), divide by 255 in float64 and normalise. Those three helpers come
from the checkout's ``package_utils/`` (``transform.py`` imports ``albumentations`` at the top,
which only its training augmentations use, so a stand-in module is installed while it is
imported). The normalisation is upstream's ``ToTensor`` and ``Normalize`` written as the numpy
operations they are, in the same order and precision.
"""

from __future__ import annotations

import contextlib
import sys
import types
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ._upstream import load_module, scoped_modules


def stub_module(name: str, **attrs: Any) -> types.ModuleType:
    """An empty module called ``name`` with ``attrs`` set."""
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def unused(*args: Any, **kwargs: Any) -> Any:
    """Stands in for a training-only upstream helper; calling it is a bug here."""
    raise RuntimeError("a training-only upstream helper was called")


@contextlib.contextmanager
def checkout_imports(
    root: Path, purge: Iterable[str], stubs: dict[str, types.ModuleType] | None = None
) -> Iterator[None]:
    """Make a checkout importable by its own top-level names for one ``with`` block.

    ``stubs`` are installed in ``sys.modules`` meanwhile; on exit they are restored, every
    module whose top-level package is in ``purge`` is removed, and ``sys.path`` is put back.
    Objects built inside the block keep working.
    """
    saved = list(sys.path)
    sys.path.insert(0, str(root))
    try:
        with scoped_modules(stubs or {}, purge=purge):
            yield
    finally:
        sys.path[:] = saved


def read_config(root: Path, relative: str) -> dict[str, Any]:
    """A YAML config of the checkout (plain data, so ``safe_load`` reads it)."""
    import yaml

    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run scripts/setup_models.py")
    return yaml.safe_load(path.read_text())


class InputWarp:
    """Upstream's per-image test preprocessing: margin crop, affine warp, normalise.

    Args:
        root: the checkout.
        tag: names the loaded upstream modules (they are registered once per process).
        size: the network's square input size (``DATASET.IMAGE_SIZE``).
        margin: pixels cut from each side first (``crop_by_margin``).
        mean, std: ``DATASET.TRANSFORM.normalize``.
    """

    def __init__(
        self, root: Path, tag: str, size: int, margin: int, mean: list[float], std: list[float]
    ) -> None:
        self.size = size
        self.margin = margin
        self.mean = np.asarray(mean, dtype=np.float64).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float64).reshape(1, 1, 3)
        utils = Path(root) / "package_utils"
        # transform.py imports albumentations for its augmentations only
        stubs = {"albumentations": stub_module("albumentations")}
        with scoped_modules(stubs):
            transform = load_module(utils / "transform.py", f"dtb_upstream_{tag}_transform")
        self._center_scale = transform.get_center_scale
        self._affine = transform.get_affine_transform
        self._crop_by_margin = load_module(
            utils / "image_utils.py", f"dtb_upstream_{tag}_image_utils"
        ).crop_by_margin

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """``[h, w, 3]`` uint8 RGB to a normalised float64 ``[3, size, size]`` array."""
        import cv2

        cropped = self._crop_by_margin(image, margin=[self.margin, self.margin])
        center, scale = self._center_scale(cropped.shape[:2], 1.0, 200)
        trans = self._affine(center, scale, 0, (self.size, self.size))
        warped = cv2.warpAffine(cropped, trans, (self.size, self.size), flags=cv2.INTER_LINEAR)
        # ToTensor on a float64 array keeps float64; Normalize subtracts then divides
        x = (warped / 255 - self.mean) / self.std
        return np.ascontiguousarray(x.transpose(2, 0, 1))
