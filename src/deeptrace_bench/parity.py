"""Parity checks: does an adapter score exactly what the upstream code scores?

An adapter counts only after its scores on about 200 items match upstream's within about
1e-3 (``docs/models.md``). The check runs in three steps, so the upstream side can run in an
environment of its own (old PyTorch, no access to this package):

1. ``scripts/parity.py export`` picks items, cuts them into model inputs through the same
   loaders scoring uses, saves the inputs, and scores them with the adapter on CPU.
2. A reference script in ``scripts/parity_reference/`` loads the same inputs, runs the
   upstream code with the original checkpoint, and saves its scores.
3. ``scripts/parity.py compare`` compares the two and writes a committable summary to
   ``results/parity/<model>.json`` (counts and differences only, no item ids).

The run folder lives in the store (``parity/<model>/<stamp>/``) because its inputs and item
list are derived from datasets.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .paths import store_root
from .sample import _rank

TOLERANCE = 1e-3
INPUTS = "inputs.npz"
ADAPTER_SCORES = "adapter.npy"
REFERENCE_SCORES = "reference.npy"
REFERENCE_INFO = "reference.json"
RUN_INFO = "parity.json"


class ParityError(ValueError):
    """Inputs or scores that can't be compared."""


def parity_root(model_id: str):
    """Where a model's parity runs are kept in the store."""
    return store_root() / "parity" / model_id


def pick_items(items: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Up to ``n`` items, half of each label where the evalset has enough, chosen by seeded rank.

    The same items come back whatever order ``items`` arrives in.
    """
    ranked = items.assign(_rank=[_rank(seed, i) for i in items["item_id"]]).sort_values("_rank")
    half = n // 2
    real = ranked[ranked["label"] == "real"]
    fake = ranked[ranked["label"] == "fake"]
    take_real = min(len(real), max(half, n - len(fake)))
    take_fake = min(len(fake), n - take_real)
    picked = pd.concat([real.head(take_real), fake.head(take_fake)])
    return picked.sort_values("_rank").drop(columns="_rank").reset_index(drop=True)


def pack_inputs(per_item: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Stack each item's inputs (windows or crops) into one array, plus each unit's item index.

    Raises:
        ParityError: if inputs differ in shape (native-size crops aren't supported yet).
    """
    shapes = {np.asarray(unit).shape for units in per_item for unit in units}
    if len(shapes) != 1:
        raise ParityError(f"inputs differ in shape ({sorted(shapes)}); export needs one shape")
    units = np.concatenate([np.asarray(units) for units in per_item])
    index = np.concatenate([np.full(len(units), i) for i, units in enumerate(per_item)])
    return units, index


def compare(adapter: np.ndarray, reference: np.ndarray, tol: float = TOLERANCE) -> dict:
    """Per-unit differences between adapter and upstream scores.

    Raises:
        ParityError: if the two score arrays differ in length.
    """
    adapter = np.asarray(adapter, dtype=np.float64).ravel()
    reference = np.asarray(reference, dtype=np.float64).ravel()
    if len(adapter) != len(reference):
        raise ParityError(f"length {len(adapter)} (adapter) != {len(reference)} (reference)")
    diff = np.abs(adapter - reference)
    return {
        "n": int(len(diff)),
        "max_abs_diff": float(diff.max()) if len(diff) else 0.0,
        "mean_abs_diff": float(diff.mean()) if len(diff) else 0.0,
        "tolerance": tol,
        "passed": bool(len(diff)) and bool(diff.max() <= tol),
    }
