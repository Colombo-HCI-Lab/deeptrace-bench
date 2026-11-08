"""Parity checks: does an adapter score exactly what the upstream code scores?

An adapter counts only after its scores on about 200 items match upstream's within about 1e-3
(``docs/src/content/docs/models.md``). The check runs in three steps, so the upstream side can run
in an environment of its own (old PyTorch, no access to this package):

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


def pack_inputs(
    per_item: Sequence[Sequence[np.ndarray]],
) -> tuple[np.ndarray | list[np.ndarray], np.ndarray]:
    """Each item's inputs (windows or crops) as one sequence of units, plus each unit's item index.

    Inputs of one shape stack into one array. Native-size face crops (GenD's) differ in shape, so
    they stay a list, which ``save_inputs`` writes one array per unit.
    """
    index = np.concatenate([np.full(len(units), i) for i, units in enumerate(per_item)])
    shapes = {np.asarray(unit).shape for units in per_item for unit in units}
    if len(shapes) == 1:
        return np.concatenate([np.asarray(units) for units in per_item]), index
    return [np.asarray(unit) for units in per_item for unit in units], index


def save_inputs(path, units: np.ndarray | list[np.ndarray], item_index: np.ndarray) -> None:
    """``inputs.npz``: ``units`` when they share a shape, else ``unit_<j>`` and ``n_units``.

    Reference scripts read it with ``load_units``'s logic, which they copy, since they run outside
    this package.
    """
    if isinstance(units, np.ndarray):
        np.savez(path, units=units, item_index=item_index)
        return
    arrays = {f"unit_{j}": u for j, u in enumerate(units)}
    np.savez(path, n_units=np.int64(len(units)), item_index=item_index, **arrays)


def load_units(path) -> np.ndarray | list[np.ndarray]:
    """The units ``save_inputs`` wrote, an array or a list of arrays of different shapes."""
    z = np.load(path)
    if "units" in z:
        return z["units"]
    return [z[f"unit_{j}"] for j in range(int(z["n_units"]))]


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
