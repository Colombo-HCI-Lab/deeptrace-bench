"""Parity checks: picking items, packing inputs, and comparing adapter with upstream scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deeptrace_bench.parity import ParityError, compare, pack_inputs, pick_items


def _items(n_real: int, n_fake: int) -> pd.DataFrame:
    rows = [{"item_id": f"d/r{i}", "label": "real"} for i in range(n_real)]
    rows += [{"item_id": f"d/f{i}", "label": "fake"} for i in range(n_fake)]
    return pd.DataFrame(rows)


def test_items_are_picked_evenly_and_reproducibly():
    items = _items(300, 50)
    picked = pick_items(items, 200, seed=0)
    assert picked["label"].value_counts().to_dict() == {"real": 150, "fake": 50}
    shuffled = pick_items(items.sample(frac=1, random_state=1), 200, seed=0)
    assert picked["item_id"].tolist() == shuffled["item_id"].tolist()


def test_inputs_pack_into_one_array_with_an_item_index():
    units, index = pack_inputs([np.zeros((2, 8)), np.ones((1, 8))])
    assert units.shape == (3, 8) and index.tolist() == [0, 0, 1]


def test_inputs_of_different_shapes_are_refused():
    with pytest.raises(ParityError, match="shape"):
        pack_inputs([np.zeros((1, 8)), np.zeros((1, 9))])


def test_scores_within_tolerance_pass():
    adapter = np.array([0.1, 0.5, 0.9])
    result = compare(adapter, adapter + 5e-4)
    assert result["passed"] and result["n"] == 3
    assert result["max_abs_diff"] == pytest.approx(5e-4)


def test_one_score_out_of_tolerance_fails():
    adapter = np.array([0.1, 0.5, 0.9])
    assert not compare(adapter, adapter + np.array([0, 0, 2e-3]))["passed"]


def test_mismatched_lengths_are_an_error():
    with pytest.raises(ParityError, match="length"):
        compare(np.zeros(3), np.zeros(4))
