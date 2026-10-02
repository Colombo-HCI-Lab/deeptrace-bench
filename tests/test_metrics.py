"""Metrics on inputs whose answers are known."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from deeptrace_bench.eval.metrics import (
    auc,
    bootstrap_ci,
    eer,
    per_group,
    rates_at_threshold,
    score_direction_ok,
    summarize_scores,
)


def test_perfect_separation():
    labels = ["real", "real", "fake", "fake"]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert auc(labels, scores) == 1.0
    rate, thr = eer(labels, scores)
    assert rate == 0.0
    assert 0.2 < thr <= 0.8


def test_inverted_scores_fail_the_direction_check():
    labels = ["real", "real", "fake", "fake"]
    assert not score_direction_ok(labels, [0.9, 0.8, 0.2, 0.1])
    assert score_direction_ok(labels, [0.1, 0.2, 0.8, 0.9])


def test_eer_on_overlapping_scores():
    # One real scores above one fake: at the best threshold 1 of 4 reals and 1 of 4 fakes err.
    labels = ["real"] * 4 + ["fake"] * 4
    scores = [0.1, 0.2, 0.3, 0.7, 0.6, 0.8, 0.9, 0.95]
    rate, _ = eer(labels, scores)
    assert math.isclose(rate, 0.25, abs_tol=1e-9)


def test_single_class_gives_nan():
    assert math.isnan(auc(["fake", "fake"], [0.1, 0.9]))
    assert all(math.isnan(x) for x in eer(["real"], [0.3]))


def test_rates_at_threshold():
    labels = ["real", "real", "fake", "fake"]
    rates = rates_at_threshold(labels, [0.1, 0.6, 0.4, 0.9], threshold=0.5)
    assert rates == {"fpr": 0.5, "fnr": 0.5}


def test_bootstrap_interval_brackets_the_estimate():
    rng = np.random.default_rng(1)
    labels = np.array([0] * 200 + [1] * 200)
    scores = np.concatenate([rng.normal(0, 1, 200), rng.normal(1, 1, 200)])
    lo, hi = bootstrap_ci(labels, scores, auc, n=200, seed=0)
    point = auc(labels, scores)
    assert lo < point < hi
    assert lo > 0.6 and hi < 0.9


def test_per_group_counts_failures_and_flags_small_groups():
    df = pd.DataFrame(
        {
            "label": ["real", "fake"] * 40 + ["real", "fake"],
            "score": [0.2, 0.8] * 40 + [np.nan, np.nan],
            "status": ["ok"] * 80 + ["no_face", "no_face"],
            "g_gender": ["f"] * 40 + ["m"] * 40 + [None, None],
        }
    )
    overall = summarize_scores(df, n_boot=50)
    table = per_group(df, "g_gender", threshold=overall["threshold"], min_n=10, n_boot=50)
    by_group = table.set_index("group")
    assert by_group.loc["f", "auc"] == 1.0
    assert not by_group.loc["f", "small"]
    assert by_group.loc["<null>", "n_failed"] == 2
    assert by_group.loc["<null>", "small"]
    assert overall["n_failed"] == 2
