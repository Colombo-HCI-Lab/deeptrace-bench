"""Detection metrics, overall and per group.

Conventions: ``fake`` is the positive class and scores are P(fake), so a higher score means
more likely fake. Items whose status isn't ``ok`` (no face, too short, unreadable) are left
out of AUC and EER but counted per group, because if they cluster in one group that is a
finding in itself.

Per-group numbers come with stratified bootstrap confidence intervals, and groups below
``min_n`` items of either class are flagged rather than hidden. FPR and FNR are reported at
one global threshold (the overall EER threshold) because a model can rank every group well
(similar AUC) yet be miscalibrated for one of them, which an AUC gap would miss.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

Metric = Callable[[np.ndarray, np.ndarray], float]


def to_binary(labels: Sequence) -> np.ndarray:
    """Map ``real``/``fake`` (or 0/1) labels to 0/1 with fake as 1."""
    arr = np.asarray(labels)
    if arr.dtype.kind in "OUS":
        return (arr == "fake").astype(int)
    return arr.astype(int)


def auc(labels: Sequence, scores: Sequence) -> float:
    """ROC AUC with fake as the positive class; NaN if only one class is present."""
    y = to_binary(labels)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores, dtype=float)))


def eer(labels: Sequence, scores: Sequence) -> tuple[float, float]:
    """Equal error rate and the threshold where it occurs.

    The EER is interpolated linearly between the two ROC points where the false negative
    rate crosses the false positive rate.

    Returns:
        ``(eer, threshold)``; ``(nan, nan)`` if only one class is present.
    """
    y = to_binary(labels)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y, np.asarray(scores, dtype=float))
    fnr = 1.0 - tpr
    diff = fnr - fpr
    j = int(np.argmax(diff <= 0))
    if j == 0:
        return float((fpr[0] + fnr[0]) / 2), float(thr[0])
    t = diff[j - 1] / (diff[j - 1] - diff[j])
    rate = fpr[j - 1] + t * (fpr[j] - fpr[j - 1])
    return float(rate), float(thr[j])


def rates_at_threshold(labels: Sequence, scores: Sequence, threshold: float) -> dict[str, float]:
    """False positive and false negative rates when ``score >= threshold`` means fake."""
    y = to_binary(labels)
    pred_fake = np.asarray(scores, dtype=float) >= threshold
    real, fake = y == 0, y == 1
    fpr = float(pred_fake[real].mean()) if real.any() else float("nan")
    fnr = float((~pred_fake[fake]).mean()) if fake.any() else float("nan")
    return {"fpr": fpr, "fnr": fnr}


def bootstrap_ci(
    labels: Sequence,
    scores: Sequence,
    metric: Metric,
    n: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for ``metric``, resampling real and fake separately.

    Stratifying keeps every resample's class balance equal to the original, so a small
    group can't produce resamples with no fakes in them.
    """
    y = to_binary(labels)
    s = np.asarray(scores, dtype=float)
    real_idx, fake_idx = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    if len(real_idx) == 0 or len(fake_idx) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = np.empty(n)
    for b in range(n):
        idx = np.concatenate(
            [rng.choice(real_idx, len(real_idx)), rng.choice(fake_idx, len(fake_idx))]
        )
        values[b] = metric(y[idx], s[idx])
    alpha = (1 - ci) / 2
    lo, hi = np.nanquantile(values, [alpha, 1 - alpha])
    return float(lo), float(hi)


def _eer_only(y: np.ndarray, s: np.ndarray) -> float:
    return eer(y, s)[0]


def summarize_scores(
    df: pd.DataFrame,
    threshold: float | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """AUC and EER with intervals, rates at ``threshold``, and counts, for one set of rows.

    Args:
        df: rows with ``label``, ``score`` and ``status``.
        threshold: the global threshold for FPR/FNR; defaults to this set's own EER point.
    """
    ok = df[df["status"] == "ok"]
    y, s = to_binary(ok["label"]), ok["score"].to_numpy(dtype=float)
    e, e_thr = eer(y, s)
    thr = e_thr if threshold is None else threshold
    out = {
        "n": int(len(df)),
        "n_real": int((df["label"] == "real").sum()),
        "n_fake": int((df["label"] == "fake").sum()),
        "n_failed": int((df["status"] != "ok").sum()),
        "auc": auc(y, s),
        "eer": e,
        "threshold": thr,
    }
    out["auc_lo"], out["auc_hi"] = bootstrap_ci(y, s, auc, n=n_boot, seed=seed)
    out["eer_lo"], out["eer_hi"] = bootstrap_ci(y, s, _eer_only, n=n_boot, seed=seed)
    out.update(rates_at_threshold(y, s, thr) if len(ok) else {"fpr": np.nan, "fnr": np.nan})
    return out


def per_group(
    df: pd.DataFrame,
    group_col: str,
    threshold: float,
    min_n: int = 30,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """One row of ``summarize_scores`` per value of ``group_col``.

    Missing values form their own ``<null>`` group, since who lacks a label matters too.
    ``small`` marks groups with fewer than ``min_n`` real or fake items: report them, but
    don't draw conclusions from their gaps.
    """
    rows = []
    keys = df[group_col].astype(object).where(df[group_col].notna(), "<null>")
    for value, part in df.groupby(keys, sort=True):
        row = {"group": value, **summarize_scores(part, threshold, n_boot, seed)}
        row["small"] = row["n_real"] < min_n or row["n_fake"] < min_n
        rows.append(row)
    return pd.DataFrame(rows)


def score_direction_ok(labels: Sequence, scores: Sequence) -> bool:
    """True when fakes score higher than reals on the whole (AUC above 0.5).

    Run on a reproduction set before trusting an adapter: an inverted logit gives AUC near
    1 - expected and would quietly flip every downstream result.
    """
    value = auc(labels, scores)
    return bool(value > 0.5)


def group_gap(per_group_values: Mapping[str, float]) -> float:
    """The headline fairness number: how far apart groups are on one metric.

    Args:
        per_group_values: metric value (e.g. AUC) per group, already restricted to groups
            that aren't flagged ``small``.

    Returns:
        A single non-negative gap. NaN if fewer than two groups are given.
    """
    # TODO(Oshan): choose the definition. Options:
    #   - max minus min: simple, matches decision 4's "shift of 5 AUC points across
    #     values", but one noisy group decides it.
    #   - worst group vs the rest (or vs overall): stable, and names who is underserved.
    #   - largest pair whose bootstrap intervals don't overlap: conservative, but needs the
    #     intervals passed in as well.
    raise NotImplementedError("group_gap definition pending; see the TODO above")
