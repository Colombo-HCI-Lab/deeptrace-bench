"""Turning an evalset config into the table of items to score."""

from __future__ import annotations

import hashlib

import pandas as pd

from .manifest import read_manifest
from .registry import EvalsetConfig, Subset


class EvalsetError(ValueError):
    """Raised when an evalset can't produce a usable item table."""


def apply_filter(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Keep rows matching every ``column: value`` (or ``column: [values]``) in ``spec``."""
    mask = pd.Series(True, index=df.index)
    for column, wanted in spec.items():
        if column not in df.columns:
            raise EvalsetError(f"filter column {column!r} not in manifest")
        values = wanted if isinstance(wanted, list) else [wanted]
        mask &= df[column].isin(values)
    return df[mask]


_TRACK_COLUMNS = {"from_video": "label_video", "from_audio": "label_audio"}


def _track_label(df: pd.DataFrame, column: str, evalset_id: str) -> pd.Series:
    """The label of one track, which every selected item must have."""
    if column not in df.columns:
        raise EvalsetError(f"{evalset_id}: the manifest has no {column} column")
    unknown = df[column].isna()
    if unknown.any():
        raise EvalsetError(
            f"{evalset_id}: {int(unknown.sum())} items have no {column}; filter them out in "
            f"the component (e.g. {column}: [real, fake])"
        )
    return df[column]


def take_subset(items: pd.DataFrame, subset: Subset, evalset_id: str) -> pd.DataFrame:
    """Up to ``subset.per_stratum`` items of every stratum, in the table's own order."""
    missing = [c for c in subset.strata if c not in items.columns]
    if missing:
        raise EvalsetError(f"{evalset_id}: subset strata {missing} not in the item table")
    rank = items["item_id"].map(
        lambda item: hashlib.sha1(f"{subset.seed}:{item}".encode()).hexdigest()
    )
    ranked = items.assign(_rank=rank).sort_values("_rank", kind="stable")
    kept = ranked.groupby(subset.strata, dropna=False, sort=False).head(subset.per_stratum)
    return kept.sort_index().drop(columns="_rank")


def load_items(
    evalset: EvalsetConfig, manifests: dict[str, pd.DataFrame] | None = None
) -> pd.DataFrame:
    """Build the evalset's item table from its components' manifests.

    Args:
        evalset: the evalset config.
        manifests: preloaded manifests by dataset id (tests pass these); otherwise each is
            read from the store.

    Returns:
        The concatenated rows, with ``label`` overridden where a component forces it or
        takes a track's label, plus
        ``component`` (its index in the config) and ``pairing``; only the evalset's
        ``subset`` of them when it has one.

    Raises:
        EvalsetError: if the evalset ends up without both real and fake items, or with
            duplicate item ids.
    """
    parts = []
    for index, comp in enumerate(evalset.components):
        df = manifests[comp.dataset] if manifests else read_manifest(comp.dataset)
        df = apply_filter(df, comp.filter).copy()
        if comp.label in _TRACK_COLUMNS:
            df["label"] = _track_label(df, _TRACK_COLUMNS[comp.label], evalset.id)
        elif comp.label != "from_manifest":
            df["label"] = comp.label
        df["component"] = index
        parts.append(df)
    items = pd.concat(parts, ignore_index=True)
    items["pairing"] = evalset.pairing

    if items["item_id"].duplicated().any():
        raise EvalsetError(f"{evalset.id}: an item appears in more than one component")
    if evalset.subset is not None:
        items = take_subset(items, evalset.subset, evalset.id).reset_index(drop=True)
    labels = set(items["label"])
    if labels != {"real", "fake"}:
        raise EvalsetError(f"{evalset.id}: needs real and fake items, has only {labels}")
    return items
