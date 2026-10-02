"""The per-item table every dataset is normalised into.

One row per item (a clip, video or image). Builders in ``deeptrace_bench.datasets`` produce
it from each dataset's own layout; everything downstream (evalsets, scoring, metrics) reads
only manifests, never raw dataset folders.

Columns:

- ``item_id`` (required): ``<dataset>/<local id>``, stable across rebuilds.
- ``dataset``, ``rel_path`` (relative to the dataset dir), ``modality`` and ``label``
  (``real`` or ``fake``) are required.
- ``method`` and ``method_family``: the generator (e.g. ``xtts_v2``) and its kind (``tts``,
  ``vc``, ``face_swap``, ``lip_sync``, ``reenactment``); null for real items.
- ``language``: ISO 639-1 where known.
- ``subject_id``: whose face or voice the item shows. ``source_subject_id``: the identity a
  fake was made from. Both drive identity-safe splits (see ``splits.py``).
- ``split``: the dataset's official split, if any. ``duration_s``.
- Group attributes: one ``g_<attr>`` column per attribute (``g_gender``, ``g_skin_tone``,
  ``g_language``...) and a ``g_<attr>_src`` column saying where each value came from
  (``dataset``, ``self_reported``, ``annotated`` or ``inferred``). Unknown is null, never a
  string like "unknown", so it can't be mistaken for a group.

Manifests are written to ``DTB_ROOT/manifests/`` and never committed: item names count as
derived data under several licences. ``summarize`` produces the committable part.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .paths import manifest_path

REQUIRED = ["item_id", "dataset", "rel_path", "modality", "label"]
OPTIONAL = [
    "method",
    "method_family",
    "language",
    "subject_id",
    "source_subject_id",
    "split",
    "duration_s",
]
LABELS = {"real", "fake"}
MODALITIES = {"video", "audio", "audio_video", "image"}
ATTR_SOURCES = {"dataset", "self_reported", "annotated", "inferred"}
GROUP_PREFIX = "g_"
SRC_SUFFIX = "_src"


class ManifestError(ValueError):
    """Raised when a manifest breaks the schema."""


def group_columns(df: pd.DataFrame) -> list[str]:
    """Return the group attribute columns (``g_*`` without the ``_src`` companions)."""
    return [c for c in df.columns if c.startswith(GROUP_PREFIX) and not c.endswith(SRC_SUFFIX)]


def validate_manifest(df: pd.DataFrame, dataset_id: str) -> None:
    """Check a manifest against the schema.

    Raises:
        ManifestError: listing every problem found.
    """
    problems: list[str] = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ManifestError(f"{dataset_id}: missing columns {missing}")

    unknown = [
        c
        for c in df.columns
        if c not in REQUIRED and c not in OPTIONAL and not c.startswith(GROUP_PREFIX)
    ]
    if unknown:
        problems.append(f"unexpected columns {unknown} (group attributes need the g_ prefix)")
    if df["item_id"].isna().any() or df["item_id"].duplicated().any():
        problems.append("item_id must be present and unique")
    elif not df["item_id"].str.startswith(f"{dataset_id}/").all():
        problems.append(f"every item_id must start with '{dataset_id}/'")
    if (df["dataset"] != dataset_id).any():
        problems.append(f"dataset column must be {dataset_id!r} throughout")
    bad_labels = set(df["label"].dropna().unique()) - LABELS
    if bad_labels or df["label"].isna().any():
        problems.append(f"label must be real or fake, found {sorted(map(str, bad_labels))}")
    bad_modalities = set(df["modality"].dropna().unique()) - MODALITIES
    if bad_modalities:
        problems.append(f"unknown modalities {sorted(bad_modalities)}")

    for col in group_columns(df):
        src = col + SRC_SUFFIX
        if src not in df.columns:
            problems.append(f"{col} needs a {src} column")
            continue
        has_value = df[col].notna()
        bad_src = set(df.loc[has_value, src].dropna().unique()) - ATTR_SOURCES
        if bad_src:
            problems.append(f"{src} has unknown sources {sorted(bad_src)}")
        if df.loc[has_value, src].isna().any():
            problems.append(f"{src} must be set wherever {col} has a value")
        if df[col].astype(str).str.lower().isin({"unknown", "n/a", "na", "none", ""}).any():
            problems.append(f"{col} uses a placeholder string; use null for unknown")
    for col in df.columns:
        if col.endswith(SRC_SUFFIX) and col.removesuffix(SRC_SUFFIX) not in df.columns:
            problems.append(f"{col} has no matching attribute column")

    if problems:
        raise ManifestError(f"{dataset_id}: " + "; ".join(problems))


def write_manifest(df: pd.DataFrame, dataset_id: str) -> Path:
    """Validate and write a manifest to the store. Returns its path."""
    validate_manifest(df, dataset_id)
    path = manifest_path(dataset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("item_id").to_parquet(path, index=False)
    return path


def read_manifest(dataset_id: str) -> pd.DataFrame:
    """Read a dataset's manifest from the store.

    Raises:
        FileNotFoundError: if it hasn't been built (``setup_datasets.py <id> --manifest``).
    """
    path = manifest_path(dataset_id)
    if not path.exists():
        raise FileNotFoundError(
            f"no manifest for {dataset_id} at {path}; build it with "
            f"scripts/setup_datasets.py {dataset_id} --manifest"
        )
    return pd.read_parquet(path)


def content_hash(df: pd.DataFrame) -> str:
    """Hash of (item_id, label) pairs, independent of row order. Changes when items do."""
    pairs = sorted(zip(df["item_id"], df["label"], strict=True))
    digest = hashlib.sha256()
    for item_id, label in pairs:
        digest.update(f"{item_id}\t{label}\n".encode())
    return digest.hexdigest()


def summarize(df: pd.DataFrame) -> dict:
    """Counts with no item names: safe to commit, and shows when a manifest drifts."""
    summary: dict = {
        "rows": int(len(df)),
        "content_hash": content_hash(df),
        "by_label": {str(k): int(v) for k, v in df["label"].value_counts().items()},
        "groups": {},
    }
    for col in group_columns(df):
        counts = df.groupby([df[col].fillna("<null>"), "label"]).size()
        summary["groups"][col] = {
            f"{value}|{label}": int(n) for (value, label), n in counts.items()
        }
    return summary
