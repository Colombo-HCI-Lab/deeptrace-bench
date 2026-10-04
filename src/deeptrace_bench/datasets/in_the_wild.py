"""In-the-Wild (Müller et al., Interspeech 2022), audio deepfakes of public figures.

Layout (checked in the archive at revision ``eee168f``, 2026-10-05): one zip,
``release_in_the_wild/<n>.wav`` (31,779 clips: 19,963 bona fide, 11,816 spoof, 54 speakers),
plus ``meta.csv`` (``file, speaker, label`` with labels ``bona-fide`` / ``spoof``) and
``attribution.txt``. Labels live only in ``meta.csv``, so the builder reads labels from it
(``METADATA_FILES``) rather than from paths; the sampler reads the same file first.

English, no generator names, no official split. The speakers are public figures named in
``meta.csv``, so ``subject_id`` holds names: manifests never leave the store, and the evalset
groups by nothing finer than the whole set.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pandas as pd

DATASET = "in_the_wild"
META = "release_in_the_wild/meta.csv"
METADATA_FILES = [META]
_LABELS = {"bona-fide": "real", "spoof": "fake"}


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def labels_from_metadata(files: dict[str, bytes]) -> dict[str, str]:
    """``{rel_path: label}`` for every clip listed in ``meta.csv``."""
    return {
        f"release_in_the_wild/{row['file']}": _LABELS[row["label"]]
        for row in _rows(files[META].decode())
    }


def build_manifest(root: Path) -> pd.DataFrame:
    """One row per clip listed in ``meta.csv`` and present under ``root``."""
    rows = []
    for row in _rows((root / META).read_text()):
        rel = f"release_in_the_wild/{row['file']}"
        if not (root / rel).exists():
            continue
        rows.append(
            {
                "item_id": f"{DATASET}/{Path(row['file']).stem}",
                "dataset": DATASET,
                "rel_path": rel,
                "modality": "audio",
                "label": _LABELS[row["label"]],
                "language": "en",
                "subject_id": row["speaker"],
            }
        )
    return pd.DataFrame(rows)
