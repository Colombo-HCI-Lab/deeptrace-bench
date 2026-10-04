"""Celeb-DF (v2), the standard cross-dataset benchmark for face-swap detectors.

Layout (from the authors' README; not yet checked against a copy, 2026-10-05):
``Celeb-real/`` (590 celebrity interview videos), ``YouTube-real/`` (300 more real videos),
``Celeb-synthesis/`` (5,639 face swaps) and ``List_of_testing_videos.txt``, one
``<label> <path>`` line per test video (label 1 real, 0 fake): 518 videos. Videos outside the
list get ``split`` train.

No demographics: the paper reports 88% Caucasian subjects. It is here to reproduce published
numbers (SBI, UCF, Effort and GenD all report Celeb-DF v2 AUC), not for South Asian results.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pandas as pd

from . import iter_files

DATASET = "celeb_df_v2"
TEST_LIST = "List_of_testing_videos.txt"
_FOLDERS = {"Celeb-real": "real", "YouTube-real": "real", "Celeb-synthesis": "fake"}


def label_from_path(rel_path: str) -> str | None:
    """The label from the top folder, for ``.mp4`` files directly inside it."""
    path = PurePosixPath(rel_path)
    if len(path.parts) != 2 or path.suffix.lower() != ".mp4":
        return None
    return _FOLDERS.get(path.parts[0])


def build_manifest(root: Path) -> pd.DataFrame:
    """One row per video under ``root``; the official test list sets ``split``."""
    listed = root / TEST_LIST
    test = set()
    if listed.exists():
        test = {line.split()[1] for line in listed.read_text().splitlines() if line.strip()}
    rows = []
    for rel in iter_files(root):
        label = label_from_path(rel)
        if label is None:
            continue
        path = PurePosixPath(rel)
        rows.append(
            {
                "item_id": f"{DATASET}/{path.parts[0]}/{path.stem}",
                "dataset": DATASET,
                "rel_path": rel,
                "modality": "video",
                "label": label,
                "method_family": "face_swap" if label == "fake" else None,
                "split": "test" if rel in test else "train",
            }
        )
    return pd.DataFrame(rows)
