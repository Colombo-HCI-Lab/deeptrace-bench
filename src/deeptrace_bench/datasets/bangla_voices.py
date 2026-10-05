"""Bangla Audio Dataset: Original and DeepFake Voices (University of Asia Pacific, Mendeley
Data 10.17632/4ftmwt86vr.1, 2024, CC BY 4.0).

Layout (checked by range reads of Mendeley's whole-dataset zip, 2026-10-05; 4,500 WAV files):
``<title>/Dataset/S<set><F|M><nn>/{Real,Fake}/<n>.wav``. 75 speakers in 15 sets of five, each
reading the set's 30 sentences, once real and once faked, so every fake has a real twin by the
same speaker. The folder name gives the speaker (``subject_id`` and ``source_subject_id``)
and the gender (``g_gender``, source ``dataset``). The record doesn't name the generator.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

import pandas as pd

from . import iter_files

DATASET = "bangla_voices"
_SPEAKER = re.compile(r"^S(\d+)([FM])(\d+)$")
_LABELS = {"Real": "real", "Fake": "fake"}


def label_from_path(rel_path: str) -> str | None:
    """Real or fake from the folder under a speaker folder."""
    path = PurePosixPath(rel_path)
    if path.suffix.lower() != ".wav" or len(path.parts) < 3:
        return None
    if not _SPEAKER.match(path.parts[-3]):
        return None
    return _LABELS.get(path.parts[-2])


def build_manifest(root: Path) -> pd.DataFrame:
    """One row per clip found under ``root``."""
    rows = []
    for rel in iter_files(root):
        label = label_from_path(rel)
        if label is None:
            continue
        path = PurePosixPath(rel)
        speaker = path.parts[-3]
        gender = {"F": "female", "M": "male"}[_SPEAKER.match(speaker).group(2)]
        rows.append(
            {
                "item_id": f"{DATASET}/{speaker}/{label}/{path.stem}",
                "dataset": DATASET,
                "rel_path": rel,
                "modality": "audio",
                "label": label,
                "language": "bn",
                "subject_id": speaker,
                "source_subject_id": speaker,
                "g_gender": gender,
                "g_gender_src": "dataset",
            }
        )
    return pd.DataFrame(rows)
