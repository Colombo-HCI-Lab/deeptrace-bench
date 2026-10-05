"""IndicSUPERB's Kathbath (AI4Bharat), real read speech in 12 Indian languages: the real side
of IndicSynth, whose fakes were generated from these speakers.

Layout (object store, checked by range reads 2026-10-05): ``testkn_audio.tar`` (3.6 GB, test
speakers also in train) and ``testunk_audio.tar`` (2.2 GB, unseen speakers), plain tars of
``kb_data_clean_m4a/<language>/<test_known|test_unknown>/audio/<utterance>-<speaker>-<m|f>.m4a``,
plus ``transcripts_n2w.tar``. Tars can't be read in place, so the dataset is downloaded in
full and samples are drawn from that copy. The audio loader decodes m4a itself (PyAV).

Every item is real. ``subject_id`` is the speaker number, the same numbering IndicSynth's
``Target Speaker ID`` uses (its reference files follow the same ``<utt>-<speaker>-<g>`` names),
so the two can be matched per language. ``g_gender`` from the file name (source ``dataset``),
``split`` from the folder.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

import pandas as pd

from . import iter_files

DATASET = "indicsuperb"
LANGUAGES = {
    "bengali": "bn",
    "gujarati": "gu",
    "hindi": "hi",
    "kannada": "kn",
    "malayalam": "ml",
    "marathi": "mr",
    "odia": "or",
    "punjabi": "pa",
    "sanskrit": "sa",
    "tamil": "ta",
    "telugu": "te",
    "urdu": "ur",
}
_NAME = re.compile(r"^(?P<utt>\d+)-(?P<speaker>\d+)-(?P<gender>[mf])$")


def _parts(rel_path: str) -> tuple[str, str, re.Match[str]] | None:
    path = PurePosixPath(rel_path)
    if path.suffix.lower() != ".m4a" or len(path.parts) < 4 or path.parts[-2] != "audio":
        return None
    match = _NAME.match(path.stem)
    language, split = path.parts[-4], path.parts[-3]
    if match is None or language.lower() not in LANGUAGES:
        return None
    return language.lower(), split, match


def label_from_path(rel_path: str) -> str | None:
    """Every Kathbath clip is real; anything else is no item."""
    return "real" if _parts(rel_path) else None


def sample_stratum(rel_path: str) -> str:
    """Sample per language, so every language's evalset gets real items."""
    found = _parts(rel_path)
    return found[0] if found else ""


def build_manifest(root: Path) -> pd.DataFrame:
    """One row per m4a clip found under ``root`` (the extracted tars, or a sample)."""
    rows = []
    for rel in iter_files(root):
        found = _parts(rel)
        if found is None:
            continue
        language, split, match = found
        rows.append(
            {
                "item_id": f"{DATASET}/{language}/{split}/{match['utt']}",
                "dataset": DATASET,
                "rel_path": rel,
                "modality": "audio",
                "label": "real",
                "language": LANGUAGES[language],
                "subject_id": match["speaker"],
                "split": split,
                "g_gender": {"m": "male", "f": "female"}[match["gender"]],
                "g_gender_src": "dataset",
            }
        )
    return pd.DataFrame(rows)
