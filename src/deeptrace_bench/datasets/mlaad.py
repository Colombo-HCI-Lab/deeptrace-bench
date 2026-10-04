"""MLAAD (Multi-Language Audio Anti-Spoofing Dataset), fake clips only.

Layout (checked on Hugging Face at revision ``30c3dec``, 2026-10-05):
``fake/<lang>/<generator>/<file>.wav`` with a ``meta.csv`` in each generator folder (path,
original file, language, duration, and so on). Folder names are ISO 639-1 codes; Sinhala is
one folder, ``fake/si/Edge-TTS`` (1,000 clips). Every item is fake: ``method`` is the
generator folder (lower case) and ``language`` its parent. File names come from the English
M-AILABS books the text was translated from, not from speakers, so ``subject_id`` is null.

Its real counterpart, M-AILABS, has no South Asian language, so evalsets pair it with real
speech from another corpus (``openslr_sinhala`` for Sinhala): cross-corpus, never headline.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pandas as pd

from . import iter_files

DATASET = "mlaad"


def label_from_path(rel_path: str) -> str | None:
    """``fake`` for ``fake/<lang>/<generator>/<file>.wav``, None for anything else."""
    path = PurePosixPath(rel_path)
    if len(path.parts) != 4 or path.parts[0] != "fake" or path.suffix.lower() != ".wav":
        return None
    return "fake"


def build_manifest(root: Path) -> pd.DataFrame:
    """One row per clip found under ``root``."""
    rows = []
    for rel in iter_files(root):
        if label_from_path(rel) is None:
            continue
        path = PurePosixPath(rel)
        _, lang, generator, _ = path.parts
        rows.append(
            {
                "item_id": f"{DATASET}/{lang}/{generator}/{path.stem}",
                "dataset": DATASET,
                "rel_path": rel,
                "modality": "audio",
                "label": "fake",
                "method": generator.lower(),
                "method_family": "tts",
                "language": lang,
            }
        )
    return pd.DataFrame(rows)
