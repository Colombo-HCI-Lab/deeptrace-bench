"""IndicSynth (Sharma et al., ACL 2025), synthetic speech in 12 Indian languages.

Layout (checked on Hugging Face at revision ``c0a1038``, 2026-10-05):
``<Language>/train-NNNNN-of-MMMMM.parquet`` per language (``Hindi``: 107 shards, 205,938
rows, 56 GB), rows of 100 per row group, the WAV bytes embedded in an ``audio`` struct.
Columns: ``Generative Model`` (``xtts_v2``, ``vits``: TTS; ``freevc24``: voice conversion),
``Target Speaker ID`` (int), ``Source Speaker_ID`` (float, null for TTS), ``Gender`` (Male or
Female), ``Source Reference Audio``, ``Target Reference Audio`` and ``TTS Transcript``. Shards
are sorted by generator, so a small sample covers whichever generators its shards hold.

Every row is fake, generated from IndicSUPERB (Kathbath) speakers: the reference file names
follow Kathbath's ``<utterance>-<speaker>-<m|f>`` scheme, so ``subject_id`` is the target
speaker id and ``source_subject_id`` the source speaker of a voice conversion, both as
Kathbath numbers. ``g_gender`` from ``Gender`` (source ``dataset``). Audio is written out once
to ``<Language>/wav/<shard>-<row>.wav``; ``item_id`` is ``indicsynth/<Language>/<shard>/<row>``.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from ._parquet import iter_rows, write_audio

DATASET = "indicsynth"
ROW_COLUMNS = ["Generative Model"]
_LANGUAGES = {
    "Hindi": "hi",
    "Bengali": "bn",
    "Urdu": "ur",
    "Tamil": "ta",
    "Marathi": "mr",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Telugu": "te",
    "Odia": "or",
    "Sanskrit": "sa",
}
_FAMILIES = {"xtts_v2": "tts", "vits": "tts", "freevc24": "vc"}
_COLUMNS = ["audio", "Generative Model", "Target Speaker ID", "Source Speaker_ID", "Gender"]


def label_from_row(row: dict[str, Any]) -> str | None:
    """Every IndicSynth row is fake."""
    return "fake" if row.get("Generative Model") else None


def row_stratum(row: dict[str, Any]) -> str:
    """Sample per generator where the sampled shards hold more than one."""
    return str(row.get("Generative Model"))


def _speaker(value: Any) -> str | None:
    if value is None or value != value:  # None or NaN
        return None
    return str(int(value))


def build_manifest(root: Path) -> pd.DataFrame:
    """One row per parquet row under ``root``, its audio written out as a WAV file."""
    rows = []
    for shard, number, row in iter_rows(root, "*/*.parquet", _COLUMNS):
        folder = PurePosixPath(shard).parts[0]
        stem = PurePosixPath(shard).stem
        rel = write_audio(root, f"{folder}/wav/{stem}-{number:06d}", row["audio"])
        method = row["Generative Model"]
        gender = (row.get("Gender") or "").lower() or None
        rows.append(
            {
                "item_id": f"{DATASET}/{folder}/{stem}/{number}",
                "dataset": DATASET,
                "rel_path": rel,
                "modality": "audio",
                "label": "fake",
                "method": method,
                "method_family": _FAMILIES.get(method),
                "language": _LANGUAGES.get(folder),
                "subject_id": _speaker(row.get("Target Speaker ID")),
                "source_subject_id": _speaker(row.get("Source Speaker_ID")),
                "g_gender": gender,
                "g_gender_src": "dataset" if gender else None,
            }
        )
    return pd.DataFrame(rows)
