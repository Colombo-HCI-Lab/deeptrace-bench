"""IndicTTS Deepfake Challenge data (SherryT997/IndicTTS-Deepfake-Challenge-Data, CC BY 4.0).

Real and TTS speech in 16 Indian languages, Nepali among them (no Urdu, Punjabi or Sinhala).
Layout (checked on Hugging Face at revision ``5734751``, 2026-10-05): ``data/train-NNNNN-of-
00035.parquet`` (31,102 rows) and ``data/test-*.parquet`` (2,635 rows whose ``is_tts`` is -1,
so unusable here and not fetched), rows of 100 per row group. Columns: ``id``, ``language``
(the language's English name), ``text``, ``is_tts`` (1 TTS, 0 real) and ``audio`` (16 kHz,
bytes embedded).

The card names neither the TTS system nor the real recordings' source. The ids start with a
language code and the speaker's gender (``ASM_F_ANGER_00342``, ``en_f_lj_LJ031-0022``): the
Indian languages look like a studio emotional-speech corpus with one female and one male
voice per language, while the English rows include LJSpeech (``lj``) and LibriSpeech
(``libri``) utterances, which the config lists in ``derived_from``. ``subject_id`` is
``<CODE>_<GENDER>`` and ``g_gender`` comes from the id (source ``dataset``). ``method``
stays null for fakes (the generator is unnamed). Audio is written once to
``wav/<shard>-<row>.<ext>``; ``item_id`` is ``indictts_challenge/<shard>/<row>``.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from ._parquet import iter_rows, write_audio

DATASET = "indictts_challenge"
ROW_COLUMNS = ["is_tts", "language"]
# ISO 639-1 where there is one, else 639-3 (Bodo, Dogri, Manipuri)
LANGUAGES = {
    "Assamese": "as",
    "Bengali": "bn",
    "Bodo": "brx",
    "Dogri": "doi",
    "English": "en",
    "Gujarati": "gu",
    "Hindi": "hi",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Manipuri": "mni",
    "Marathi": "mr",
    "Nepali": "ne",
    "Odia": "or",
    "Sanskrit": "sa",
    "Tamil": "ta",
    "Telugu": "te",
}
_GENDERS = {"F": "female", "M": "male"}


def label_from_row(row: dict[str, Any]) -> str | None:
    """``is_tts`` 1 is fake, 0 real; the test split's -1 is unlabelled."""
    return {1: "fake", 0: "real"}.get(row.get("is_tts"))


def row_stratum(row: dict[str, Any]) -> str:
    """Sample per language where the sampled row groups hold more than one."""
    return str(row.get("language"))


def build_manifest(root: Path) -> pd.DataFrame:
    """One row per labelled parquet row under ``root``, its audio written out."""
    rows = []
    for shard, number, row in iter_rows(
        root, "data/train-*.parquet", ["id", *ROW_COLUMNS, "audio"]
    ):
        label = label_from_row(row)
        if label is None:
            continue
        stem = PurePosixPath(shard).stem
        rel = write_audio(root, f"wav/{stem}-{number:06d}", row["audio"])
        code, letter, *_ = (row.get("id") or "").split("_") + ["", ""]
        gender = _GENDERS.get(letter.upper())
        rows.append(
            {
                "item_id": f"{DATASET}/{stem}/{number}",
                "dataset": DATASET,
                "rel_path": rel,
                "modality": "audio",
                "label": label,
                "method": None,
                "method_family": "tts" if label == "fake" else None,
                "language": LANGUAGES.get(row.get("language")),
                "subject_id": f"{code.upper()}_{letter.upper()}" if code and gender else None,
                "split": "train",
                "g_gender": gender,
                "g_gender_src": "dataset" if gender else None,
            }
        )
    return pd.DataFrame(rows)
