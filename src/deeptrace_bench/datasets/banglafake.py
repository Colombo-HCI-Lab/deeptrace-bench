"""BanglaFake (Bengali TTS deepfakes).

Layout (checked in the archive at revision ``1a4bfe3``, 2026-10-05): one zip, ``final_data.zip``,
holding three sub-corpora, each ``final_data/deepfake_data_<sub>/{real_wav,deepfake_wav}/``:

- ``sust``: 9,999 real and 9,999 fake clips. Real ``0NNNN.wav``, fake ``1NNNN.wav``; no
  speaker information.
- ``mozilla``: 2,797 real and 2,797 fake clips, paired by file name
  (``common_voice_s<N>_<i>.wav``). Real clips are Common Voice recordings of five speakers,
  ``s1`` to ``s5``; each fake re-speaks its pair's text in that speaker's voice.
- ``news``: 1,000 real clips with no fakes.

Each sub-corpus has a transcript file (``metadata.csv`` or ``metadata.txt``) and no speaker
or gender column. Totals: 13,796 real and 12,796 fake, not the README's 12,260 and 13,260.
Fakes are VITS. ``g_subcorpus`` records the sub-corpus, because ``news`` has no fakes: an
evalset that kept it would partly pair real news recordings against TTS from other sources.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

import pandas as pd

from . import iter_files

DATASET = "banglafake"
_FOLDERS = {"real_wav": "real", "deepfake_wav": "fake"}
_MOZILLA_SPEAKER = re.compile(r"_s(\d+)_")


def label_from_path(rel_path: str) -> str | None:
    """``real`` under ``real_wav/``, ``fake`` under ``deepfake_wav/``, for ``.wav`` files."""
    path = PurePosixPath(rel_path)
    if len(path.parts) != 4 or path.parts[0] != "final_data" or path.suffix.lower() != ".wav":
        return None
    if not path.parts[1].startswith("deepfake_data_"):
        return None
    return _FOLDERS.get(path.parts[2])


def build_manifest(root: Path) -> pd.DataFrame:
    """One row per clip found under ``root``."""
    rows = []
    for rel in iter_files(root):
        label = label_from_path(rel)
        if label is None:
            continue
        path = PurePosixPath(rel)
        sub = path.parts[1].removeprefix("deepfake_data_")
        speaker = None
        if sub == "mozilla" and (match := _MOZILLA_SPEAKER.search(path.name)):
            speaker = f"mozilla_s{match.group(1)}"
        rows.append(
            {
                "item_id": f"{DATASET}/{sub}/{label}/{path.stem}",
                "dataset": DATASET,
                "rel_path": rel,
                "modality": "audio",
                "label": label,
                "method": "vits" if label == "fake" else None,
                "method_family": "tts" if label == "fake" else None,
                "language": "bn",
                "subject_id": speaker,
                "source_subject_id": speaker,
                "g_subcorpus": sub,
                "g_subcorpus_src": "dataset",
            }
        )
    return pd.DataFrame(rows)
