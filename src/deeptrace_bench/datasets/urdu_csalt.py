"""Urdu Deepfake Audio (CSALT-LUMS).

Layout (checked on Hugging Face at revision ``eb8f166``, 2026-10-05; 6,794 clips):

- ``Bonafide/Speaker_NN/Part N/<n>.wav``: 3,398 real clips.
- ``Spoofed_TTS/Speaker_NN/<n>.wav``: 1,698 VITS fakes.
- ``Spoofed_Tacotron/Speaker_NN/<n>.wav``: 1,698 Tacotron fakes.

No metadata files: ``label`` and ``method`` come from the top folder and ``subject_id`` from
``Speaker_NN``. Fakes imitate the same 17 speakers, so ``source_subject_id`` equals
``subject_id``. File names repeat across parts, so the part stays in ``item_id``.
``language`` ur. No official split.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pandas as pd

from . import iter_files

DATASET = "urdu_csalt"
# top folder -> (label, method, method_family)
_TOP = {
    "Bonafide": ("real", None, None),
    "Spoofed_TTS": ("fake", "vits", "tts"),
    "Spoofed_Tacotron": ("fake", "tacotron", "tts"),
}


def label_from_path(rel_path: str) -> str | None:
    """The label from the top folder, for ``.wav`` files under a speaker folder."""
    path = PurePosixPath(rel_path)
    if len(path.parts) < 3 or path.suffix.lower() != ".wav" or path.parts[0] not in _TOP:
        return None
    return _TOP[path.parts[0]][0]


def build_manifest(root: Path) -> pd.DataFrame:
    """One row per clip found under ``root``."""
    rows = []
    for rel in iter_files(root):
        if label_from_path(rel) is None:
            continue
        path = PurePosixPath(rel)
        label, method, family = _TOP[path.parts[0]]
        speaker = path.parts[1]
        local = path.with_suffix("").as_posix().replace(" ", "_")
        rows.append(
            {
                "item_id": f"{DATASET}/{local}",
                "dataset": DATASET,
                "rel_path": rel,
                "modality": "audio",
                "label": label,
                "method": method,
                "method_family": family,
                "language": "ur",
                "subject_id": speaker,
                "source_subject_id": speaker,
            }
        )
    return pd.DataFrame(rows)
