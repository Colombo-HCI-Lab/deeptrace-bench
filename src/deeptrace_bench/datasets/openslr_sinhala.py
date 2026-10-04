"""Large Sinhala ASR training data (OpenSLR SLR52), real speech only.

Layout (checked in ``asr_sinhala_0.zip``, 2026-10-05): ``asr_sinhala/data/<xx>/<id>.flac``
(11,550 clips in this zip, one of 16 split by the first character of the id) plus
``asr_sinhala/utt_spk_text.tsv`` (``id``, anonymised speaker id, transcript, for all ~185k
utterances) and ``LICENSE``. Crowd-sourced, manually checked recordings, CC BY-SA 4.0.

Only the first zip is fetched: it already gives eleven times as many real clips as MLAAD has
Sinhala fakes. ``subject_id`` comes from the tsv, which the sampler copies too. No gender.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pandas as pd

from . import iter_files

DATASET = "openslr_sinhala"
SPEAKERS = "asr_sinhala/utt_spk_text.tsv"
# Not needed for labels, but the builder reads speakers from it, so samples carry it too.
METADATA_FILES = [SPEAKERS]


def label_from_path(rel_path: str) -> str | None:
    """``real`` for every ``.flac`` under ``asr_sinhala/data/``."""
    path = PurePosixPath(rel_path)
    if path.parts[:2] != ("asr_sinhala", "data") or path.suffix.lower() != ".flac":
        return None
    return "real"


def _speakers(root: Path) -> dict[str, str]:
    table = root / SPEAKERS
    if not table.exists():
        return {}
    # Plain tab splitting: transcripts contain quote characters that a csv reader would take
    # as field delimiters and run together across lines.
    speakers = {}
    with table.open(encoding="utf-8") as fh:
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 2:
                speakers[fields[0]] = fields[1]
    return speakers


def build_manifest(root: Path) -> pd.DataFrame:
    """One row per clip found under ``root``."""
    speakers = _speakers(root)
    rows = []
    for rel in iter_files(root):
        if label_from_path(rel) is None:
            continue
        stem = PurePosixPath(rel).stem
        rows.append(
            {
                "item_id": f"{DATASET}/{stem}",
                "dataset": DATASET,
                "rel_path": rel,
                "modality": "audio",
                "label": "real",
                "language": "si",
                "subject_id": speakers.get(stem),
            }
        )
    return pd.DataFrame(rows)
