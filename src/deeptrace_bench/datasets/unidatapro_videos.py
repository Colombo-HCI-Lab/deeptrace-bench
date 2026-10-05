"""UniDataPro deepfake videos, the free preview on Hugging Face (CC BY-NC-ND 4.0).

A **pipeline test**: the only open, licensed set of paired real and fake *videos* small
enough to fetch on the fly, so it exercises video decoding, frame sampling, face detection
and frame-score averaging. Its numbers are never reported.

Layout (Hugging Face ``UniDataPro/deepfake-videos-dataset`` at ``bdbf7fc9``, 2026-10-04):

- ``video/<i>.mp4`` (one ``.MOV``): five real phone recordings.
- ``deepfake/<i>.mp4`` (one ``.mov``): five fakes, video ``i`` with the face of
  ``image/<i>`` swapped in by one of three online services (the card doesn't say which).
- ``image/<i>.jpg`` and ``DeepFake Videos Dataset.csv``: the source faces and the pairing
  table. Not items. The CSV says ``video/5.mov`` but the file is ``video/5.MOV``, so the
  builder scans files instead of trusting it, matching extensions case-insensitively.

Every video has an audio track (checked 2026-10-05), so the items are ``audio_video``: video
models score their frames and audio-visual models (HAVIC) both tracks. ``label_video`` is the
item's label; ``label_audio`` is real for the real recordings and unknown for the fakes, since
the card doesn't say whether the services kept the original audio.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pandas as pd

from . import iter_files

DATASET = "unidatapro_videos"
_EXTENSIONS = {".mp4", ".mov"}
_FOLDER_LABELS = {"video": "real", "deepfake": "fake"}


def label_from_path(rel_path: str) -> str | None:
    """``real`` for ``video/``, ``fake`` for ``deepfake/``, None for everything else."""
    path = PurePosixPath(rel_path)
    if len(path.parts) != 2 or path.suffix.lower() not in _EXTENSIONS:
        return None
    return _FOLDER_LABELS.get(path.parts[0])


def build_manifest(root: Path) -> pd.DataFrame:
    """One row per video found under ``root``."""
    rows = []
    for rel in iter_files(root):
        label = label_from_path(rel)
        if label is None:
            continue
        path = PurePosixPath(rel)
        rows.append(
            {
                "item_id": f"{DATASET}/{path.parent}/{path.stem}",
                "dataset": DATASET,
                "rel_path": rel,
                "modality": "audio_video",
                "label": label,
                "label_video": label,
                "label_audio": "real" if label == "real" else None,
                "method_family": "face_swap" if label == "fake" else None,
            }
        )
    return pd.DataFrame(rows)
