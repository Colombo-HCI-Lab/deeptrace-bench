"""Comprehensive Deepfake Detection Dataset: real and synthetic frames from Roop and Akool.

Mendeley Data ``pdcp9mjy3z`` version 3 (CC BY 4.0), from Daffodil International University.
A **pipeline test**: its faces are South Asian and it needs no approval, which makes it the
first real data the image path runs on, but its numbers are never reported.

Layout (read from the zip's central directory over HTTP range requests, 2026-10-04): one
ZIP64 archive, ``deepfake_dataset.zip``, with 110,275 entries:

- ``deepfake_dataset/real/<source video>_frame_<k>_face_0.jpg``: 3,744 frames of 30 real
  videos.
- ``deepfake_dataset/deepfake/<name>_Vid-<n>_Tech-<t>_frame_<k>_face_0.jpg`` (the stem
  varies; see ``_TECH``): 104,200 frames of the fakes, 49,997 Tech-1 and 54,203 Tech-2 (the
  page says 106,948). ``<name>`` is the person whose face was swapped in.
- ``__MACOSX/`` and ``.DS_Store``: archive junk, skipped.

Every item is a 500 x 500 JPEG that is already a tight face crop, so ``modality`` is
``image``. The dataset page says the fakes were made with Roop and Akool but not which of
``Tech-1`` and ``Tech-2`` is which, so ``method`` stays ``tech_1`` / ``tech_2``. Frames of
one video are near-duplicates: never split them across train and test.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

import pandas as pd

from . import iter_files

DATASET = "mendeley_roop_akool"
_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_FOLDER_LABELS = {"real": "real", "deepfake": "fake"}
# File names vary ("..._Vid-3_Tech-2_frame...", "..._3_Tech-02.mp4_frame..."), but every fake
# names its tool as Tech-<n>, zero-padded or not (checked on all 104,200 fakes, 2026-10-04).
_TECH = re.compile(r"Tech-0*(\d+)(?!\d)")


def label_from_path(rel_path: str) -> str | None:
    """``real`` or ``fake`` from the folder, or None for anything that isn't a frame."""
    path = PurePosixPath(rel_path)
    if "__MACOSX" in path.parts or path.name.startswith("."):
        return None
    if path.suffix.lower() not in _EXTENSIONS or len(path.parts) < 2:
        return None
    return _FOLDER_LABELS.get(path.parts[-2])


def build_manifest(root: Path) -> pd.DataFrame:
    """One row per frame found under ``root`` (the full archive or a sample of it)."""
    rows = []
    for rel in iter_files(root):
        label = label_from_path(rel)
        if label is None:
            continue
        path = PurePosixPath(rel)
        local = PurePosixPath(*path.parts[-2:]).with_suffix("")
        tech = _TECH.search(path.name) if label == "fake" else None
        rows.append(
            {
                "item_id": f"{DATASET}/{local}",
                "dataset": DATASET,
                "rel_path": rel,
                "modality": "image",
                "label": label,
                "method": f"tech_{tech.group(1)}" if tech else None,
                "method_family": "face_swap" if label == "fake" else None,
            }
        )
    return pd.DataFrame(rows)
