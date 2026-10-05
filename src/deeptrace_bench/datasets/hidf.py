"""HiDF (Kang et al., KDD 2025, Zenodo 16140829, CC BY-NC 4.0): deepfakes made with commercial
tools and curated until people couldn't tell them from real.

Layout (checked by range reads, 2026-10-05): ``Real-vid.zip`` holds ``Real-vid/<id>.mp4`` and
``Fake-vid.zip`` ``Fake-vid/<base>_<face>.mp4`` (4,361 each): the real video ``<base>`` with
the face of ``<face>`` swapped in. ``metadata.csv`` gives ``ID``, ``Img/Vid``, ``Base/Swap``,
``Race`` (White, Asian, Latino, Black, Indian), ``Gender`` and ``Age`` (Adult, Child,
Elderly). Image and video ids are separate namespaces (``c00003`` is a different person in
each): a real video's subject is its ``Vid`` row, and a fake's swapped-in face is an ``Img``
row (all 4,361 donor faces are, against 350 among the ``Vid`` rows). The 62k images (40 GB)
aren't fetched.

``subject_id`` is the video's base id and ``source_subject_id`` a fake's face id. The group
columns describe the face on screen: a real video's own subject, a fake's donor face
(``g_race``, ``g_gender``, ``g_age``, source ``dataset``). ``g_race`` "indian" is the South
Asian slice (90 base videos). ``method`` stays null: the record doesn't say which tool made
which video.
"""

from __future__ import annotations

import io
from pathlib import Path, PurePosixPath

import pandas as pd

from . import iter_files

DATASET = "hidf"
METADATA_FILES = ["metadata.csv"]
_FOLDERS = {"Real-vid": "real", "Fake-vid": "fake"}


def label_from_path(rel_path: str) -> str | None:
    """Real or fake from the video folder; images and metadata are no items."""
    path = PurePosixPath(rel_path)
    if path.suffix.lower() != ".mp4" or len(path.parts) < 2:
        return None
    return _FOLDERS.get(path.parts[-2])


def people(metadata: bytes, kind: str) -> dict[str, dict[str, str | None]]:
    """``{id: {race, gender, age}}`` for the ``Vid`` or ``Img`` rows, values lower case."""
    df = pd.read_csv(io.BytesIO(metadata), dtype=str)
    df = df[df["Img/Vid"].str.strip() == kind]
    out = {}
    for row in df.itertuples(index=False):
        values = {
            key: (str(value).strip().lower().replace(" ", "_") or None)
            if isinstance(value, str)
            else None
            for key, value in (("race", row.Race), ("gender", row.Gender), ("age", row.Age))
        }
        out[str(row.ID).strip()] = values
    return out


def build_manifest(root: Path) -> pd.DataFrame:
    """One row per video under ``root``, with the shown face's attributes."""
    meta = root / "metadata.csv"
    videos = people(meta.read_bytes(), "Vid") if meta.exists() else {}
    faces = people(meta.read_bytes(), "Img") if meta.exists() else {}
    rows = []
    for rel in iter_files(root):
        label = label_from_path(rel)
        if label is None:
            continue
        stem = PurePosixPath(rel).stem
        base, _, face = stem.partition("_")
        shown = faces.get(face, {}) if label == "fake" else videos.get(base, {})
        row = {
            "item_id": f"{DATASET}/{label}/{stem}",
            "dataset": DATASET,
            "rel_path": rel,
            "modality": "video",
            "label": label,
            "method_family": "face_swap" if label == "fake" else None,
            "subject_id": base,
            "source_subject_id": face or None,
        }
        for key in ("race", "gender", "age"):
            row[f"g_{key}"] = shown.get(key)
            row[f"g_{key}_src"] = "dataset" if shown.get(key) else None
        rows.append(row)
    return pd.DataFrame(rows)
