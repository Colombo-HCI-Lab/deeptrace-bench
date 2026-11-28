"""Counterfactual-Deepfake-Bench, synthetic half (WVU Machine Intelligence Lab, CC BY-NC 4.0).

Layout (Hugging Face at revision ``613e158``, checked 2026-11-28):
``synthetic/<generator>/<race>_<gender>_<age>/<prompt_id>.mp4`` with a ``<prompt_id>.json``
sidecar beside each clip, and ``metadata.csv`` at the root, one row per clip (``file_name`` is
the clip's path). 7,056 fully generated talking-head clips, about 4.6 GB, from four
text-to-video generators (``ltx23``, ``ltx25``, ``wan22``, ``hunyuan15``) over 7 perceived
races x 2 genders x 4 age bands. Only fakes: the authors' real half (CelebV-HQ) isn't
redistributed, so evalsets pair it with reals from another corpus.

The group columns are the demographic cell the prompt asked for, not who the generator drew:
the authors' FairFace check agrees with the full requested cell for only 28 to 48% of clips.
``split`` carries the authors' QC inclusion set: ``primary`` (technical QC and no background
faces, 6,435 clips) or ``excluded``.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pandas as pd

from . import iter_files

DATASET = "counterfactual_deepfake_bench"
METADATA_FILES = ["metadata.csv"]
_GENDER = {"f": "female", "m": "male"}


def label_from_path(rel_path: str) -> str | None:
    """Every clip under ``synthetic/`` is fake; sidecars and metadata are no items."""
    path = PurePosixPath(rel_path)
    if path.suffix.lower() != ".mp4" or path.parts[0] != "synthetic":
        return None
    return "fake"


def sample_stratum(rel_path: str) -> str:
    """Sample per generator and requested race (``synthetic/<gen>/<race>_<g>_<age>/...``)."""
    parts = PurePosixPath(rel_path).parts
    return f"{parts[1]}/{parts[2].rsplit('_', 2)[0]}" if len(parts) > 3 else "other"


def build_manifest(root: Path) -> pd.DataFrame:
    """One row per clip under ``root``, with its requested cell, scene lighting and QC split."""
    meta_path = root / "metadata.csv"
    meta = pd.read_csv(meta_path).set_index("file_name") if meta_path.exists() else None
    rows = []
    for rel in iter_files(root):
        if label_from_path(rel) is None:
            continue
        info = meta.loc[rel] if meta is not None and rel in meta.index else None
        row = {
            "item_id": f"{DATASET}/{PurePosixPath(rel).stem}",
            "dataset": DATASET,
            "rel_path": rel,
            "modality": "video",
            "label": "fake",
            "method": PurePosixPath(rel).parts[1],
            "method_family": "fully_synthetic",
        }
        # A clip without a metadata row keeps these columns missing, so they come out null.
        if info is not None:
            row["split"] = "primary" if int(info["primary"]) == 1 else "excluded"
            row["duration_s"] = float(info["duration_s"])
            groups = {
                "race": info["race"],
                "gender": _GENDER.get(info["gender"], info["gender"]),
                "age": info["age_band"],
                "lighting": info["lighting"],
            }
            for key, value in groups.items():
                row[f"g_{key}"] = value
                row[f"g_{key}_src"] = "dataset"
        rows.append(row)
    return pd.DataFrame(rows)
