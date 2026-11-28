"""GenVideo-Val (Chen et al., arXiv 2405.19707, ModelScope ``cccnju/Gen-Video``, CC BY-NC 4.0).

Layout (``GenVideo-Val.zip``, listed by range reads 2026-11-28):
``GenVideo-Val/Real/msrvtt_<n>.mp4`` (10,000 MSR-VTT videos) and
``GenVideo-Val/Fake/<generator>/<file>`` (8,584 videos from ten generators, mostly mp4, some
gif and mov). The zip also carries macOS junk (``__MACOSX/``, ``.DS_Store``) and a few stray
images, none of them items. Labels and the generator come from the folders.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pandas as pd

from . import iter_files

DATASET = "genvideo_val"
_VIDEO = {".mp4", ".gif", ".mov"}


def label_from_path(rel_path: str) -> str | None:
    """``real`` under ``GenVideo-Val/Real/``, ``fake`` under ``GenVideo-Val/Fake/<generator>/``."""
    parts = PurePosixPath(rel_path).parts
    if parts[0] != "GenVideo-Val" or PurePosixPath(rel_path).suffix.lower() not in _VIDEO:
        return None
    if len(parts) == 3 and parts[1] == "Real":
        return "real"
    if len(parts) == 4 and parts[1] == "Fake":
        return "fake"
    return None


def sample_stratum(rel_path: str) -> str:
    """Sample per generator, so every one of them shows up."""
    parts = PurePosixPath(rel_path).parts
    return parts[2] if len(parts) == 4 else "real"


def build_manifest(root: Path) -> pd.DataFrame:
    """One row per video under ``root``."""
    rows = []
    for rel in iter_files(root):
        label = label_from_path(rel)
        if label is None:
            continue
        path = PurePosixPath(rel)
        method = path.parts[2] if label == "fake" else None
        rows.append(
            {
                "item_id": f"{DATASET}/{method or 'real'}/{path.name}",
                "dataset": DATASET,
                "rel_path": rel,
                "modality": "video",
                "label": label,
                "method": method,
                "method_family": "fully_synthetic" if label == "fake" else None,
            }
        )
    return pd.DataFrame(rows)
