"""MAVOS-DD (multilingual audio-video open-set deepfake detection): what every language shares.

The Hugging Face repo keeps one folder per language, ``<language>/<method>/<uuid>.mp4``, with
the same generators in each (file listing checked at revision ``3ea7e6d``, 2026-11-08). Labels
come from the folder. Per-track labels (``label_video``, ``label_audio``) say which track a
method changed; where nobody has checked, the track is null, and evalsets scoring that track
leave those items out. ``mavos_dd_hi`` and ``mavos_dd_de`` are thin modules over this one.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pandas as pd

from . import iter_files

# method -> (label_video, label_audio, method_family); None where unchecked
METHODS: dict[str, tuple[str, str | None, str | None]] = {
    "real": ("real", "real", None),
    "knnvc": ("real", "fake", "vc"),
    "freevc": ("real", "fake", "vc"),
    "inswapper": ("fake", None, "face_swap"),
    "hififace": ("fake", None, "face_swap"),
    "roop": ("fake", None, "face_swap"),
    "echomimic": ("fake", None, "talking_head"),
    "sonic": ("fake", None, "talking_head"),
    "memo": ("fake", None, "talking_head"),
    "liveportrait": ("fake", None, "reenactment"),
}


def label_from_path(rel_path: str, folder: str, methods: frozenset[str]) -> str | None:
    """``real`` under ``<folder>/real/``, ``fake`` under one of ``methods``' folders."""
    path = PurePosixPath(rel_path)
    if len(path.parts) != 3 or path.parts[0] != folder or path.suffix.lower() != ".mp4":
        return None
    if path.parts[1] not in methods:
        return None
    return "real" if path.parts[1] == "real" else "fake"


def sample_stratum(rel_path: str) -> str:
    """Sample per method, so every generator (and the only fakes with known audio) shows up."""
    return PurePosixPath(rel_path).parts[1]


def build_manifest(
    root: Path, dataset: str, folder: str, language: str, methods: frozenset[str]
) -> pd.DataFrame:
    """One row per video of one language found under ``root``."""
    rows = []
    for rel in iter_files(root):
        label = label_from_path(rel, folder, methods)
        if label is None:
            continue
        path = PurePosixPath(rel)
        method = path.parts[1]
        video, audio, family = METHODS[method]
        rows.append(
            {
                "item_id": f"{dataset}/{method}/{path.stem}",
                "dataset": dataset,
                "rel_path": rel,
                "modality": "audio_video",
                "label": label,
                "label_video": video,
                "label_audio": audio,
                "method": None if method == "real" else method,
                "method_family": family,
                "language": language,
            }
        )
    return pd.DataFrame(rows)
