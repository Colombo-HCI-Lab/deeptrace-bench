"""MAVOS-DD, Hindi subset (multilingual audio-video open-set deepfake detection).

Layout (file listing checked on Hugging Face at revision ``3ea7e6d``, 2026-10-05; the files
themselves need the dataset's terms accepted): ``hindi/<method>/<uuid>.mp4``, 7,513 videos,
35.7 GB. ``real`` holds 3,632 real videos (from YouTube); eight generators made the 3,881
fakes:

- face swaps, video track fake: ``inswapper`` 1,051, ``hififace`` 320, ``roop`` 186;
- audio-driven talking heads, video fake: ``echomimic`` 481, ``sonic`` 404, ``memo`` 208;
- reenactment, video fake: ``liveportrait`` 400;
- voice conversion, audio fake and video untouched: ``knnvc`` 831.

Labels come from the folder. Per-track labels (``label_video``, ``label_audio``) say which
track a method changed; where nobody has checked, the track is null, and evalsets scoring that
track leave those items out. The audio of the face-swap, talking-head and reenactment videos
stays null until the dataset's metadata (a ``datasets`` arrow file at the repo root, readable
once the terms are accepted) settles whether it is the original or generated. No speaker or
demographic labels. Hindi is an open-set language in MAVOS-DD: test only in the paper.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pandas as pd

from . import iter_files

DATASET = "mavos_dd_hi"
# method -> (label_video, label_audio, method_family); None where unchecked
_METHODS: dict[str, tuple[str, str | None, str | None]] = {
    "real": ("real", "real", None),
    "knnvc": ("real", "fake", "vc"),
    "inswapper": ("fake", None, "face_swap"),
    "hififace": ("fake", None, "face_swap"),
    "roop": ("fake", None, "face_swap"),
    "echomimic": ("fake", None, "lip_sync"),
    "sonic": ("fake", None, "lip_sync"),
    "memo": ("fake", None, "lip_sync"),
    "liveportrait": ("fake", None, "reenactment"),
}


def label_from_path(rel_path: str) -> str | None:
    """``real`` under ``hindi/real/``, ``fake`` under a known generator folder."""
    path = PurePosixPath(rel_path)
    if len(path.parts) != 3 or path.parts[0] != "hindi" or path.suffix.lower() != ".mp4":
        return None
    method = _METHODS.get(path.parts[1])
    if method is None:
        return None
    return "real" if path.parts[1] == "real" else "fake"


def sample_stratum(rel_path: str) -> str:
    """Sample per method, so every generator (and the only fakes with known audio) shows up."""
    return PurePosixPath(rel_path).parts[1]


def build_manifest(root: Path) -> pd.DataFrame:
    """One row per video found under ``root``."""
    rows = []
    for rel in iter_files(root):
        label = label_from_path(rel)
        if label is None:
            continue
        path = PurePosixPath(rel)
        method = path.parts[1]
        video, audio, family = _METHODS[method]
        rows.append(
            {
                "item_id": f"{DATASET}/{method}/{path.stem}",
                "dataset": DATASET,
                "rel_path": rel,
                "modality": "audio_video",
                "label": label,
                "label_video": video,
                "label_audio": audio,
                "method": None if method == "real" else method,
                "method_family": family,
                "language": "hi",
            }
        )
    return pd.DataFrame(rows)
