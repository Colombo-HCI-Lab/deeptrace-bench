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

from pathlib import Path

import pandas as pd

from . import mavos_dd

DATASET = "mavos_dd_hi"
# Hindi has no freevc folder; the others' methods are mavos_dd.METHODS.
_METHODS = frozenset(mavos_dd.METHODS) - {"freevc"}
sample_stratum = mavos_dd.sample_stratum


def label_from_path(rel_path: str) -> str | None:
    """``real`` under ``hindi/real/``, ``fake`` under a known generator folder."""
    return mavos_dd.label_from_path(rel_path, "hindi", _METHODS)


def build_manifest(root: Path) -> pd.DataFrame:
    """One row per video found under ``root``."""
    return mavos_dd.build_manifest(root, DATASET, "hindi", "hi", _METHODS)
