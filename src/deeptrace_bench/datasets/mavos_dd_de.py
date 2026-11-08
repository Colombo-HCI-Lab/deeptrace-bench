"""MAVOS-DD, German subset: the language control for MAVOS-DD Hindi.

Layout (file listing checked on Hugging Face at revision ``3ea7e6d``, 2026-11-08):
``german/<method>/<uuid>.mp4``, 10,301 videos, 27.7 GB. ``real`` holds 4,366 real videos (from
YouTube); the eight generators of the Hindi subset made 5,735 fakes (inswapper 898, hififace
246, roop 156, echomimic 299, sonic 1,491, memo 887, liveportrait 908, knnvc 850), and a ninth,
``freevc`` (200, voice conversion like knnvc), adds audio-only fakes. Same generators, same
labelling and the same unknowns as ``mavos_dd_hi``, so a model's drop from German to Hindi on
the same generators is the language and faces, not the kind of fake. German is a training
language in MAVOS-DD, which none of the benchmark's models trained on.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import mavos_dd

DATASET = "mavos_dd_de"
_METHODS = frozenset(mavos_dd.METHODS)
sample_stratum = mavos_dd.sample_stratum


def label_from_path(rel_path: str) -> str | None:
    """``real`` under ``german/real/``, ``fake`` under a known generator folder."""
    return mavos_dd.label_from_path(rel_path, "german", _METHODS)


def build_manifest(root: Path) -> pd.DataFrame:
    """One row per video found under ``root``."""
    return mavos_dd.build_manifest(root, DATASET, "german", "de", _METHODS)
