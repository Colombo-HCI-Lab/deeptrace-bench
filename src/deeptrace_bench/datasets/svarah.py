"""Svarah (Javed et al., Interspeech 2023, AI4Bharat), real Indian-accented English.

Gated (accept the terms), so the schema is unread (2026-10-05): ``data/test-0000N-of-00003.
parquet`` with the audio embedded and per-speaker metadata (gender, age group, native state
and district, first language, per the paper). Every item is real. Write the builder against
the delivered schema (``label_from_row`` returning "real", as in ``indicsynth.py``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_manifest(root: Path) -> pd.DataFrame:
    """Not written yet: the schema is gated."""
    raise NotImplementedError("write this builder against the delivered files")
