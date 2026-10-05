"""SEA-Spoof (Wu et al., 2025), real and spoofed speech in seven languages, Hindi and Tamil
among them. Gated with manual approval, so the layout is only known from the card and paper
(2026-10-05): parquet shards under ``data/<split>/`` with columns ``language``, ``label``,
``spoof_type``, ``source_model``, ``source_dataset``, ``speaker_or_voice``, ``category``
(offline or online spoof) and ``split``, the audio embedded. Write the builder against the
real schema once access is granted (``label_from_row`` and ``ROW_COLUMNS``, as in
``indictts_challenge.py``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_manifest(root: Path) -> pd.DataFrame:
    """Not written yet: the schema is gated."""
    raise NotImplementedError("write this builder against the delivered files")
