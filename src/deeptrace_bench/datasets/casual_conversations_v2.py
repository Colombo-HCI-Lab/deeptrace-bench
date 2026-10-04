"""Casual Conversations v2 (Meta, 2023), real videos of consenting participants.

Not yet obtained (request form plus licence, 2026-10-05). 26,467 videos of 5,567
participants in seven countries, India the largest group (spoken Hindi, Tamil, Telugu and
English); every video is real. Labels: Fitzpatrick and Monk skin tone (annotated), age,
gender and language (self-reported), country. Some labels are licensed for evaluation only.

Its use here: false positive rates on real South Asian faces per skin tone and language,
the real half of a fairness comparison. Layout to be written against the delivered copy.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_manifest(root: Path) -> pd.DataFrame:
    """Not written yet: the layout is unknown until the copy arrives."""
    raise NotImplementedError("write this builder against the delivered files")
