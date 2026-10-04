"""BanglaFake.

Layout: a single ``final_data.zip`` (5.6 GB); what is inside is unverified until downloaded.
Per the GitHub README: 12,260 real and 13,260 VITS fakes from 7 speakers, no official split.
``language`` bn; ``method`` vits for fakes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_manifest(root: Path) -> pd.DataFrame:
    """Build the manifest from the files under ``root``. Not written yet; see module doc."""
    raise NotImplementedError("write this builder against the downloaded files")
