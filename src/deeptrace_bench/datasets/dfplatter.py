"""DF-Platter. Layout unknown until the licence is signed.

Known: sets A (single face), B and C (multiple faces), c23 and c40 compression, with gender,
age, Fitzpatrick skin tone and occlusion labels. The builder also draws the stratified
sample (by skin tone, gender and method) and marks those rows with ``split`` "sample".
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_manifest(root: Path) -> pd.DataFrame:
    """Build the manifest from the files under ``root``. Not written yet; see module doc."""
    raise NotImplementedError("write this builder against the downloaded files")
