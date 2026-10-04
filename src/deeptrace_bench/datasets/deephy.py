"""DeePhy. Layout unknown until the licence is signed.

Known: 100 real and 5,040 fake videos (FaceShifter, FaceSwap, FSGAN; single, double and
triple swaps), 10 attributes per video including gender, age group and skin tone.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_manifest(root: Path) -> pd.DataFrame:
    """Build the manifest from the files under ``root``. Not written yet; see module doc."""
    raise NotImplementedError("write this builder against the downloaded files")
