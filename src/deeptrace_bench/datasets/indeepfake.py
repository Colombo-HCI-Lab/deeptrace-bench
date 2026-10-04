"""InDeepFake. Layout unknown until access is granted.

Known: 389 real and 4,680 fake videos, seven Indian languages, labels for gender and age
group, generators FSGAN, FaceSwap, DeepFaceLab and TTS plus Wav2Lip.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_manifest(root: Path) -> pd.DataFrame:
    """Build the manifest from the files under ``root``. Not written yet; see module doc."""
    raise NotImplementedError("write this builder against the downloaded files")
