"""Deepfake-Eval-2024.

Layout partly known: ``video/``, ``audio/``, ``image/`` and ``*-metadata-publish.csv`` per
modality; columns unverified until access is granted. Evaluation only: never use it for
training or fine-tuning.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_manifest(root: Path) -> pd.DataFrame:
    """Build the manifest from the files under ``root``. Not written yet; see module doc."""
    raise NotImplementedError("write this builder against the downloaded files")
