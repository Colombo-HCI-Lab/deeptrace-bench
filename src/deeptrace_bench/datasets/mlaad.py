"""MLAAD, South Asian folders only.

Layout (from the maintainers): ``fake/<lang>/<generator>/*.wav`` with a ``meta.csv`` per
folder (path, original file, language, is-original-language, duration, training data).
1,000 clips per language and generator. All items are fake; ``method`` is the generator.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_manifest(root: Path) -> pd.DataFrame:
    """Build the manifest from the files under ``root``. Not written yet; see module doc."""
    raise NotImplementedError("write this builder against the downloaded files")
