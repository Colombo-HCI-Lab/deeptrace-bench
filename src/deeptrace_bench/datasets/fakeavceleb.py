"""FakeAVCeleb.

Expected layout (verify on arrival):
``FakeAVCeleb_v1.2/<category>/<ethnicity>/<gender>/<id>/*.mp4`` with categories
RealVideo-RealAudio, RealVideo-FakeAudio, FakeVideo-RealAudio and FakeVideo-FakeAudio, plus
``meta_data.csv``.

For the video evalset, ``label`` is fake when the video track is fake. ``g_ethnicity`` and
``g_gender`` come from the folders (source ``dataset``); "Asian (South)" is the South Asian
group. ``method`` from the metadata (faceswap, fsgan, wav2lip, rtvc and so on).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_manifest(root: Path) -> pd.DataFrame:
    """Build the manifest from the files under ``root``. Not written yet; see module doc."""
    raise NotImplementedError("write this builder against the downloaded files")
