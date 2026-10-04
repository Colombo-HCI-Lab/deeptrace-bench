"""FaceForensics++ (c23).

Layout (from the FaceForensics repo):

- ``original_sequences/youtube/c23/videos/<NNN>.mp4``: 1,000 real videos.
- ``manipulated_sequences/<Method>/c23/videos/<target>_<source>.mp4`` for Deepfakes,
  Face2Face, FaceSwap, NeuralTextures and FaceShifter.
- Official splits: ``dataset/splits/{train,val,test}.json`` in the FaceForensics GitHub repo,
  lists of ``[target, source]`` pairs. A real video is in the split of any pair it appears in.

Columns: ``method`` from the folder; ``subject_id`` is the target id and
``source_subject_id`` the source id of a fake; ``split`` from the JSON files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_manifest(root: Path) -> pd.DataFrame:
    """Build the manifest from the files under ``root``. Not written yet; see module doc."""
    raise NotImplementedError("write this builder against the downloaded files")
