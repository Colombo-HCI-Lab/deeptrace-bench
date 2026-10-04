"""ASVspoof 2019 Logical Access.

Layout (inside ``LA.zip``):

- ``LA/ASVspoof2019_LA_{train,dev,eval}/flac/<utt_id>.flac``
- ``LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{train.trn,dev.trl,eval.trl}.txt``,
  space-separated: ``speaker_id utt_id - attack_id key``, where ``attack_id`` is ``-`` for
  bona fide or ``A01``..``A19`` and ``key`` is ``bonafide`` or ``spoof``.

Columns: ``label`` from ``key``; ``method`` from ``attack_id``; ``subject_id`` from
``speaker_id``; ``split`` is train, dev or eval; ``language`` en.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_manifest(root: Path) -> pd.DataFrame:
    """Build the manifest from the files under ``root``. Not written yet; see module doc."""
    raise NotImplementedError("write this builder against the downloaded files")
