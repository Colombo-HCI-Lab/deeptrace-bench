"""IndicSynth.

Layout (checked on Hugging Face, 2026-10-03): ``<Language>/train-NNNNN-of-NNNNN.parquet``
per language, audio embedded in the parquet rows. Columns include Generative Model
(xtts_v2, vits, freevc24), Source and Target Speaker ID, Gender, Source and Target Reference
Audio, and TTS Transcript.

The builder extracts each row's audio to ``<Language>/wav/<row id>.wav`` under the dataset
dir (once) so the loader reads plain files. All items are fake. ``subject_id`` is the target
speaker, ``source_subject_id`` the source speaker; ``g_gender`` from Gender with source
``dataset``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_manifest(root: Path) -> pd.DataFrame:
    """Build the manifest from the files under ``root``. Not written yet; see module doc."""
    raise NotImplementedError("write this builder against the downloaded files")
