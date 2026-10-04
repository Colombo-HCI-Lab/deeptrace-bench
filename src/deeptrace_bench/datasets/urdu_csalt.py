"""Urdu Deepfake Audio (CSALT-LUMS).

Layout (checked on Hugging Face, 2026-10-03):

- ``Bonafide/Speaker_NN/Part N/*.wav``: 3,398 real clips.
- ``Spoofed_TTS/Speaker_NN/...``: 1,698 VITS fakes.
- ``Spoofed_Tacotron/Speaker_NN/...``: 1,698 Tacotron fakes.

No metadata files: ``label`` and ``method`` come from the top folder and ``subject_id`` from
``Speaker_NN``. Fakes imitate the same 17 speakers, so ``source_subject_id`` equals
``subject_id``. ``language`` ur. No official split.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_manifest(root: Path) -> pd.DataFrame:
    """Build the manifest from the files under ``root``. Not written yet; see module doc."""
    raise NotImplementedError("write this builder against the downloaded files")
