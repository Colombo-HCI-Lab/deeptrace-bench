"""IndicSUPERB (Kathbath), real speech for pairing with IndicSynth.

Layout: tar archives ``testkn_audio.tar`` and ``testunk_audio.tar`` (clean test splits) with
per-language folders of m4a files, plus ``transcripts_n2w.tar``. The builder converts m4a to
16 kHz mono WAV (soundfile cannot read m4a; use ffmpeg) and reads speaker and gender from the
transcript metadata. All items are real.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_manifest(root: Path) -> pd.DataFrame:
    """Build the manifest from the files under ``root``. Not written yet; see module doc."""
    raise NotImplementedError("write this builder against the downloaded files")
