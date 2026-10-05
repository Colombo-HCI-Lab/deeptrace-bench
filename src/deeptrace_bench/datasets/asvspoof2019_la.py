"""ASVspoof 2019 Logical Access.

Layout (inside ``LA.zip``, checked by range requests on 2026-10-05; 122,328 entries):

- ``LA/ASVspoof2019_LA_{train,dev,eval}/flac/<utt_id>.flac``: 25,380, 24,986 and 71,933
  utterances.
- ``LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{train.trn,dev.trl,eval.trl}.txt``,
  space-separated: ``speaker_id utt_id - attack_id key``, where ``attack_id`` is ``-`` for
  bona fide or ``A01``..``A19`` and ``key`` is ``bonafide`` or ``spoof``.

Labels live only in the protocol files, so the builder takes them from there
(``METADATA_FILES``), and the sampler reads the same files first; it samples per split
(``sample_stratum``), so the eval-only evalset gets both labels. Columns: ``label`` from
``key``; ``method`` the attack id; ``method_family`` from the challenge's description of each
attack (``tts``, ``vc``, or ``tts_vc`` for A13 to A15, TTS output converted by VC);
``subject_id`` the speaker; ``split`` train, dev or eval; ``language`` en. The partitions
have disjoint speakers.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pandas as pd

DATASET = "asvspoof2019_la"
_PROTOCOL = "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{}.txt"
_SPLITS = {"train": "train.trn", "dev": "dev.trl", "eval": "eval.trl"}
METADATA_FILES = [_PROTOCOL.format(name) for name in _SPLITS.values()]
_LABELS = {"bonafide": "real", "spoof": "fake"}
_FAMILIES = {
    **{f"A{n:02d}": "tts" for n in (1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 16)},
    **{f"A{n:02d}": "vc" for n in (5, 6, 17, 18, 19)},
    **{f"A{n:02d}": "tts_vc" for n in (13, 14, 15)},
}


def _entries(files: dict[str, bytes]):
    """``(split, rel_path, speaker, attack, label)`` for every protocol line."""
    for split, name in _SPLITS.items():
        text = files.get(_PROTOCOL.format(name), b"").decode()
        for line in text.splitlines():
            fields = line.split()
            if len(fields) != 5:
                continue
            speaker, utt, _, attack, key = fields
            rel = f"LA/ASVspoof2019_LA_{split}/flac/{utt}.flac"
            yield split, rel, speaker, None if attack == "-" else attack, _LABELS[key]


def labels_from_metadata(files: dict[str, bytes]) -> dict[str, str]:
    """``{rel_path: label}`` for every utterance in the CM protocols."""
    return {rel: label for _, rel, _, _, label in _entries(files)}


def sample_stratum(rel_path: str) -> str:
    """Sample per split, so the eval-only evalset gets real and fake items."""
    return PurePosixPath(rel_path).parts[1].removeprefix("ASVspoof2019_LA_")


def build_manifest(root: Path) -> pd.DataFrame:
    """One row per protocol entry whose audio is present under ``root``."""
    files = {name: (root / name).read_bytes() for name in METADATA_FILES if (root / name).exists()}
    rows = []
    for split, rel, speaker, attack, label in _entries(files):
        if not (root / rel).exists():
            continue
        rows.append(
            {
                "item_id": f"{DATASET}/{split}/{Path(rel).stem}",
                "dataset": DATASET,
                "rel_path": rel,
                "modality": "audio",
                "label": label,
                "method": attack,
                "method_family": _FAMILIES.get(attack) if attack else None,
                "language": "en",
                "subject_id": speaker,
                "split": split,
            }
        )
    return pd.DataFrame(rows)
