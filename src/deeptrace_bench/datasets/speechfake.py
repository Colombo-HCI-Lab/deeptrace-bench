"""SpeechFake (Huang et al., ACL 2025, inclusionAI/SPEECHFAKE on ModelScope), the South Asian
part.

Layout (checked by range reads, 2026-10-05): the multilingual fakes in one split zip
(``MD.z01``, ``MD.z02``, ``MD.zip``) as ``MD/<model>/<language>/...wav`` (Edge-TTS under a
voice folder, ``MD/EdgeTTS/hi/hi-IN-SwaraNeural/...``; SeedVC flat,
``MD/SeedVC/hi/<LibriTTS utterance>_common_voice_hi_<clip>.wav``), and the real speech in
``Real/CommonVoice.zip`` as ``CommonVoice/<language>/<split>/...wav`` (the metadata CSVs write
``Real/CommonVoice/...``; both are read). The metadata CSVs (``metadata.zip``) carry nothing
the path doesn't: label, generator kind, model, language.

``label`` from the top folder, ``method`` the model (``EdgeTTS``: TTS; ``SeedVC``: voice
conversion), ``language`` from the path. An Edge-TTS voice is its ``subject_id``; the other
items name no speaker.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pandas as pd

from . import iter_files

DATASET = "speechfake"
LANGUAGES = {"hi", "bn", "ta", "mr", "ml"}
_FAMILIES = {"EdgeTTS": "tts", "SeedVC": "vc"}


def _parse(rel_path: str) -> tuple[str, str, str | None, PurePosixPath] | None:
    """``(label, language, model, path)`` for a South Asian clip, else None."""
    path = PurePosixPath(rel_path)
    if path.suffix.lower() != ".wav":
        return None
    parts = path.parts
    if len(parts) >= 4 and parts[0] == "MD" and parts[2] in LANGUAGES:
        return "fake", parts[2], parts[1], path
    if parts[:1] == ("Real",):
        parts = parts[1:]
    if len(parts) >= 3 and parts[0] == "CommonVoice" and parts[1] in LANGUAGES:
        return "real", parts[1], None, path
    return None


def label_from_path(rel_path: str) -> str | None:
    """Fake under ``MD/``, real under ``CommonVoice/``, South Asian languages only."""
    found = _parse(rel_path)
    return found[0] if found else None


def sample_stratum(rel_path: str) -> str:
    """Sample per language and model, so Hindi's voice conversion and every language turn up."""
    found = _parse(rel_path)
    return f"{found[1]}/{found[2]}" if found else ""


def build_manifest(root: Path) -> pd.DataFrame:
    """One row per South Asian clip found under ``root``."""
    rows = []
    for rel in iter_files(root):
        found = _parse(rel)
        if found is None:
            continue
        label, language, model, path = found
        voice = path.parts[3] if model == "EdgeTTS" and len(path.parts) > 4 else None
        rows.append(
            {
                "item_id": f"{DATASET}/{path.with_suffix('').as_posix()}",
                "dataset": DATASET,
                "rel_path": rel,
                "modality": "audio",
                "label": label,
                "method": model,
                "method_family": _FAMILIES.get(model) if model else None,
                "language": language,
                "subject_id": voice,
            }
        )
    return pd.DataFrame(rows)
