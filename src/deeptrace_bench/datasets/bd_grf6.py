"""BD-GRF6 (Zenodo 18656689, 2026, CC BY 4.0): real and fake Bangla speech of Bangladeshi
speakers in six classes, real and fake for each of male, female and third gender.

Layout (checked by range reads, 2026-10-05): ``Real and Fake.zip`` (3.0 GB) holds one zip per
class, ``Real and Fake/<Gender> <Real|Fake>.zip`` with ``<Gender>`` ``Femail`` (sic),
``Male`` or ``Third Gender``, and each of those holds ``<Gender> <Real|Fake>/<name>.wav``
(``Femail Fake.zip``: 2,380 WAV files; ``Third Gender Real.zip``: 2,022 MP3 files). 13,566
clips, 18.7 h in all.

The inner zips are compressed inside the outer one, so the builder extracts each into a
folder named after it (once, marked by ``.extracted-<name>``) and the sampler fetches whole
inner zips (``nested_zip_label``). ``label`` and ``g_gender`` (``female``, ``male``,
``third_gender``; source ``dataset``) come from the class folder. The record names no
generator and no speakers, so ``method`` and ``subject_id`` stay null and nobody can say yet
whether the fakes clone the real speakers.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

from . import iter_files

DATASET = "bd_grf6"
_CLASS = re.compile(r"^(Femail|Female|Male|Third Gender) (Real|Fake)$")
_GENDERS = {"Femail": "female", "Female": "female", "Male": "male", "Third Gender": "third_gender"}
_AUDIO = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def _class(name: str) -> tuple[str, str] | None:
    match = _CLASS.match(name)
    if match is None:
        return None
    return _GENDERS[match.group(1)], match.group(2).lower()


def nested_zip_label(member: str) -> str | None:
    """The label of every clip in an inner class zip, from its name."""
    path = PurePosixPath(member)
    if path.suffix.lower() != ".zip":
        return None
    found = _class(path.stem)
    return found[1] if found else None


def label_from_path(rel_path: str) -> str | None:
    """The label of an audio file inside a class folder."""
    path = PurePosixPath(rel_path)
    if path.suffix.lower() not in _AUDIO:
        return None
    found = next((_class(part) for part in reversed(path.parts[:-1]) if _class(part)), None)
    return found[1] if found else None


def _extract_inner(root: Path) -> None:
    for archive in sorted(root.rglob("*.zip")):
        if nested_zip_label(archive.relative_to(root).as_posix()) is None:
            continue
        marker = archive.parent / f".extracted-{archive.name}"
        if marker.exists():
            continue
        target = archive.with_suffix("")
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                out = (target / info.filename).resolve()
                if not out.is_relative_to(target.resolve()):
                    raise ValueError(f"{archive}: member {info.filename} leaves its folder")
            zf.extractall(target)
        marker.touch()


def build_manifest(root: Path) -> pd.DataFrame:
    """One row per clip, after extracting any inner class zips found under ``root``."""
    _extract_inner(root)
    rows = []
    for rel in iter_files(root):
        label = label_from_path(rel)
        if label is None:
            continue
        path = PurePosixPath(rel)
        gender = next(_class(p)[0] for p in reversed(path.parts[:-1]) if _class(p))
        rows.append(
            {
                "item_id": f"{DATASET}/{gender}_{label}/{path.stem}",
                "dataset": DATASET,
                "rel_path": rel,
                "modality": "audio",
                "label": label,
                "language": "bn",
                "g_gender": gender,
                "g_gender_src": "dataset",
            }
        )
    return pd.DataFrame(rows)
