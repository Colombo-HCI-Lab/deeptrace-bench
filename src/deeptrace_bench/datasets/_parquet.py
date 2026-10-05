"""Shared pieces for datasets that embed their audio in parquet shards (Hugging Face style).

Such a dataset's items are rows, not files: each row carries an ``audio`` struct
(``bytes``, ``path``) next to its metadata. Builders of these datasets define:

- ``ROW_COLUMNS``: the metadata columns that label a row (read without the audio);
- ``label_from_row(row: dict) -> "real" | "fake" | None``, the row counterpart of
  ``label_from_path``; the sampler applies it to rows read over HTTP, ``build_manifest`` to
  local rows, so the two can't disagree;
- optionally ``row_stratum(row: dict) -> str``, as ``sample_stratum`` for files.

A sample keeps each shard's relative path but holds only the chosen rows, with their row
number in the full shard in a ``__source_row`` column (``source_rows``), so item ids are the
same in a sample and in the full copy. ``build_manifest`` writes each row's audio out once as
a plain file (``write_audio``) so the loaders read files as for any other dataset.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

SOURCE_ROW = "__source_row"
_BATCH = 64


def iter_rows(
    root: Path, pattern: str, columns: list[str]
) -> Iterator[tuple[str, int, dict[str, Any]]]:
    """``(shard rel path, row number in the full shard, row)`` for every row of every shard.

    ``columns`` are read in batches, so a shard's audio is never all in memory at once.
    """
    import pyarrow.parquet as pq

    for path in sorted(root.glob(pattern)):
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        rel = path.relative_to(root).as_posix()
        shard = pq.ParquetFile(path)
        has_source = SOURCE_ROW in shard.schema_arrow.names
        wanted = [*columns, SOURCE_ROW] if has_source else columns
        position = 0
        for batch in shard.iter_batches(columns=wanted, batch_size=_BATCH):
            for row in batch.to_pylist():
                source = row.pop(SOURCE_ROW) if has_source else position
                position += 1
                yield rel, int(source), row


def audio_suffix(audio: dict[str, Any]) -> str:
    """The file extension for an embedded clip: from its stored path, else its magic bytes."""
    stored = PurePosixPath(audio.get("path") or "").suffix.lower()
    if stored in {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}:
        return stored
    head = (audio.get("bytes") or b"")[:12]
    if head.startswith(b"RIFF"):
        return ".wav"
    if head.startswith(b"fLaC"):
        return ".flac"
    if head.startswith(b"OggS"):
        return ".ogg"
    if head.startswith(b"ID3") or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return ".mp3"
    return ".bin"


def write_audio(root: Path, rel_stem: str, audio: dict[str, Any]) -> str:
    """Write a row's audio under ``root`` once and return its path relative to ``root``."""
    rel = rel_stem + audio_suffix(audio)
    path = root / rel
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        part.write_bytes(audio["bytes"])
        part.replace(path)
    return rel
