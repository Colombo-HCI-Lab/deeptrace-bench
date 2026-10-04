"""Manifest builders: one module per dataset, each turning its layout into a manifest.

Every builder has the signature ``build_manifest(root: Path) -> pandas.DataFrame``, where
``root`` is the dataset's folder in the store, and returns rows in the schema of
``deeptrace_bench.manifest``. Builders are written against the real files once a dataset is
downloaded; until then each module documents the layout as far as it is known.

A builder that can be sampled (``setup_datasets.py --sample``) also defines
``label_from_path(rel_path: str) -> "real" | "fake" | None``: the label of a file from its
path alone, or None for anything that isn't an item (metadata, archive junk). The sampler
uses it on the remote file list before anything is downloaded, and ``build_manifest`` uses it
on the local files, so the two can't disagree. Builders must work on partial trees: skip
hidden files (a sample leaves ``.sample.json``) and never assume the full item count.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path


def iter_files(root: Path) -> Iterator[str]:
    """Relative POSIX paths of every file under ``root``, skipping hidden files and folders."""
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if path.is_file():
            yield rel.as_posix()
