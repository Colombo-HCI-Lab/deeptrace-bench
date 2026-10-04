"""Fetching a few items of a dataset instead of all of it, for pipeline tests.

A sample is ``per_label`` real and ``per_label`` fake items, chosen from the dataset's file
list before anything is downloaded and written under the dataset's folder with their
original relative paths, so the dataset's own builder runs on the sample unchanged.

- **Hugging Face** datasets: the repo's file list (at the pinned revision), then one
  ``hf_hub_download`` per chosen file.
- **Zip archives at a URL**: the archive's central directory, read with HTTP range requests,
  then only the chosen members (see ``remote_zip``).
- **A local copy** (``--from``): the files under it, symlinked rather than copied.

Labels come from the builder: ``label_from_path`` where the path says it, or, for datasets that
keep labels in a file (a ``meta.csv``, protocol files), ``METADATA_FILES`` plus
``labels_from_metadata``. Those files are read from the source first and copied into the
sample, so the builder sees them too. A dataset with only fake (or only real) items is
sampled on the labels its config says it contains.

Which items are chosen depends only on the file list and the seed: files are ordered by
``sha1("<seed>:<path>")``, never by Python's ``hash``, so the same sample comes back on any
machine. ``.sample.json`` records the spec, the source (with the archive's size and entry
count, since a sampled archive can't be checked against its full sha256) and the chosen
paths; asking for a different sample rebuilds the folder.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import shutil
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .fetch import ManualStepRequiredError, ensure_inside, gated_error, hf_patterns
from .registry import DatasetConfig

log = logging.getLogger(__name__)

LabelOf = Callable[[str], str | None]
SAMPLE_FILE = ".sample.json"


class SampleError(RuntimeError):
    """A dataset can't be sampled (no label function, or a label with no candidates)."""


def _builder_module(config: DatasetConfig) -> Any:
    if config.builder is None:
        raise SampleError(f"{config.id} has no builder, so it can't be sampled")
    return importlib.import_module(config.builder.partition(":")[0])


def metadata_files(config: DatasetConfig) -> list[str]:
    """Files the builder reads labels from (empty when labels come from paths)."""
    return list(getattr(_builder_module(config), "METADATA_FILES", []))


def label_function(config: DatasetConfig, read: Callable[[str], bytes] | None = None) -> LabelOf:
    """How the dataset's builder labels a path.

    Args:
        config: the dataset.
        read: reads one file of the source by relative path; needed when the builder takes
            labels from ``METADATA_FILES`` rather than from paths.

    Raises:
        SampleError: if the dataset has no builder, or its builder can't label items.
    """
    module = _builder_module(config)
    label_of = getattr(module, "label_from_path", None)
    if label_of is not None:
        return label_of
    names = getattr(module, "METADATA_FILES", None)
    from_metadata = getattr(module, "labels_from_metadata", None)
    if names and from_metadata and read is not None:
        labels = from_metadata({name: read(name) for name in names})
        return labels.get
    raise SampleError(
        f"{config.id}: its builder has neither label_from_path nor METADATA_FILES with "
        "labels_from_metadata to sample with"
    )


def _rank(seed: int, path: str) -> str:
    return hashlib.sha1(f"{seed}:{path}".encode()).hexdigest()


def choose(
    paths: Iterable[str],
    label_of: LabelOf,
    per_label: int,
    seed: int,
    labels: Iterable[str] = ("real", "fake"),
) -> list[str]:
    """Pick up to ``per_label`` paths of each wanted label, deterministically.

    Args:
        labels: the labels to sample; a fake-only dataset passes ``["fake"]``.

    Returns:
        The chosen paths, real first, each label in rank order.

    Raises:
        SampleError: if a wanted label has no candidates at all.
    """
    wanted = [label for label in ("real", "fake") if label in set(labels)]
    by_label: dict[str, list[str]] = {"real": [], "fake": []}
    for path in paths:
        label = label_of(path)
        if label is not None:
            by_label[label].append(path)
    empty = [label for label in wanted if not by_label[label]]
    if empty:
        raise SampleError(f"no {' or '.join(empty)} items among the candidates")
    chosen = []
    for label in wanted:
        chosen += sorted(by_label[label], key=lambda p: _rank(seed, p))[:per_label]
    return chosen


# --- sources --------------------------------------------------------------------------------


class _Source:
    """Where a sample comes from: lists candidate paths and materialises chosen ones."""

    def key(self) -> dict[str, Any]:
        """Identifies the source without touching the network (for the no-op check)."""
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        """``key`` plus whatever was learnt while listing (archive sizes, entry counts)."""
        return self.key()

    def candidates(self) -> list[str]:
        raise NotImplementedError

    def read(self, name: str) -> bytes:
        """The bytes of one file (a metadata file) without materialising it."""
        raise NotImplementedError

    def materialize(self, chosen: list[str], dest: Path) -> None:
        raise NotImplementedError


class _HfSource(_Source):
    def __init__(self, config: DatasetConfig, languages: list[str] | None) -> None:
        self.config = config
        self.patterns = hf_patterns(config, languages)

    def key(self) -> dict[str, Any]:
        a = self.config.access
        return {"kind": "hf", "repo": a.repo, "revision": a.revision, "patterns": self.patterns}

    def candidates(self) -> list[str]:
        import fnmatch

        from huggingface_hub import list_repo_files
        from huggingface_hub.errors import GatedRepoError

        access = self.config.access
        try:
            files = list_repo_files(access.repo, repo_type="dataset", revision=access.revision)
        except GatedRepoError as exc:
            raise gated_error(self.config) from exc
        if not self.patterns:
            return files
        return [f for f in files if any(fnmatch.fnmatch(f, p) for p in self.patterns)]

    def read(self, name: str) -> bytes:
        from huggingface_hub import hf_hub_download

        access = self.config.access
        path = hf_hub_download(access.repo, name, repo_type="dataset", revision=access.revision)
        return Path(path).read_bytes()

    def materialize(self, chosen: list[str], dest: Path) -> None:
        from huggingface_hub import hf_hub_download

        access = self.config.access
        for name in chosen:
            target = ensure_inside(dest, dest / name)
            target.parent.mkdir(parents=True, exist_ok=True)
            path = hf_hub_download(access.repo, name, repo_type="dataset", revision=access.revision)
            shutil.copyfile(path, target)


class _ZipSource(_Source):
    def __init__(self, config: DatasetConfig) -> None:
        urls = config.access.urls or {}
        self.urls = {name: url for name, url in urls.items() if name.lower().endswith(".zip")}
        if not self.urls or len(self.urls) != len(urls):
            raise SampleError(f"{config.id}: sampling over URLs needs every archive to be a zip")
        self.archives: dict[str, tuple[str, int, Any]] = {}
        self.where: dict[str, str] = {}

    def _open(self) -> None:
        from .remote_zip import open_remote, open_zip

        for name, url in self.urls.items():
            if name not in self.archives:
                raw = open_remote(url)
                self.archives[name] = (url, raw.size, open_zip(raw))

    def key(self) -> dict[str, Any]:
        return {"kind": "zip", "urls": dict(self.urls)}

    def describe(self) -> dict[str, Any]:
        self._open()
        return {
            **self.key(),
            "archives": {
                name: {"bytes": size, "entries": len(zf.infolist())}
                for name, (_, size, zf) in self.archives.items()
            },
        }

    def candidates(self) -> list[str]:
        self._open()
        paths = []
        for name, (_, _, zf) in self.archives.items():
            for info in zf.infolist():
                if not info.is_dir():
                    self.where[info.filename] = name
                    paths.append(info.filename)
        return paths

    def read(self, name: str) -> bytes:
        if not self.where:
            self.candidates()
        if name not in self.where:
            raise SampleError(f"{name} is in none of {sorted(self.urls)}")
        return self.archives[self.where[name]][2].read(name)

    def materialize(self, chosen: list[str], dest: Path) -> None:
        from .remote_zip import extract_member

        for member in chosen:
            zf = self.archives[self.where[member]][2]
            extract_member(zf, member, ensure_inside(dest, dest / member))


class _LocalSource(_Source):
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

    def key(self) -> dict[str, Any]:
        return {"kind": "local", "path": str(self.root)}

    def candidates(self) -> list[str]:
        return [p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()]

    def read(self, name: str) -> bytes:
        return ensure_inside(self.root, self.root / name).read_bytes()

    def materialize(self, chosen: list[str], dest: Path) -> None:
        for name in chosen:
            target = ensure_inside(dest, dest / name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(ensure_inside(self.root, self.root / name))


def _source(config: DatasetConfig, local: Path | None, languages: list[str] | None) -> _Source:
    if local is not None:
        return _LocalSource(local)
    match config.access.kind:
        case "hf" | "hf_gated":
            return _HfSource(config, languages)
        case "url":
            return _ZipSource(config)
        case _:
            raise ManualStepRequiredError(
                f"{config.id} can't be fetched by a script ({config.access.kind}); sample a "
                "copy you have with --from PATH"
            )


# --- entry point ----------------------------------------------------------------------------


def read_sample(dest: Path) -> dict[str, Any] | None:
    """The ``.sample.json`` of a sampled dataset folder, or None."""
    path = dest / SAMPLE_FILE
    return json.loads(path.read_text()) if path.exists() else None


def sample_dataset(
    config: DatasetConfig,
    dest: Path,
    per_label: int,
    seed: int = 0,
    local: Path | None = None,
    languages: list[str] | None = None,
) -> dict[str, Any]:
    """Fetch ``per_label`` real and fake items of a dataset into ``dest``.

    Re-running with the same spec is a no-op once every chosen file is present; a different
    spec (count, seed or source) clears ``dest`` and samples again.

    Returns:
        The sample record written to ``dest/.sample.json``.

    Raises:
        SampleError: if the dataset can't be labelled, or lacks candidates of a label it
            contains.
        ManualStepRequiredError: for datasets a script can't reach (request-only, gated).
    """
    if per_label < 1:
        raise ValueError("per_label must be at least 1")
    metadata = metadata_files(config)
    source = _source(config, local, languages)
    spec = {"per_label": per_label, "seed": seed, "source": source.key()}

    previous = read_sample(dest)
    kept = [*(previous or {}).get("chosen", []), *(previous or {}).get("metadata", [])]
    in_place = previous and all((dest / p).exists() for p in kept)
    if in_place and previous.get("spec") == spec:
        log.info("%s: sample already in place (%d items)", config.id, len(previous["chosen"]))
        return previous
    if dest.is_symlink():
        raise FileExistsError(f"{dest} links to a full copy; refusing to sample over it")
    if dest.exists():
        shutil.rmtree(dest)

    paths = source.candidates()
    label_of = label_function(config, read=source.read)
    chosen = choose(paths, label_of, per_label, seed, labels=config.contains)
    counts = {"real": 0, "fake": 0}
    for path in paths:
        label = label_of(path)
        if label:
            counts[label] += 1
    log.info(
        "%s: %d of %d real and %d of %d fake candidates",
        config.id,
        sum(label_of(p) == "real" for p in chosen),
        counts["real"],
        sum(label_of(p) == "fake" for p in chosen),
        counts["fake"],
    )
    dest.mkdir(parents=True, exist_ok=True)
    source.materialize([*chosen, *metadata], dest)
    record = {
        "dataset": config.id,
        "spec": spec,
        "source": source.describe(),
        "candidates": counts,
        "chosen": chosen,
        "metadata": metadata,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (dest / SAMPLE_FILE).write_text(json.dumps(record, indent=2) + "\n")
    return record
