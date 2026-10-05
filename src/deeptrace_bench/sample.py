"""Fetching a few items of a dataset instead of all of it, for pipeline tests.

A sample is ``per_label`` real and ``per_label`` fake items, chosen from the dataset's file
list before anything is downloaded and written under the dataset's folder with their
original relative paths, so the dataset's own builder runs on the sample unchanged.

- **Hugging Face** datasets: the repo's file list (at the pinned revision), then one
  ``hf_hub_download`` per chosen file.
- **Zip archives at a URL**: the archive's central directory, read with HTTP range requests,
  then only the chosen members (see ``remote_zip``).
- **Parquet rows on Hugging Face**, for datasets that embed audio in parquet shards (their
  builders define ``label_from_row``, see ``datasets/_parquet.py``): the label columns of a
  few shards, read over HTTP without the audio, then only the row groups holding the chosen
  rows; each shard is written back with just those rows.
- **A local copy** (``--from``): the files under it, symlinked rather than copied.

Labels come from the builder: ``label_from_path`` where the path says it, or, for datasets that
keep labels in a file (a ``meta.csv``, protocol files), ``METADATA_FILES`` plus
``labels_from_metadata``. Those files are read from the source first and copied into the
sample, so the builder sees them too. A dataset with only fake (or only real) items is
sampled on the labels its config says it contains. A builder may also define
``sample_stratum(rel_path)``; then N items are taken from every stratum of each label (MAVOS-DD
samples per generator, so the rare voice-conversion fakes, the only ones with a known audio
label, are always in).

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
import re
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
    stratum_of: Callable[[str], str] | None = None,
) -> list[str]:
    """Pick up to ``per_label`` paths of each wanted label, deterministically.

    Args:
        labels: the labels to sample; a fake-only dataset passes ``["fake"]``.
        stratum_of: splits each label further (a builder's ``sample_stratum``); then
            ``per_label`` paths are taken from every stratum of every label, so a rare
            method still turns up in a small sample.

    Returns:
        The chosen paths, real first, each label (and stratum) in rank order.

    Raises:
        SampleError: if a wanted label has no candidates at all.
    """
    wanted = [label for label in ("real", "fake") if label in set(labels)]
    groups: dict[tuple[str, str], list[str]] = {}
    for path in paths:
        label = label_of(path)
        if label is not None:
            stratum = stratum_of(path) if stratum_of else ""
            groups.setdefault((label, stratum), []).append(path)
    empty = [label for label in wanted if not any(k[0] == label for k in groups)]
    if empty:
        raise SampleError(f"no {' or '.join(empty)} items among the candidates")
    chosen = []
    for key in sorted(k for k in groups if k[0] in wanted):
        chosen += sorted(groups[key], key=lambda p: _rank(seed, p))[:per_label]
    return sorted(chosen, key=lambda p: wanted.index(label_of(p)))


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

    def files(self, chosen: list[str]) -> list[str]:
        """The files that hold the chosen items, relative to the sample folder."""
        return list(chosen)


class _HfSource(_Source):
    def __init__(self, config: DatasetConfig, languages: list[str] | None) -> None:
        self.config = config
        self.patterns = hf_patterns(config, languages)

    def key(self) -> dict[str, Any]:
        a = self.config.access
        return {"kind": "hf", "repo": a.repo, "revision": a.revision, "patterns": self.patterns}

    def candidates(self) -> list[str]:
        import fnmatch

        from huggingface_hub.errors import GatedRepoError

        try:
            files = self._list()
        except GatedRepoError as exc:
            raise gated_error(self.config) from exc
        if not self.patterns:
            return files
        return [f for f in files if any(fnmatch.fnmatch(f, p) for p in self.patterns)]

    def _list(self) -> list[str]:
        """The repo's files, listing only the folders the patterns can match.

        MLAAD has well over 100k files; listing all of them to sample one language takes
        many minutes, while listing ``fake/si`` takes seconds. A pattern without wildcards
        names one file and needs no listing.
        """
        import huggingface_hub
        from huggingface_hub.hf_api import RepoFile

        access = self.config.access
        prefixes = [_literal_prefix(p) for p in self.patterns]
        if not self.patterns or "" in prefixes:
            return huggingface_hub.list_repo_files(
                access.repo, repo_type="dataset", revision=access.revision
            )
        files = [p for p in self.patterns if not _has_wildcard(p)]
        globs = zip(self.patterns, prefixes, strict=True)
        for folder in sorted({prefix for pattern, prefix in globs if _has_wildcard(pattern)}):
            tree = huggingface_hub.list_repo_tree(
                access.repo,
                path_in_repo=folder,
                recursive=True,
                revision=access.revision,
                repo_type="dataset",
            )
            files += [entry.path for entry in tree if isinstance(entry, RepoFile)]
        return files

    def _download(self, name: str) -> Path:
        import huggingface_hub
        from huggingface_hub.errors import GatedRepoError

        access = self.config.access
        try:
            path = huggingface_hub.hf_hub_download(
                access.repo, name, repo_type="dataset", revision=access.revision
            )
        except GatedRepoError as exc:
            raise gated_error(self.config) from exc
        return Path(path)

    def read(self, name: str) -> bytes:
        return self._download(name).read_bytes()

    def materialize(self, chosen: list[str], dest: Path) -> None:
        for name in chosen:
            target = ensure_inside(dest, dest / name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self._download(name), target)


def _has_wildcard(pattern: str) -> bool:
    return any(c in pattern for c in "*?[")


def _literal_prefix(pattern: str) -> str:
    """The folder a glob pattern stays inside (``fake/si/*`` -> ``fake/si``)."""
    if not _has_wildcard(pattern):
        return pattern.rpartition("/")[0] if "/" in pattern else pattern
    parts = []
    for part in pattern.split("/"):
        if _has_wildcard(part):
            break
        parts.append(part)
    return "/".join(parts)


class _HfParquetSource(_HfSource):
    """Rows of a dataset's parquet shards on the Hub, sampled without fetching every shard.

    Candidates come from at most ``max_files`` shards per folder and one row group of each
    (both picked by the seed), so a sample costs a few small label reads and one row group of
    audio per shard. A candidate is ``<shard path>#<row number in the shard>``.
    """

    def __init__(
        self,
        config: DatasetConfig,
        languages: list[str] | None,
        builder: Any,
        seed: int,
        max_files: int = 2,
        open_file: Callable[[str], Any] | None = None,
    ) -> None:
        super().__init__(config, languages)
        self.builder = builder
        self.seed = seed
        self.max_files = max_files
        self.open_file = open_file or self._open_on_hub
        self.labels: dict[str, str] = {}
        self.strata: dict[str, str] = {}

    def key(self) -> dict[str, Any]:
        return {**super().key(), "kind": "hf_parquet_rows", "max_files": self.max_files}

    def _open_on_hub(self, name: str) -> Any:
        from huggingface_hub import HfFileSystem

        access = self.config.access
        revision = f"@{access.revision}" if access.revision else ""
        return HfFileSystem().open(f"datasets/{access.repo}{revision}/{name}", "rb")

    def candidates(self) -> list[str]:
        import pyarrow.parquet as pq

        shards = [f for f in super().candidates() if f.endswith(".parquet")]
        # up to max_files shards per folder (IndicSynth keeps one per language), each folder
        # its own stratum, so every language asked for turns up
        folders: dict[str, list[str]] = {}
        for name in shards:
            folders.setdefault(name.rpartition("/")[0], []).append(name)
        shards = [
            name
            for group in folders.values()
            for name in sorted(group, key=lambda p: _rank(self.seed, p))[: self.max_files]
        ]
        stratum_of = getattr(self.builder, "row_stratum", None)
        paths = []
        for name in shards:
            folder = name.rpartition("/")[0]
            with self.open_file(name) as handle:
                shard = pq.ParquetFile(handle)
                sizes = [shard.metadata.row_group(g).num_rows for g in range(shard.num_row_groups)]
                group = min(range(len(sizes)), key=lambda g: _rank(self.seed, f"{name}#{g}"))
                table = shard.read_row_group(group, columns=list(self.builder.ROW_COLUMNS))
            for offset, row in enumerate(table.to_pylist()):
                label = self.builder.label_from_row(row)
                if label is None:
                    continue
                path = f"{name}#{sum(sizes[:group]) + offset}"
                self.labels[path] = label
                inner = stratum_of(row) if stratum_of is not None else ""
                self.strata[path] = f"{folder}/{inner}" if len(folders) > 1 else inner
                paths.append(path)
        return paths

    def files(self, chosen: list[str]) -> list[str]:
        return sorted({path.rpartition("#")[0] for path in chosen})

    def materialize(self, chosen: list[str], dest: Path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        from .datasets._parquet import SOURCE_ROW

        rows: dict[str, list[int]] = {}
        for path in chosen:
            name, _, row = path.rpartition("#")
            rows.setdefault(name, []).append(int(row))
        for name, wanted in rows.items():
            with self.open_file(name) as handle:
                shard = pq.ParquetFile(handle)
                starts = [0]
                for g in range(shard.num_row_groups):
                    starts.append(starts[-1] + shard.metadata.row_group(g).num_rows)
                groups = sorted(
                    {next(g for g in range(len(starts) - 1) if starts[g + 1] > r) for r in wanted}
                )
                pieces = []
                for g in groups:
                    table = shard.read_row_group(g)
                    take = [r - starts[g] for r in sorted(wanted) if starts[g] <= r < starts[g + 1]]
                    picked = table.take(take)
                    source = pa.array([starts[g] + t for t in take], type=pa.int64())
                    pieces.append(picked.append_column(SOURCE_ROW, source))
            target = ensure_inside(dest, dest / name)
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.concat_tables(pieces), target)


class _ZipSource(_Source):
    """Zip archives at URLs, read by range requests; other URLs are small metadata files.

    A builder whose archives hold further zips (BD-GRF6) defines
    ``nested_zip_label(member) -> "real" | "fake" | None``. Inner zips are compressed inside
    their archive, so they can't be read in place: for each label the smallest inner zip is
    fetched whole (its compressed bytes only, by range) into the cache, and its members join
    the candidates as ``<inner zip path without .zip>/<member>``, the path a full copy's
    builder extracts them to.
    """

    def __init__(self, config: DatasetConfig, builder: Any = None) -> None:
        urls = config.access.urls or {}
        self.urls = {name: url for name, url in urls.items() if name.lower().endswith(".zip")}
        # a split archive: <stem>.z01, <stem>.z02, ... next to <stem>.zip, read as one
        self.split_parts = {
            name: sorted(
                (n for n in urls if re.fullmatch(re.escape(name[:-4]) + r"\.z\d\d", n, re.I)),
                key=str.lower,
            )
            for name in self.urls
        }
        parts = {n for group in self.split_parts.values() for n in group}
        self.plain = {n: u for n, u in urls.items() if n not in self.urls and n not in parts}
        self.all_urls = dict(urls)
        if not self.urls:
            raise SampleError(f"{config.id}: sampling over URLs needs at least one zip")
        self.nested_label = getattr(builder, "nested_zip_label", None)
        self.label_of = getattr(builder, "label_from_path", None)
        self.archives: dict[str, tuple[str, int, Any]] = {}
        self.where: dict[str, Any] = {}

    def _open(self) -> None:
        from .remote_zip import SplitZip, open_remote, open_zip

        for name, url in self.urls.items():
            if name in self.archives:
                continue
            raw = open_remote(url)
            if self.split_parts[name]:
                parts = [open_remote(self.all_urls[p]) for p in self.split_parts[name]]
                # list only what the builder can label: a split set can hold millions of names
                keep = (lambda n: self.label_of(n) is not None) if self.label_of else None
                log.info("reading the central directory of split archive %s", name)
                self.archives[name] = (url, raw.size, SplitZip([*parts, raw], keep=keep))
            else:
                self.archives[name] = (url, raw.size, open_zip(raw))

    def key(self) -> dict[str, Any]:
        return {"kind": "zip", "urls": dict(self.all_urls)}

    def describe(self) -> dict[str, Any]:
        self._open()
        return {
            **self.key(),
            "archives": {
                name: {"bytes": size, "entries": len(_names(zf))}
                for name, (_, size, zf) in self.archives.items()
            },
        }

    def candidates(self) -> list[str]:
        self._open()
        paths = []
        inner: dict[str, list[tuple[int, str, str, Any]]] = {}
        for url, _, zf in self.archives.values():
            if not hasattr(zf, "infolist"):  # a split archive: read by name only
                for name in zf.namelist():
                    self.where[name] = (zf, name)
                    paths.append(name)
                continue
            for info in zf.infolist():
                if info.is_dir():
                    continue
                label = self.nested_label(info.filename) if self.nested_label else None
                if label is not None:
                    inner.setdefault(label, []).append((info.compress_size, info.filename, url, zf))
                    continue
                self.where[info.filename] = zf
                paths.append(info.filename)
        for label in sorted(inner):
            _, member, url, zf = min(inner[label], key=lambda t: (t[0], t[1]))
            paths += self._open_inner(member, url, zf)
        return paths

    def _open_inner(self, member: str, url: str, outer: Any) -> list[str]:
        """Fetch one inner zip into the cache and list its members as candidates."""
        import zipfile

        from .paths import dtb_cache
        from .remote_zip import extract_member

        digest = hashlib.sha1(f"{url}:{member}".encode()).hexdigest()[:16]
        local = dtb_cache() / "sample_zips" / f"{digest}.zip"
        if not local.exists():
            log.info("fetching inner archive %s of %s", member, url)
            extract_member(outer, member, local)
        zf = zipfile.ZipFile(local)
        prefix = member.removesuffix(".zip").removesuffix(".ZIP") + "/"
        paths = []
        for info in zf.infolist():
            if not info.is_dir():
                path = prefix + info.filename
                self.where[path] = (zf, info.filename)
                paths.append(path)
        return paths

    def _member(self, path: str) -> tuple[Any, str]:
        found = self.where[path]
        return found if isinstance(found, tuple) else (found, path)

    def read(self, name: str) -> bytes:
        if name in self.plain:
            import requests

            response = requests.get(self.plain[name], timeout=60)
            response.raise_for_status()
            return response.content
        if not self.where:
            self.candidates()
        if name not in self.where:
            raise SampleError(f"{name} is in none of {sorted(self.urls)}")
        zf, member = self._member(name)
        return zf.read(member)

    def materialize(self, chosen: list[str], dest: Path) -> None:
        from .remote_zip import extract_member

        for path in chosen:
            target = ensure_inside(dest, dest / path)
            if path in self.plain:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(self.read(path))
                continue
            zf, member = self._member(path)
            if hasattr(zf, "infolist"):
                extract_member(zf, member, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                part = target.with_name(target.name + ".part")
                part.write_bytes(zf.read(member))
                part.replace(target)


def _names(zf: Any) -> list[str]:
    return [i.filename for i in zf.infolist()] if hasattr(zf, "infolist") else zf.namelist()


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


def _source(
    config: DatasetConfig, local: Path | None, languages: list[str] | None, seed: int = 0
) -> _Source:
    builder = _builder_module(config) if config.builder else None
    rows = builder is not None and hasattr(builder, "label_from_row")
    if local is not None:
        if rows:
            raise SampleError(
                f"{config.id} keeps its items in parquet rows; sample it from the Hub"
            )
        return _LocalSource(local)
    match config.access.kind:
        case "hf" | "hf_gated":
            if rows:
                return _HfParquetSource(config, languages, builder, seed)
            return _HfSource(config, languages)
        case "url":
            if any(name.lower().endswith(".zip") for name in config.access.urls or {}):
                return _ZipSource(config, builder)
            # Archives that can't be read in place (tar): sample the full copy if one exists.
            from .paths import dtb_root

            full = dtb_root() / "datasets" / config.id
            if full.is_dir() and any(full.iterdir()):
                log.info("%s: sampling the full copy at %s", config.id, full)
                return _LocalSource(full)
            raise ManualStepRequiredError(
                f"{config.id}'s archives can't be sampled remotely; download it in full first "
                f"(setup_datasets.py {config.id}), then sample"
            )
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
    stratum_of = getattr(_builder_module(config), "sample_stratum", None)
    source = _source(config, local, languages, seed)
    spec = {
        "per_label": per_label,
        "seed": seed,
        "source": source.key(),
        "stratified": stratum_of is not None,
    }

    previous = read_sample(dest)
    held = (previous or {}).get("files", (previous or {}).get("chosen", []))
    kept = [*held, *(previous or {}).get("metadata", [])]
    in_place = previous and all((dest / p).exists() for p in kept)
    if in_place and previous.get("spec") == spec:
        log.info("%s: sample already in place (%d items)", config.id, len(previous["chosen"]))
        return previous
    if dest.is_symlink():
        raise FileExistsError(f"{dest} links to a full copy; refusing to sample over it")
    if dest.exists():
        shutil.rmtree(dest)

    paths = source.candidates()
    if isinstance(source, _HfParquetSource):
        label_of: LabelOf = source.labels.get
        stratum_of = source.strata.get if any(source.strata.values()) else None
    else:
        label_of = label_function(config, read=source.read)
    chosen = choose(paths, label_of, per_label, seed, labels=config.contains, stratum_of=stratum_of)
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
        "files": source.files(chosen),
        "metadata": metadata,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (dest / SAMPLE_FILE).write_text(json.dumps(record, indent=2) + "\n")
    return record
