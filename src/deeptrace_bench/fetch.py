"""Downloading weights and datasets, and checking what was downloaded.

Weight hashes live in ``configs/weights.lock.yaml``: the first fetch records each file's
sha256 there (with ``--record``), and every later fetch must match it. Google Drive is the
only host for several checkpoints, so a recorded hash is what tells us a re-upload or a
quota page didn't slip in.

Every download lands in a staging folder first and is checked there (not an HTML page, hash
matches the lock) before it is moved to its final path, so a bad file never sits where an
adapter would load it. Dataset archives are checked against ``access.sha256`` the same way
before they are extracted, and archive members may not escape the dataset folder.

The lock is keyed by owner: a model id, or a tool id (the face detector), never both.
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

import requests
import yaml

from .paths import CONFIG_DIR
from .registry import DatasetConfig, WeightSpec

log = logging.getLogger(__name__)

LOCK_PATH = CONFIG_DIR / "weights.lock.yaml"
_CHUNK = 1 << 20


class ManualStepRequiredError(RuntimeError):
    """Raised when a person has to do something first (request access, accept terms)."""


class ChecksumMismatchError(RuntimeError):
    """Raised when a file doesn't match the hash recorded in the lock file."""


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


_HTML_MARKERS = (b"<!doctype html", b"<html", b"<head", b"<body")


def looks_like_html(path: Path) -> bool:
    """True if a file starts like an HTML page (a quota notice or login wall saved as data)."""
    with path.open("rb") as fh:
        head = fh.read(1024).lstrip().lower()
    return head.startswith(_HTML_MARKERS)


def ensure_inside(base: Path, path: Path) -> Path:
    """Return ``path`` if it resolves inside ``base``.

    Raises:
        ValueError: if it escapes ``base`` (via ``..``, an absolute path or a symlink).
    """
    resolved = path.resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError(f"{path} escapes {base}")
    return path


def download_url(url: str, dest: Path) -> Path:
    """Stream ``url`` to ``dest`` (via a ``.part`` file so a failed download leaves no file).

    Raises:
        requests.HTTPError: on a non-2xx response.
        ManualStepRequiredError: if the server returns an HTML page instead of a file, which
            is how Google Drive quota pages and login walls usually show up.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    log.info("downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=60, allow_redirects=True) as resp:
        resp.raise_for_status()
        if "text/html" in resp.headers.get("content-type", "") and not url.endswith(".html"):
            raise ManualStepRequiredError(f"{url} returned an HTML page, not a file")
        with part.open("wb") as fh:
            for chunk in resp.iter_content(_CHUNK):
                fh.write(chunk)
    part.rename(dest)
    return dest


# --- weights ------------------------------------------------------------------------------


def read_lock() -> dict[str, dict[str, str]]:
    """Return ``{owner_id: {file_name: sha256}}`` from the lock file (owners: models, tools)."""
    if not LOCK_PATH.exists():
        return {}
    return yaml.safe_load(LOCK_PATH.read_text()) or {}


def locked_hashes(owner_id: str) -> dict[str, str]:
    """One owner's recorded hashes (empty if none yet)."""
    return read_lock().get(owner_id, {})


def is_pinned(owner_id: str, spec: WeightSpec) -> bool:
    """True if the lock has the weight's hash (for a folder, at least one of its files)."""
    keys = locked_hashes(owner_id)
    return spec.name in keys or any(k.startswith(f"{spec.name}/") for k in keys)


def record_hash(model_id: str, name: str, digest: str) -> None:
    """Write one file's hash into the lock file, keeping it sorted."""
    lock = read_lock()
    lock.setdefault(model_id, {})[name] = digest
    ordered = {k: dict(sorted(v.items())) for k, v in sorted(lock.items())}
    header = "# sha256 of every weight file, recorded on first fetch. Edit only via --record.\n"
    LOCK_PATH.write_text(header + yaml.safe_dump(ordered, sort_keys=False))


def fetch_weight(model_id: str, spec: WeightSpec, dest_dir: Path, record: bool = False) -> Path:
    """Fetch one weight (file or Drive folder) into ``dest_dir`` and check it.

    A new download goes to ``dest_dir/.incoming/`` first; it is moved into place only after
    every file in it passes the checks. An existing file or folder is re-checked rather than
    re-downloaded.

    Args:
        model_id: owning model (or tool), the key in the lock file.
        spec: the weight to fetch.
        dest_dir: the model's weights directory.
        record: write hashes into the lock file where none is recorded yet.

    Returns:
        Path of the file (or folder, for ``gdrive_folder`` and ``hf_snapshot``).

    Raises:
        ManualStepRequiredError: for weights a person must fetch by hand, and for downloads
            that turn out to be HTML pages.
        ChecksumMismatchError: if a file differs from its recorded hash.
    """
    dest = ensure_inside(dest_dir, dest_dir / spec.name)
    if dest.exists():
        _verify_weight(model_id, spec, dest, record)
        return dest

    staging = ensure_inside(dest_dir, dest_dir / ".incoming" / spec.name)
    _remove(staging)
    staging.parent.mkdir(parents=True, exist_ok=True)
    try:
        _download_weight(spec, staging)
        _verify_weight(model_id, spec, staging, record)
    except BaseException:
        _remove(staging)
        raise
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, dest)
    return dest


def _verify_weight(model_id: str, spec: WeightSpec, path: Path, record: bool) -> None:
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ManualStepRequiredError(f"{model_id}/{spec.name}: download is empty")
    lock = read_lock().get(model_id, {})
    for file in files:
        key = spec.name if file == path else f"{spec.name}/{file.relative_to(path).as_posix()}"
        if looks_like_html(file):
            raise ManualStepRequiredError(
                f"{model_id}/{key} is an HTML page, not weights (quota or login wall); fetch "
                "it by hand"
            )
        digest = sha256_file(file)
        expected = lock.get(key)
        if expected is None:
            if record:
                record_hash(model_id, key, digest)
                log.info("%s/%s: recorded sha256 %s", model_id, key, digest)
            else:
                log.warning(
                    "%s/%s: no recorded sha256 (got %s); re-run with --record to pin it",
                    model_id,
                    key,
                    digest,
                )
        elif digest != expected:
            raise ChecksumMismatchError(f"{model_id}/{key}: sha256 {digest} != recorded {expected}")


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _download_weight(spec: WeightSpec, dest: Path) -> None:
    match spec.kind:
        case "github_release" | "github_raw" | "url":
            assert spec.url is not None
            if spec.member:
                _download_member(spec.url, spec.member, dest)
            else:
                download_url(spec.url, dest)
        case "hf":
            from huggingface_hub import hf_hub_download

            assert spec.repo is not None and spec.filename is not None
            path = hf_hub_download(spec.repo, spec.filename, revision=spec.revision)
            shutil.copyfile(path, dest)
        case "hf_snapshot":
            _download_hf_snapshot(spec, dest)
        case "gdrive":
            import gdown

            out = gdown.download(id=spec.drive_id, output=str(dest), quiet=False)
            if out is None or not dest.exists():
                raise ManualStepRequiredError(
                    f"Google Drive refused {spec.drive_id} (quota or permissions); fetch "
                    f"{spec.name} by hand"
                )
        case "gdrive_folder":
            import gdown

            files = gdown.download_folder(id=spec.drive_id, output=str(dest), quiet=False)
            if not files:
                raise ManualStepRequiredError(
                    f"Google Drive refused folder {spec.drive_id}; fetch {spec.name} by hand"
                )
        case "manual":
            raise ManualStepRequiredError(
                f"{spec.name}: {spec.instructions} Place it in the model's weights folder."
            )


def _download_member(url: str, member: str, dest: Path) -> None:
    """One file out of a remote zip: by range requests if the host allows, else in full."""
    from .remote_zip import RangeNotSupportedError, extract_member, open_remote_zip

    try:
        with open_remote_zip(url) as zf:
            log.info("reading %s out of %s by range requests", member, url)
            extract_member(zf, member, dest)
        return
    except RangeNotSupportedError:
        log.info("%s ignores range requests; downloading the whole archive", url)
    archive = dest.with_name(dest.name + ".archive")
    try:
        download_url(url, archive)
        with zipfile.ZipFile(archive) as zf:
            extract_member(zf, member, dest)
    finally:
        archive.unlink(missing_ok=True)


def _download_hf_snapshot(spec: WeightSpec, dest: Path) -> None:
    """The files of one Hugging Face repo at a pinned revision, copied into a folder.

    Copies rather than links out of the HF cache, so the folder is self-contained and is
    hashed file by file like any other folder weight.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    assert spec.repo is not None and spec.revision is not None
    files = list_repo_files(spec.repo, revision=spec.revision)
    wanted = [
        f
        for f in files
        if not spec.allow_patterns or any(fnmatch.fnmatch(f, p) for p in spec.allow_patterns)
    ]
    if not wanted:
        raise ManualStepRequiredError(f"{spec.repo}@{spec.revision}: no files match the patterns")
    dest.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        target = ensure_inside(dest, dest / name)
        target.parent.mkdir(parents=True, exist_ok=True)
        log.info("%s: %s", spec.repo, name)
        shutil.copyfile(hf_hub_download(spec.repo, name, revision=spec.revision), target)


# --- datasets -----------------------------------------------------------------------------


def fetch_dataset(config: DatasetConfig, dest: Path, languages: list[str] | None = None) -> Path:
    """Download a dataset that can be fetched without a person in the loop.

    Args:
        config: the dataset.
        dest: where its files go (``DTB_ROOT/datasets/<id>``).
        languages: for datasets with per-language patterns, which languages to fetch;
            defaults to the config's ``default_languages``.

    Raises:
        ManualStepRequiredError: for request-only, deferred and dead datasets, and for gated
            Hugging Face repos whose terms haven't been accepted.
    """
    access = config.access
    match access.kind:
        case "hf" | "hf_gated":
            return _fetch_hf_dataset(config, dest, languages)
        case "url":
            assert access.urls is not None
            for name, url in access.urls.items():
                archive = ensure_inside(dest, dest / name)
                if not archive.exists():
                    download_url(url, archive)
                _check_archive(config, name, archive)
                _extract(archive, dest)
            return dest
        case "manual":
            raise ManualStepRequiredError(
                f"{config.id} needs a request: {access.request} Once you have it, run "
                f"scripts/setup_datasets.py {config.id} --from <path>."
            )
        case _:
            raise ManualStepRequiredError(
                f"{config.id} is {config.status} ({access.kind}); see docs/datasets.md"
            )


def hf_patterns(config: DatasetConfig, languages: list[str] | None = None) -> list[str]:
    """The ``allow_patterns`` for a Hugging Face dataset, plus the chosen languages' patterns.

    Raises:
        ValueError: for a language the config has no patterns for.
    """
    access = config.access
    patterns = list(access.allow_patterns or [])
    if access.language_patterns:
        chosen = languages or access.default_languages or list(access.language_patterns)
        unknown = set(chosen) - set(access.language_patterns)
        if unknown:
            raise ValueError(f"{config.id}: no patterns for languages {sorted(unknown)}")
        for lang in chosen:
            patterns.extend(access.language_patterns[lang])
    return patterns


def gated_error(config: DatasetConfig) -> ManualStepRequiredError:
    """The error for a gated Hugging Face dataset whose terms haven't been accepted."""
    return ManualStepRequiredError(
        f"{config.id} is gated: accept its terms at {config.access.page} with your Hugging Face "
        "account, then set HF_TOKEN in .env and re-run."
    )


def _fetch_hf_dataset(config: DatasetConfig, dest: Path, languages: list[str] | None) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    access = config.access
    patterns = hf_patterns(config, languages)
    log.info("%s: snapshot of %s (%s)", config.id, access.repo, patterns or "everything")
    try:
        snapshot_download(
            repo_id=access.repo,
            repo_type="dataset",
            revision=access.revision,
            allow_patterns=patterns or None,
            local_dir=dest,
        )
    except GatedRepoError as exc:
        raise gated_error(config) from exc
    return dest


def _check_archive(config: DatasetConfig, name: str, archive: Path) -> None:
    """Refuse HTML pages and archives that don't match ``access.sha256``."""
    if looks_like_html(archive):
        archive.unlink()
        raise ManualStepRequiredError(f"{config.id}/{name} downloaded as an HTML page")
    expected = config.access.sha256.get(name)
    digest = sha256_file(archive)
    if expected is None:
        log.warning(
            "%s/%s: no sha256 in the config (got %s); add it under access.sha256",
            config.id,
            name,
            digest,
        )
    elif digest != expected:
        raise ChecksumMismatchError(f"{config.id}/{name}: sha256 {digest} != expected {expected}")


def _extract(archive: Path, dest: Path) -> None:
    marker = dest / f".extracted-{archive.name}"
    if marker.exists():
        return
    log.info("extracting %s", archive)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            # zipfile already strips ".." and absolute paths; check anyway, so a malformed
            # archive fails loudly instead of being silently rewritten.
            for member in zf.namelist():
                ensure_inside(dest, dest / member)
            zf.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            # The "data" filter refuses absolute paths, "..", and links pointing outside.
            tf.extractall(dest, filter="data")
    else:
        return
    marker.touch()


def verify_supplied(config: DatasetConfig, source: Path, dest: Path) -> Path:
    """Link a manually obtained copy into the store after checking its expected paths.

    The copy stays where it is; ``dest`` becomes a symlink to it, so a 500 GB dataset isn't
    duplicated.

    Raises:
        FileNotFoundError: if ``source`` or any path in ``access.expect`` is missing.
    """
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    missing = [p for p in config.access.expect if not ensure_inside(source, source / p).exists()]
    if missing:
        raise FileNotFoundError(f"{config.id}: {source} lacks expected paths {missing}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        if dest.resolve() == source:
            return dest
        raise FileExistsError(f"{dest} already exists and points elsewhere")
    dest.symlink_to(source, target_is_directory=True)
    log.info("%s: linked %s -> %s", config.id, dest, source)
    return dest
