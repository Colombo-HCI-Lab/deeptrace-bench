"""Where data, weights and caches live.

Everything large sits outside the repo, under two roots read from the environment or from a
``.env`` file at the repo root (see ``.env.example``):

- ``DTB_ROOT``: the shared store for datasets, weights, upstream tarballs, manifests, splits,
  scores and results. On Curnagl this is a folder in the project's ``/work`` space.
- ``DTB_CACHE``: fast scratch for preprocessed inputs (face crops, audio segments). Defaults to
  ``$TMPDIR/dtb-cache``, which inside a SLURM job is the node-local NVMe.

Manifests and per-item scores live under ``DTB_ROOT`` rather than in git because several
dataset licences treat item names and derived data as part of the dataset.

**Namespaces.** Pipeline tests (``scripts/smoke.py``, ``--sample``, ``--smoke``) set
``DTB_NAMESPACE=smoke``, which moves everything item-level (datasets, manifests, face caches,
scores, results, reports) under ``DTB_ROOT/smoke/``. Weights and upstream archives stay shared
at the root, so a smoke run never downloads a model twice and never mixes with real runs. The
full tree is in ``docs/store_layout.md``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
PATCHES_DIR = REPO_ROOT / "patches"
PUBLISHED_RESULTS_DIR = REPO_ROOT / "results"

load_dotenv(REPO_ROOT / ".env", override=False)


class RootNotConfiguredError(RuntimeError):
    """Raised when ``DTB_ROOT`` is needed but not set."""


def dtb_root() -> Path:
    """Return the shared store root.

    Raises:
        RootNotConfiguredError: if ``DTB_ROOT`` is unset.
    """
    value = os.environ.get("DTB_ROOT")
    if not value:
        raise RootNotConfiguredError(
            "DTB_ROOT is not set. Copy .env.example to .env and point DTB_ROOT at the shared "
            "store (see docs/cluster.md)."
        )
    return Path(value).expanduser()


def dtb_cache() -> Path:
    """Return the scratch root for preprocessed inputs (``DTB_CACHE``, else ``$TMPDIR``)."""
    if value := os.environ.get("DTB_CACHE"):
        return Path(value).expanduser()
    if tmp := os.environ.get("TMPDIR"):
        return Path(tmp) / "dtb-cache"
    return dtb_root() / "cache"


NAMESPACES = {"smoke"}

_NAMESPACE_README = """\
Pipeline tests written by scripts/smoke.py and the --sample / --smoke flags.

Same layout as the store root (datasets, manifests, faces, scores, results), plus reports/.
Nothing here is a result: it is never published or reported. Safe to delete at any time;
weights and upstream archives live one level up and are shared.
"""


def namespace() -> str | None:
    """The active namespace (``DTB_NAMESPACE``), or None for the main store.

    Raises:
        ValueError: if ``DTB_NAMESPACE`` names an unknown namespace.
    """
    value = os.environ.get("DTB_NAMESPACE") or None
    if value is not None and value not in NAMESPACES:
        raise ValueError(f"DTB_NAMESPACE={value!r} is not one of {sorted(NAMESPACES)}")
    return value


def use_namespace(name: str | None) -> None:
    """Switch the namespace for this process and any subprocess it starts."""
    if name is None:
        os.environ.pop("DTB_NAMESPACE", None)
        return
    if name not in NAMESPACES:
        raise ValueError(f"unknown namespace {name!r}; known: {sorted(NAMESPACES)}")
    os.environ["DTB_NAMESPACE"] = name


def store_root() -> Path:
    """Root of everything item-level: ``DTB_ROOT``, or ``DTB_ROOT/<namespace>``."""
    ns = namespace()
    return dtb_root() / ns if ns else dtb_root()


def prepare_store() -> Path:
    """Create the store root; in a namespace, also leave a README saying what it is."""
    root = store_root()
    root.mkdir(parents=True, exist_ok=True)
    readme = root / "README.txt"
    if namespace() and not readme.exists():
        readme.write_text(_NAMESPACE_README)
    return root


def dataset_dir(dataset_id: str) -> Path:
    """Directory holding one dataset's files as downloaded, sampled or supplied."""
    return store_root() / "datasets" / dataset_id


def weights_dir(owner_id: str) -> Path:
    """Directory holding one model's (or tool's) weight files. Shared by every namespace."""
    return dtb_root() / "weights" / owner_id


def upstream_archive_dir() -> Path:
    """Directory of tarballs of each pinned upstream checkout. Shared by every namespace."""
    return dtb_root() / "upstream"


def manifest_path(dataset_id: str) -> Path:
    """Parquet manifest for one dataset."""
    return store_root() / "manifests" / f"{dataset_id}.parquet"


def splits_dir() -> Path:
    """Directory of split assignments (item id to split) for fine-tuning."""
    return store_root() / "splits"


def faces_dir(dataset_id: str, key: str) -> Path:
    """Cached face detections for one dataset under one detector setup (``key``)."""
    return store_root() / "faces" / dataset_id / key


def scores_root() -> Path:
    """Directory holding every run's scores."""
    return store_root() / "scores"


def run_dir(run_id: str) -> Path:
    """Directory of one scoring run: ``run.json``, score parts and any saved crops."""
    return scores_root() / run_id


def results_dir(run_id: str) -> Path:
    """Directory of one run's full evaluation output, including per-item tables."""
    return store_root() / "results" / run_id


def reports_dir() -> Path:
    """Directory of human-readable run reports (smoke reports live here)."""
    return store_root() / "reports"
