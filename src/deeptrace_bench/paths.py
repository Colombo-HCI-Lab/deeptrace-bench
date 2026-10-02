"""Where data, weights and caches live.

Everything large sits outside the repo, under two roots read from the environment or from a
``.env`` file at the repo root (see ``.env.example``):

- ``DTB_ROOT``: the shared store for datasets, weights, upstream tarballs, manifests, splits,
  scores and results. On Curnagl this is a folder in the project's ``/work`` space.
- ``DTB_CACHE``: fast scratch for preprocessed inputs (face crops, audio segments). Defaults to
  ``$TMPDIR/dtb-cache``, which inside a SLURM job is the node-local NVMe.

Manifests and per-item scores live under ``DTB_ROOT`` rather than in git because several
dataset licences treat item names and derived data as part of the dataset.
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


def dataset_dir(dataset_id: str) -> Path:
    """Directory holding one dataset's files as downloaded or supplied."""
    return dtb_root() / "datasets" / dataset_id


def weights_dir(model_id: str) -> Path:
    """Directory holding one model's weight files."""
    return dtb_root() / "weights" / model_id


def upstream_archive_dir() -> Path:
    """Directory of tarballs of each pinned upstream checkout."""
    return dtb_root() / "upstream"


def manifest_path(dataset_id: str) -> Path:
    """Parquet manifest for one dataset."""
    return dtb_root() / "manifests" / f"{dataset_id}.parquet"


def splits_dir() -> Path:
    """Directory of split assignments (item id to split) for fine-tuning."""
    return dtb_root() / "splits"


def run_dir(run_id: str) -> Path:
    """Directory of one scoring run: ``run.json`` plus score shards."""
    return dtb_root() / "scores" / run_id


def results_dir(run_id: str) -> Path:
    """Directory of one run's full evaluation output, including per-item tables."""
    return dtb_root() / "results" / run_id
