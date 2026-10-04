"""Checking out each detector's upstream repo at its pinned commit.

Upstream code is cloned into ``third_party/<repo>-<commit12>/`` (git-ignored), never copied
into this repo, because several upstream licences forbid redistribution (DeepfakeBench is
CC BY-NC, SBI is research-only, Effort has no licence at all). Adapters import the one model
file they need from the checkout.

Each checkout is also tarred to ``DTB_ROOT/upstream/`` so a deleted or rewritten upstream
repo can't stop us reproducing a run. Compatibility patches for running on current PyTorch
live in ``patches/<repo>/*.patch`` and are applied in name order after checkout.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tarfile
from pathlib import Path

from .paths import PATCHES_DIR, THIRD_PARTY_DIR, upstream_archive_dir
from .registry import Upstream

log = logging.getLogger(__name__)


def checkout_dir(upstream: Upstream) -> Path:
    """Where the pinned checkout lives."""
    return THIRD_PARTY_DIR / upstream.slug


def ensure_upstream(upstream: Upstream, archive: bool = True) -> Path:
    """Clone the repo at its pinned commit if needed, apply patches, and archive it.

    Args:
        upstream: the pinned repo.
        archive: also write ``DTB_ROOT/upstream/<slug>.tar.gz`` if it doesn't exist.

    Returns:
        The checkout directory.

    Raises:
        subprocess.CalledProcessError: if git fails (bad commit, network).
    """
    target = checkout_dir(upstream)
    if not (target / ".git").exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        log.info("cloning %s at %s", upstream.url, upstream.commit[:12])
        # Hugging Face repos keep their weights in LFS next to the code; the checkout only
        # needs the code (weights are fetched and hash-checked separately), so LFS objects
        # stay as pointer files.
        env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
        _git("clone", "--filter=blob:none", "--no-checkout", upstream.url, str(target), env=env)
        _git("-C", str(target), "checkout", "--detach", upstream.commit, env=env)
        _apply_patches(upstream, target)
    else:
        head = _git("-C", str(target), "rev-parse", "HEAD").strip()
        if head != upstream.commit:
            raise RuntimeError(
                f"{target} is at {head[:12]}, not the pinned {upstream.commit[:12]}; delete it "
                "and re-run"
            )
    if archive:
        _archive(upstream, target)
    return target


def _apply_patches(upstream: Upstream, target: Path) -> None:
    patch_dir = PATCHES_DIR / upstream.repo.split("/")[1]
    for patch in sorted(patch_dir.glob("*.patch")):
        log.info("applying %s", patch.name)
        _git("-C", str(target), "apply", str(patch))


def _archive(upstream: Upstream, target: Path) -> None:
    out = upstream_archive_dir() / f"{upstream.slug}.tar.gz"
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    log.info("archiving %s -> %s", target.name, out)
    part = out.with_name(out.name + ".part")
    with tarfile.open(part, "w:gz") as tf:
        tf.add(target, arcname=target.name, filter=lambda ti: None if "/.git" in ti.name else ti)
    part.rename(out)


def _git(*args: str, env: dict[str, str] | None = None) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True, env=env
    ).stdout
