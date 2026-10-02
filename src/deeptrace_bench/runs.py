"""Run records: enough provenance to know exactly what produced a score file.

A run is one model on one evalset under one evaluation setup. Its id is derived from the
hashes of the model, evalset and eval configs plus the weight hashes, so re-running the same
thing resumes the same run, and changing any of them starts a new one. ``run.json`` keeps
the inputs and a list of sessions (one per invocation, with the repo commit at the time).
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .paths import REPO_ROOT, run_dir


def config_hash(obj: BaseModel | dict[str, Any]) -> str:
    """Stable sha256 of a config, independent of key order."""
    data = obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def git_state(repo: Path = REPO_ROOT) -> tuple[str | None, bool]:
    """Return (HEAD commit, whether the worktree has uncommitted changes)."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None, True
    return commit, dirty


@dataclass
class RunRecord:
    """Provenance for one run."""

    run_id: str
    model_id: str
    evalset_id: str
    upstream_commit: str | None
    weights_sha256: dict[str, str]
    config_hashes: dict[str, str]
    preprocessing: str
    sessions: list[dict[str, Any]] = field(default_factory=list)

    def save(self, directory: Path) -> Path:
        """Write ``run.json`` into ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "run.json"
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        return path


def make_run_id(model_id: str, evalset_id: str, inputs: dict[str, Any]) -> str:
    """``<model>__<evalset>__<8 hex>``, the hex derived from everything that defines the run."""
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:8]
    return f"{model_id}__{evalset_id}__{digest}"


def start_run(
    model_id: str,
    evalset_id: str,
    upstream_commit: str | None,
    weights_sha256: dict[str, str],
    config_hashes: dict[str, str],
    preprocessing: str = "shared",
    directory: Path | None = None,
) -> tuple[RunRecord, Path]:
    """Create or resume a run and append a session to its record.

    Args:
        preprocessing: ``shared`` (the harness pipeline) or ``native`` (the model's own,
            for reproducing published numbers). Part of the run id, so the two never mix.
        directory: override the run directory (tests); defaults to ``DTB_ROOT/scores/<id>``.

    Returns:
        The record and its directory.
    """
    inputs = {
        "upstream_commit": upstream_commit,
        "weights": weights_sha256,
        "configs": config_hashes,
        "preprocessing": preprocessing,
    }
    run_id = make_run_id(model_id, evalset_id, inputs)
    directory = directory or run_dir(run_id)
    path = directory / "run.json"
    if path.exists():
        record = RunRecord(**json.loads(path.read_text()))
    else:
        record = RunRecord(
            run_id=run_id,
            model_id=model_id,
            evalset_id=evalset_id,
            upstream_commit=upstream_commit,
            weights_sha256=weights_sha256,
            config_hashes=config_hashes,
            preprocessing=preprocessing,
        )
    commit, dirty = git_state()
    record.sessions.append(
        {
            "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "repo_commit": commit,
            "repo_dirty": dirty,
            "host": socket.gethostname(),
            "python": platform.python_version(),
        }
    )
    record.save(directory)
    return record, directory
