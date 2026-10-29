"""Copy every evaluated run's per-item results and run record to an archive outside ``DTB_ROOT``.

Per-item scores are the one output that costs GPU time to rebuild, and they never enter git
(``data_policy.md``), so a store's copy is otherwise the only one. For every run with a
``results/<run_id>/`` under a source root this copies that folder (``items.parquet``,
``summary.json``, ``by_*.csv``) and the run's ``run.json`` and ``telemetry/`` to
``<archive>/runs/<run_id>/``, and each wave ledger under ``matrix/<wave>/`` to
``<archive>/waves/<wave>/``, then writes ``<archive>/index.csv``. Score parts stay behind:
``items.parquet`` holds the same scores, and the parts only matter for resuming.

A file is copied when the archive lacks it or holds an older one, so archiving several stores
keeps the newest copy of each file. Nothing is ever deleted from the archive.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from deeptrace_bench.registry import Registry

# Datasets whose terms rule out copies beyond their own store (data_policy.md, licence clauses).
# Anything derived from one of them stays too, and so does a run whose evalset isn't configured.
NO_COPY = frozenset({"dfplatter", "deephy", "indeepfake", "unidatapro_videos", "svarah"})

INDEX_COLUMNS = [
    "run_id",
    "model",
    "evalset",
    "pairing",
    "verdict",
    "parity_passed",
    "published",
    "n",
    "n_real",
    "n_fake",
    "auc",
    "eer",
    "tpr_at_fpr05",
    "sessions",
    "gpu",
]


@dataclass
class Report:
    """What one archive pass did."""

    runs: int = 0
    copied: int = 0
    skipped_licence: list[str] = field(default_factory=list)
    waves: int = 0


def _copy_newer(src: Path, dst: Path, report: Report) -> None:
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    report.copied += 1


def _copy_tree(src: Path, dst: Path, report: Report) -> None:
    for path in sorted(p for p in src.rglob("*") if p.is_file()):
        _copy_newer(path, dst / path.relative_to(src), report)


def _lineage(registry: Registry, dataset_id: str) -> set[str]:
    seen, todo = set(), [dataset_id]
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        if current in registry.datasets:
            todo.extend(registry.datasets[current].derived_from)
    return seen


def _restricted(registry: Registry, evalset_id: str) -> bool:
    """Whether a run on ``evalset_id`` must stay in its store: unknown evalsets count as yes."""
    try:
        evalset = registry.evalset(evalset_id)
    except KeyError:
        return True
    return any(_lineage(registry, c.dataset) & NO_COPY for c in evalset.components)


def archive(sources: list[Path], dest: Path, registry: Registry, published: set[str]) -> Report:
    """Archive the runs and wave ledgers of every store in ``sources`` into ``dest``."""
    report = Report()
    for root in sources:
        for results in sorted(
            p for p in (root / "results").glob("*") if (p / "summary.json").is_file()
        ):
            run_id = results.name
            evalset = json.loads((results / "summary.json").read_text()).get("evalset", "")
            if _restricted(registry, evalset):
                report.skipped_licence.append(run_id)
                continue
            target = dest / "runs" / run_id
            _copy_tree(results, target, report)
            scores = root / "scores" / run_id
            if (scores / "run.json").is_file():
                _copy_newer(scores / "run.json", target / "run.json", report)
            if (scores / "telemetry").is_dir():
                _copy_tree(scores / "telemetry", target / "telemetry", report)
            report.runs += 1
        for ledger in sorted((root / "matrix").glob("*/matrix.json")):
            _copy_tree(ledger.parent, dest / "waves" / ledger.parent.name, report)
            report.waves += 1
    write_index(dest, published)
    return report


def index_row(run: Path, published: set[str]) -> dict:
    """One ``index.csv`` row from an archived run's ``summary.json``."""
    summary = json.loads((run / "summary.json").read_text())
    overall = summary.get("overall") or {}
    compute = summary.get("compute") or {}
    return {
        "run_id": run.name,
        "model": summary.get("model"),
        "evalset": summary.get("evalset"),
        "pairing": summary.get("pairing"),
        "verdict": summary.get("verdict"),
        "parity_passed": summary.get("parity_passed"),
        "published": run.name in published,
        **{k: overall.get(k) for k in ("n", "n_real", "n_fake", "auc", "eer", "tpr_at_fpr05")},
        "sessions": compute.get("sessions"),
        "gpu": (compute.get("env") or {}).get("gpu"),
    }


def write_index(dest: Path, published: set[str]) -> None:
    """Rewrite ``index.csv`` from every run in the archive."""
    runs = sorted(p for p in (dest / "runs").glob("*") if (p / "summary.json").is_file())
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / "index.csv.tmp"
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(index_row(run, published) for run in runs)
    os.replace(tmp, dest / "index.csv")
