"""The status page's data: which models were scored on which evalsets, and how far each got.

Reads aggregates only: the configs, parity summaries in ``results/parity/`` and run summaries
in the store's ``results/``. Smoke runs and pipeline tests never count, by the same rule as
``--publish``. A run counts only while its model and evalset configs are unchanged, and the
latest such run wins when a pair has several.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .eval.contamination import Verdict, check
from .eval.publish import publish_refusal
from .paths import PUBLISHED_RESULTS_DIR, run_dir, store_root
from .registry import EvalsetConfig, ModelConfig, Registry, accepts
from .runs import config_hash

_METRICS = ("n", "n_real", "n_fake", "n_failed")
_METRICS += ("auc", "auc_lo", "auc_hi", "eer", "eer_lo", "eer_hi")


def run_flags(summary: dict[str, Any], evalset: EvalsetConfig) -> list[str]:
    """What a reader must know before quoting a run's numbers, worst first."""
    flags = []
    if summary["verdict"] != "clean":
        flags.append(summary["verdict"].replace("_", " "))
    if evalset.pairing == "cross_corpus":
        flags.append("cross-corpus")
    # Below chance on a reproduction set means a flipped score; elsewhere it is a finding.
    if summary["overall"]["auc"] < 0.5:
        flags.append("below chance")
    if summary.get("unscored_items"):
        flags.append(f"{summary['unscored_items']} unscored")
    return flags


def _is_current(record: dict[str, Any], model: ModelConfig, evalset: EvalsetConfig) -> bool:
    hashes = record["config_hashes"]
    return hashes["model"] == config_hash(model) and hashes["evalset"] == config_hash(evalset)


def scored_runs(registry: Registry) -> list[dict[str, Any]]:
    """The latest countable run per (model, evalset) pair, as page rows."""
    best: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for path in sorted((store_root() / "results").glob("*/summary.json")):
        summary = json.loads(path.read_text())
        record_path = run_dir(summary["run_id"]) / "run.json"
        model = registry.models.get(summary["model"])
        evalset = registry.evalsets.get(summary["evalset"])
        if model is None or evalset is None or not record_path.exists():
            continue
        record = json.loads(record_path.read_text())
        if publish_refusal(evalset, summary["namespace"]):
            continue
        if not _is_current(record, model, evalset):
            continue
        started = record["sessions"][-1]["started_utc"]
        key = (model.id, evalset.id)
        if key not in best or started > best[key][0]:
            best[key] = (started, summary)
    rows = []
    for (model_id, evalset_id), (_, summary) in sorted(best.items()):
        evalset = registry.evalset(evalset_id)
        rows.append(
            {
                "model": model_id,
                "evalset": evalset_id,
                "run_id": summary["run_id"],
                "pairing": evalset.pairing,
                **{k: summary["overall"][k] for k in _METRICS},
                "verdict": summary["verdict"],
                "reasons": summary["verdict_reasons"] if summary["verdict"] != "clean" else [],
                "flags": run_flags(summary, evalset),
            }
        )
    return rows


def _display_names(models: list[ModelConfig]) -> dict[str, str]:
    """Names without their parenthetical, unless that would make two models look the same."""
    base = {m.id: m.name.split(" (")[0] for m in models}
    taken = list(base.values())
    return {m.id: base[m.id] if taken.count(base[m.id]) == 1 else m.name for m in models}


def collect(registry: Registry) -> dict[str, Any]:
    """Everything the page shows, as one JSON-ready dict."""
    models = [m for m in registry.models.values() if m.status != "test"]
    evalsets = [e for e in registry.evalsets.values() if e.role != "pipeline_test"]
    names = _display_names(models)
    model_rows = []
    for m in models:
        parity_path = PUBLISHED_RESULTS_DIR / "parity" / f"{m.id}.json"
        parity = json.loads(parity_path.read_text()) if parity_path.exists() else None
        model_rows.append(
            {
                "id": m.id,
                "name": names[m.id],
                "modality": m.modality.value,
                "adapter": m.adapter is not None,
                "blocked": m.status == "blocked",
                "parity": (
                    {k: parity[k] for k in ("evalset", "items", "max_abs_diff")}
                    if parity and parity["passed"]
                    else None
                ),
            }
        )
    # Pairs a model can't read, and pairs the contamination guard refuses outright.
    cells = {}
    for m in models:
        for e in evalsets:
            if not accepts(m, e):
                cells[f"{m.id}|{e.id}"] = "na"
            elif check(m, e, registry).verdict == Verdict.CONTAMINATED:
                cells[f"{m.id}|{e.id}"] = "guard"
    return {
        "built": datetime.now(UTC).date().isoformat(),
        "eval_status": registry.eval.status,
        "models": model_rows,
        "evalsets": [
            {
                "id": e.id,
                "modality": e.modality.value,
                "role": e.role,
                "pairing": e.pairing,
                "status": e.status,
            }
            for e in evalsets
        ],
        "cells": cells,
        "runs": scored_runs(registry),
    }


def write_page(page: Path, data: dict[str, Any]) -> None:
    """Replace the JSON inside the page's ``status-data`` script tag."""
    start, end = '<script id="status-data" type="application/json">', "</script>"
    html = page.read_text()
    head, sep, rest = html.partition(start)
    if not sep or end not in rest:
        raise ValueError(f"{page} has no status-data block")
    blob = json.dumps(data, indent=1).replace("</", "<\\/")
    page.write_text(head + start + "\n" + blob + "\n" + end + rest.split(end, 1)[1])
