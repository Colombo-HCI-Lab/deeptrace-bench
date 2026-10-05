"""The status page's data: which models were scored on which evalsets, and how far each got.

Reads aggregates only: the configs, parity summaries in ``results/parity/``, compute profiles
in ``results/profile/``, citation counts in ``results/citations.json``, the plain-language
catalog in ``docs/catalog.yaml``, and the store's run summaries, score files (for timing) and
manifests (for evalset sizes). Smoke runs and pipeline tests never count, by the same rule as
``--publish``. A run counts only while its model and evalset configs are unchanged, and the
latest such run wins when a pair has several.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

from . import evalset as evalsets
from .eval.contamination import Verdict, check
from .eval.publish import publish_refusal
from .paths import PUBLISHED_RESULTS_DIR, REPO_ROOT, run_dir, store_root
from .reference import model_input, recorded_features
from .registry import EvalsetConfig, ModelConfig, Registry, accepts
from .runs import config_hash

CATALOG = REPO_ROOT / "docs" / "catalog.yaml"
CITATIONS = PUBLISHED_RESULTS_DIR / "citations.json"
S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"

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
                **run_timing(summary["run_id"]),
            }
        )
    return rows


def run_timing(run_id: str) -> dict[str, Any]:
    """Wall time from the first session's start to the last score part, and inputs scored.

    Includes reading the data and preprocessing, so it measures the whole run, not the model.
    """
    directory = run_dir(run_id)
    parts = list(directory.glob("*.parquet"))
    if not parts:
        return {"wall_s": None, "windows": None}
    record = json.loads((directory / "run.json").read_text())
    start = min(datetime.fromisoformat(s["started_utc"]).timestamp() for s in record["sessions"])
    end = max(p.stat().st_mtime for p in parts)
    windows = sum(int(pd.read_parquet(p, columns=["n_windows"])["n_windows"].sum()) for p in parts)
    return {"wall_s": round(end - start), "windows": windows}


def evalset_size(evalset: EvalsetConfig, runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Items, and 4 s windows if a run has counted them, or None until the manifests exist.

    Every audio model cuts the same windows, so any run's count holds for all of them.
    """
    try:
        size: dict[str, Any] = {"items": len(evalsets.load_items(evalset))}
    except FileNotFoundError:
        return None
    windows = [r["windows"] for r in runs if r["evalset"] == evalset.id and r["windows"]]
    if windows:
        size["windows"] = windows[0]
    return size


def load_catalog(registry: Registry) -> dict[str, dict[str, Any]]:
    """The hand-written catalog: plain summaries, short venues and paper ids, checked by id."""
    catalog = yaml.safe_load(CATALOG.read_text()) if CATALOG.exists() else {}
    for kind, known in (("models", registry.models), ("datasets", registry.datasets)):
        unknown = set(catalog.get(kind) or {}) - set(known)
        if unknown:
            raise ValueError(f"{CATALOG.name}: unknown {kind} {sorted(unknown)}")
    return catalog


def fetch_citations(catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Look up every catalog paper on Semantic Scholar and write ``results/citations.json``."""
    ids = sorted(
        {
            e["s2_id"]
            for kind in ("models", "datasets")
            for e in (catalog.get(kind) or {}).values()
            if e.get("s2_id")
        }
    )
    # Without an API key requests share one rate-limited pool, so 429s are routine; back off.
    for attempt in range(8):
        response = requests.post(
            S2_BATCH,
            params={"fields": "title,year,citationCount"},
            json={"ids": ids},
            timeout=60,
        )
        if response.status_code != 429:
            break
        time.sleep(2 + 2**attempt)
    response.raise_for_status()
    papers = {
        s2_id: {"title": p["title"], "year": p["year"], "citations": p["citationCount"]}
        for s2_id, p in zip(ids, response.json(), strict=True)
        if p
    }
    data = {"source": "Semantic Scholar", "checked": datetime.now(UTC).date().isoformat()}
    data["papers"] = papers
    CITATIONS.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    return data


def paper_url(s2_id: str) -> str:
    """The paper itself where possible (arXiv, DOI), else its Semantic Scholar page."""
    kind, _, value = s2_id.partition(":")
    if kind == "ARXIV":
        return f"https://arxiv.org/abs/{value}"
    if kind == "DOI":
        return f"https://doi.org/{value}"
    return f"https://www.semanticscholar.org/paper/{s2_id}"


def _paper(entry: dict[str, Any], citations: dict[str, Any]) -> dict[str, Any] | None:
    s2_id = entry.get("s2_id")
    if not s2_id:
        return None
    found = citations.get("papers", {}).get(s2_id, {})
    return {
        "title": found.get("title"),
        "year": found.get("year"),
        "citations": found.get("citations"),
        "venue": entry.get("venue"),
        "url": paper_url(s2_id),
    }


def _display_names(models: list[ModelConfig]) -> dict[str, str]:
    """Names without their parenthetical, unless that would make two models look the same."""
    base = {m.id: m.name.split(" (")[0] for m in models}
    taken = list(base.values())
    return {m.id: base[m.id] if taken.count(base[m.id]) == 1 else m.name for m in models}


def _json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def _source_name(source: str, registry: Registry) -> str:
    entry = registry.datasets.get(source) or registry.corpora.get(source)
    return entry.name.split(" (")[0] if entry else source


def collect(registry: Registry) -> dict[str, Any]:
    """Everything the page shows, as one JSON-ready dict."""
    models = [m for m in registry.models.values() if m.status != "test"]
    evalset_list = [e for e in registry.evalsets.values() if e.role != "pipeline_test"]
    names = _display_names(models)
    catalog = load_catalog(registry)
    citations = _json(CITATIONS) or {}
    model_rows = []
    for m in models:
        parity = _json(PUBLISHED_RESULTS_DIR / "parity" / f"{m.id}.json")
        profile = _json(PUBLISHED_RESULTS_DIR / "profile" / f"{m.id}.json")
        entry = (catalog.get("models") or {}).get(m.id, {})
        model_rows.append(
            {
                "id": m.id,
                "name": names[m.id],
                "modality": m.modality.value,
                "adapter": m.adapter is not None,
                "blocked": m.status == "blocked",
                "licence": m.licence,
                "summary": entry.get("summary"),
                "architecture": entry.get("architecture"),
                "description": entry.get("description"),
                "input": model_input(m),
                "training": [
                    {"name": _source_name(t.dataset, registry), "level": t.level, "split": t.split}
                    for t in m.training_data
                ],
                "paper": _paper(entry, citations),
                "trained_on": sorted(
                    {
                        _source_name(t.dataset, registry)
                        for t in m.training_data
                        if t.level != "pretrain"
                    }
                ),
                "parity": (
                    {k: parity[k] for k in ("evalset", "items", "max_abs_diff")}
                    if parity and parity["passed"]
                    else None
                ),
                "profile": (
                    {
                        k: profile[k]
                        for k in (
                            "params",
                            "gflops_per_unit",
                            "ms_per_unit",
                            "ms_per_item",
                            "units_per_item",
                            "unit",
                            "peak_memory_gb",
                            "gpu",
                            "precision",
                        )
                    }
                    if profile
                    else None
                ),
            }
        )
    dataset_rows = []
    for d in registry.datasets.values():
        entry = (catalog.get("datasets") or {}).get(d.id, {})
        dataset_rows.append(
            {
                "id": d.id,
                "name": d.name,
                "modality": d.modality.value,
                "role": d.role,
                "status": d.status,
                "languages": d.languages,
                "licence": d.licence,
                "summary": entry.get("summary"),
                "scale": entry.get("scale"),
                "contains": d.contains,
                "generators": entry.get("generators"),
                "annotations": entry.get("annotations"),
                "format": entry.get("format"),
                "notes": entry.get("notes"),
                "recorded": recorded_features(d.id),
                "paper": _paper(entry, citations),
                "evalsets": sorted(
                    e.id for e in evalset_list if any(c.dataset == d.id for c in e.components)
                ),
            }
        )
    runs = scored_runs(registry)
    # Pairs a model can't read, and pairs the contamination guard refuses outright.
    cells = {}
    for m in models:
        for e in evalset_list:
            if not accepts(m, e):
                cells[f"{m.id}|{e.id}"] = "na"
            elif check(m, e, registry).verdict == Verdict.CONTAMINATED:
                cells[f"{m.id}|{e.id}"] = "guard"
    return {
        "built": datetime.now(UTC).date().isoformat(),
        "eval_status": registry.eval.status,
        "citations_checked": citations.get("checked"),
        "models": model_rows,
        "datasets": dataset_rows,
        "evalsets": [
            {
                "id": e.id,
                "modality": e.modality.value,
                "role": e.role,
                "pairing": e.pairing,
                "status": e.status,
                "size": evalset_size(e, runs),
            }
            for e in evalset_list
        ],
        "cells": cells,
        "runs": runs,
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
