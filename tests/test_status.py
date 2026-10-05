"""The status page counts only current, real runs, and flags what needs a caveat."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

import pandas as pd
import pytest

from deeptrace_bench import status
from deeptrace_bench.runs import config_hash
from deeptrace_bench.status import (
    collect,
    fetch_citations,
    load_catalog,
    run_flags,
    run_timing,
    scored_runs,
    write_page,
)


def _summary(model: str, evalset: str, run_id: str, auc: float, verdict: str = "clean") -> dict:
    overall = {k: 0 for k in ("n", "n_real", "n_fake", "n_failed")}
    overall.update(auc=auc, auc_lo=auc, auc_hi=auc, eer=0.1, eer_lo=0.1, eer_hi=0.1)
    return {
        "run_id": run_id,
        "model": model,
        "evalset": evalset,
        "namespace": None,
        "verdict": verdict,
        "verdict_reasons": [],
        "overall": overall,
        "unscored_items": 0,
    }


def _store_run(root, registry, run_id, started, stale=False, **kw):
    model, evalset = "aasist", "urdu_csalt"
    hashes = {
        "model": "0" * 64 if stale else config_hash(registry.model(model)),
        "evalset": config_hash(registry.evalset(evalset)),
        "eval": "whatever",
    }
    (root / "scores" / run_id).mkdir(parents=True)
    (root / "scores" / run_id / "run.json").write_text(
        json.dumps({"config_hashes": hashes, "sessions": [{"started_utc": started}]})
    )
    (root / "results" / run_id).mkdir(parents=True)
    summary = _summary(model, evalset, run_id, **kw)
    (root / "results" / run_id / "summary.json").write_text(json.dumps(summary))


def test_latest_current_run_wins(registry, tmp_path, monkeypatch):
    monkeypatch.setenv("DTB_ROOT", str(tmp_path))
    monkeypatch.delenv("DTB_NAMESPACE", raising=False)
    _store_run(tmp_path, registry, "aasist__urdu_csalt__old", "2026-10-01T00:00:00", auc=0.7)
    _store_run(tmp_path, registry, "aasist__urdu_csalt__new", "2026-10-02T00:00:00", auc=0.8)
    _store_run(tmp_path, registry, "aasist__urdu_csalt__x", "2026-10-03", auc=0.9, stale=True)
    [row] = scored_runs(registry)
    assert row["run_id"] == "aasist__urdu_csalt__new"


def test_flags(registry):
    summary = _summary("df_arena_1b", "banglafake", "x", auc=0.4, verdict="source_overlap")
    assert run_flags(summary, registry.evalset("banglafake")) == ["source overlap", "below chance"]
    clean = _summary("aasist", "mlaad_hi", "x", auc=0.9)
    assert run_flags(clean, registry.evalset("mlaad_hi")) == ["cross-corpus"]


def test_page_shows_no_pipeline_tests_and_marks_the_guard(registry, tmp_path, monkeypatch):
    monkeypatch.setenv("DTB_ROOT", str(tmp_path))
    data = collect(registry)
    assert all(e["role"] != "pipeline_test" for e in data["evalsets"])
    assert data["cells"]["aasist3|mlaad_hi"] == "guard"
    assert data["cells"]["aasist|hidf_video"] == "na"


def test_write_page_replaces_only_the_data(tmp_path):
    page = tmp_path / "index.html"
    block = '<script id="status-data" type="application/json">{}</script>'
    page.write_text(f"<p>keep</p>{block}<b>x</b>")
    write_page(page, {"a": "</script>"})
    text = page.read_text()
    assert text.startswith("<p>keep</p>") and text.endswith("</script><b>x</b>")
    assert '"<\\/script>"' in text


def test_run_timing_spans_first_start_to_last_part(tmp_path, monkeypatch):
    monkeypatch.setenv("DTB_ROOT", str(tmp_path))
    monkeypatch.delenv("DTB_NAMESPACE", raising=False)
    run = tmp_path / "scores" / "r"
    run.mkdir(parents=True)
    start = datetime(2026, 10, 5, tzinfo=UTC)
    (run / "run.json").write_text(json.dumps({"sessions": [{"started_utc": start.isoformat()}]}))
    for i, windows in enumerate([[1, 2], [3]]):
        part = run / f"part-000of001-0000{i}.parquet"
        pd.DataFrame({"n_windows": windows}).to_parquet(part)
        os.utime(part, (start.timestamp() + 60 * (i + 1),) * 2)
    assert run_timing("r") == {"wall_s": 120, "windows": 6}


def test_catalog_ids_must_exist(registry, tmp_path, monkeypatch):
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text("models:\n  not_a_model: {summary: x}\n")
    monkeypatch.setattr(status, "CATALOG", catalog)
    with pytest.raises(ValueError, match="not_a_model"):
        load_catalog(registry)


def test_citations_are_keyed_by_catalog_id(tmp_path, monkeypatch):
    found = {"title": "T", "year": 2022, "citationCount": 7}

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return [found, None]

    monkeypatch.setattr(status.requests, "post", lambda *a, **k: Response())
    monkeypatch.setattr(status, "CITATIONS", tmp_path / "citations.json")
    catalog = {"models": {"a": {"s2_id": "ARXIV:1"}}, "datasets": {"b": {"s2_id": "DOI:2"}}}
    data = fetch_citations(catalog)
    assert data["papers"] == {"ARXIV:1": {"title": "T", "year": 2022, "citations": 7}}
