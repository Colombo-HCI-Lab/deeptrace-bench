from __future__ import annotations

import csv
import json
import os

from deeptrace_bench.archive import archive


def _run(root, run_id, evalset, auc=0.9, mtime=None):
    results = root / "results" / run_id
    results.mkdir(parents=True)
    summary = {
        "model": "aasist",
        "evalset": evalset,
        "overall": {"n": 4, "auc": auc},
        "compute": {"sessions": 1},
    }
    (results / "summary.json").write_text(json.dumps(summary))
    (results / "items.parquet").write_bytes(b"synthetic")
    scores = root / "scores" / run_id
    (scores / "telemetry").mkdir(parents=True)
    (scores / "run.json").write_text("{}")
    (scores / "telemetry" / "part-0.json").write_text("{}")
    (scores / "part-000of001-00000.parquet").write_bytes(b"part")
    if mtime is not None:
        for p in [*results.iterdir(), scores / "run.json"]:
            os.utime(p, (mtime, mtime))


def test_archives_runs_waves_and_index(registry, tmp_path):
    store, dest = tmp_path / "store", tmp_path / "archive"
    _run(store, "aasist__urdu_csalt__aaaa0000", "urdu_csalt")
    (store / "matrix" / "a1").mkdir(parents=True)
    (store / "matrix" / "a1" / "matrix.json").write_text("{}")

    report = archive([store], dest, registry, published={"aasist__urdu_csalt__aaaa0000"})

    run = dest / "runs" / "aasist__urdu_csalt__aaaa0000"
    assert sorted(p.name for p in run.iterdir()) == [
        "items.parquet",
        "run.json",
        "summary.json",
        "telemetry",
    ]
    assert (dest / "waves" / "a1" / "matrix.json").is_file()
    rows = list(csv.DictReader((dest / "index.csv").open()))
    assert rows[0]["run_id"] == "aasist__urdu_csalt__aaaa0000" and rows[0]["published"] == "True"
    assert (report.runs, report.waves) == (1, 1)
    assert archive([store], dest, registry, published=set()).copied == 0


def test_leaves_out_restricted_datasets(registry, tmp_path):
    store, dest = tmp_path / "store", tmp_path / "archive"
    _run(store, "aasist__indeepfake__bbbb0000", "indeepfake")
    report = archive([store], dest, registry, published=set())
    assert report.skipped_licence == ["aasist__indeepfake__bbbb0000"]
    assert not (dest / "runs").exists()


def test_newest_copy_wins_across_stores(registry, tmp_path):
    old, new, dest = tmp_path / "old", tmp_path / "new", tmp_path / "archive"
    _run(old, "aasist__urdu_csalt__cccc0000", "urdu_csalt", auc=0.5, mtime=1_000)
    _run(new, "aasist__urdu_csalt__cccc0000", "urdu_csalt", auc=0.8, mtime=2_000)
    archive([new, old], dest, registry, published=set())
    rows = list(csv.DictReader((dest / "index.csv").open()))
    assert rows[0]["auc"] == "0.8"
