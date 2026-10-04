"""The smoke namespace moves item-level data and nothing else."""

from __future__ import annotations

import pytest

from deeptrace_bench import paths


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("DTB_ROOT", str(tmp_path))
    monkeypatch.delenv("DTB_NAMESPACE", raising=False)
    return tmp_path


def test_main_store_is_the_root(root):
    assert paths.dataset_dir("d") == root / "datasets" / "d"
    assert paths.run_dir("r") == root / "scores" / "r"


def test_use_namespace_sets_and_clears(root):
    paths.use_namespace("smoke")
    try:
        assert paths.store_root() == root / "smoke"
    finally:
        paths.use_namespace(None)
    assert paths.store_root() == root


def test_smoke_moves_item_level_data_only(root, monkeypatch):
    monkeypatch.setenv("DTB_NAMESPACE", "smoke")
    smoke = root / "smoke"
    assert paths.dataset_dir("d") == smoke / "datasets" / "d"
    assert paths.manifest_path("d") == smoke / "manifests" / "d.parquet"
    assert paths.faces_dir("d", "k") == smoke / "faces" / "d" / "k"
    assert paths.run_dir("r") == smoke / "scores" / "r"
    assert paths.results_dir("r") == smoke / "results" / "r"
    assert paths.reports_dir() == smoke / "reports"
    # Weights and upstream archives are shared, so a smoke run never re-downloads a model.
    assert paths.weights_dir("m") == root / "weights" / "m"
    assert paths.upstream_archive_dir() == root / "upstream"


def test_namespace_leaves_a_readme(root, monkeypatch):
    monkeypatch.setenv("DTB_NAMESPACE", "smoke")
    paths.prepare_store()
    assert "never published" in (root / "smoke" / "README.txt").read_text()


def test_unknown_namespace_is_refused(root, monkeypatch):
    monkeypatch.setenv("DTB_NAMESPACE", "scratch")
    with pytest.raises(ValueError, match="not one of"):
        paths.store_root()
    with pytest.raises(ValueError, match="unknown namespace"):
        paths.use_namespace("scratch")
