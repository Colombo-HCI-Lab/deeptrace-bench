"""Audio segmentation, evalset assembly and resumable scoring with the dummy detector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deeptrace_bench.evalset import EvalsetError, load_items
from deeptrace_bench.models.base import load_detector
from deeptrace_bench.preprocess import PreprocessError
from deeptrace_bench.preprocess.audio import segment
from deeptrace_bench.registry import Component, EvalsetConfig
from deeptrace_bench.runs import start_run
from deeptrace_bench.score import read_scores, score_items, shard_items

from .conftest import make_manifest


def test_short_clip_is_tiled_to_one_window():
    windows = segment(np.arange(20_000, dtype=np.float32), segment_samples=64_600)
    assert windows.shape == (1, 64_600)
    assert windows[0, 20_000] == 0.0  # the tile restarts


def test_long_clip_covers_its_tail():
    windows = segment(np.ones(150_000, dtype=np.float32), segment_samples=64_600)
    assert windows.shape == (3, 64_600)


def test_too_short_is_refused():
    with pytest.raises(PreprocessError) as info:
        segment(np.ones(100, dtype=np.float32), min_samples=16_000)
    assert info.value.reason == "too_short"


def test_evalset_pairs_forced_labels():
    fake = make_manifest("fk", [{"local": "1", "label": "fake", "language": "hi"}])
    real = make_manifest("rl", [{"local": "1", "label": "real", "language": "hi"}])
    evalset = EvalsetConfig(
        id="pair",
        modality="audio",
        role="south_asian",
        pairing="cross_corpus",
        components=[
            Component(dataset="fk", filter={"language": "hi"}, label="fake"),
            Component(dataset="rl", label="real"),
        ],
        status="ready",
    )
    items = load_items(evalset, manifests={"fk": fake, "rl": real})
    assert set(items["label"]) == {"real", "fake"}
    assert (items["pairing"] == "cross_corpus").all()


def test_evalset_without_both_labels_fails():
    fake = make_manifest("fk", [{"local": "1", "label": "fake"}])
    evalset = EvalsetConfig(
        id="one",
        modality="audio",
        role="south_asian",
        pairing="cross_corpus",
        components=[Component(dataset="fk", label="fake")],
        status="needs_real_source",
    )
    with pytest.raises(EvalsetError, match="needs real and fake"):
        load_items(evalset, manifests={"fk": fake})


def test_scoring_resumes_and_records_failures(registry, tmp_path):
    detector = load_detector(registry.model("dummy"))
    detector.load("cpu")
    items = pd.DataFrame({"item_id": [f"toy/{i}" for i in range(10)]})

    def loader(row: pd.Series) -> np.ndarray:
        index = int(row["item_id"].split("/")[1])
        if index == 7:
            raise PreprocessError("too_short")
        return np.full((2, 4), index / 10, dtype=np.float32)

    # A first call that stops after 4 items stands in for a killed job.
    assert score_items(detector, items, loader, tmp_path, flush_every=3, max_items=4) == 4
    assert score_items(detector, items, loader, tmp_path, flush_every=3) == 6
    scores = read_scores(tmp_path)
    assert len(scores) == 10 and scores["item_id"].is_unique
    failed = scores.set_index("item_id").loc["toy/7"]
    assert failed["status"] == "too_short" and np.isnan(failed["score"])


def test_shards_partition_the_items():
    items = pd.DataFrame({"item_id": [f"toy/{i}" for i in range(100)]})
    shards = [shard_items(items, i, 4) for i in range(4)]
    assert sum(len(s) for s in shards) == 100
    assert set().union(*(set(s["item_id"]) for s in shards)) == set(items["item_id"])


def test_same_inputs_resume_the_same_run(tmp_path, monkeypatch):
    kwargs = dict(
        model_id="dummy",
        evalset_id="toy",
        upstream_commit=None,
        weights_sha256={},
        config_hashes={"model": "a", "evalset": "b", "eval": "c"},
    )
    first, _ = start_run(**kwargs, directory=tmp_path)
    second, _ = start_run(**kwargs, directory=tmp_path)
    assert first.run_id == second.run_id
    assert len(second.sessions) == 2
    other, _ = start_run(**{**kwargs, "preprocessing": "native"}, directory=tmp_path / "n")
    assert other.run_id != first.run_id
