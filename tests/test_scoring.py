"""Audio segmentation, evalset assembly and resumable scoring with the dummy detector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deeptrace_bench.evalset import EvalsetError, load_items
from deeptrace_bench.models.base import load_detector
from deeptrace_bench.preprocess import PreprocessError
from deeptrace_bench.preprocess.audio import segment
from deeptrace_bench.registry import Component, EvalsetConfig, Subset
from deeptrace_bench.runs import start_run
from deeptrace_bench.score import (
    part_prefix,
    progress_line,
    read_scores,
    score_items,
    shard_items,
    supersede_untimed,
)

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


def _subset_evalset(per_stratum: int, seed: int = 0) -> EvalsetConfig:
    return EvalsetConfig(
        id="sub",
        modality="video",
        role="south_asian",
        pairing="same_corpus",
        components=[Component(dataset="ds")],
        status="ready",
        subset=Subset(per_stratum=per_stratum, strata=["label", "method"], seed=seed),
    )


def _strata_manifest():
    rows = [
        {"local": f"{method}{i}", "label": label, "method": method}
        for label, method, n in [("real", "real", 30), ("fake", "swap", 20), ("fake", "rare", 3)]
        for i in range(n)
    ]
    return make_manifest("ds", rows)


def test_subset_caps_each_stratum_and_keeps_small_ones_whole():
    manifest = _strata_manifest()
    items = load_items(_subset_evalset(5), manifests={"ds": manifest})
    assert items.groupby("method").size().to_dict() == {"rare": 3, "real": 5, "swap": 5}
    again = load_items(
        _subset_evalset(5), manifests={"ds": manifest.sample(frac=1, random_state=1)}
    )
    assert set(again["item_id"]) == set(items["item_id"])
    bigger = load_items(_subset_evalset(8), manifests={"ds": manifest})
    assert set(items["item_id"]) <= set(bigger["item_id"])
    other = load_items(_subset_evalset(5, seed=1), manifests={"ds": manifest})
    assert set(other["item_id"]) != set(items["item_id"])


def test_subset_on_a_missing_column_fails():
    evalset = _subset_evalset(5).model_copy(
        update={"subset": Subset(per_stratum=5, strata=["nope"])}
    )
    with pytest.raises(EvalsetError, match="nope"):
        load_items(evalset, manifests={"ds": _strata_manifest()})


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
    # every item is timed; a failed item has a load time but no inference time
    assert scores["load_s"].notna().all() and np.isnan(failed["infer_s"])
    assert scores.loc[scores["status"] == "ok", "infer_s"].notna().all()


def test_parts_from_before_timing_still_resume(registry, tmp_path):
    detector = load_detector(registry.model("dummy"))
    detector.load("cpu")
    items = pd.DataFrame({"item_id": [f"toy/{i}" for i in range(4)]})
    old = pd.DataFrame(
        {"item_id": ["toy/0", "toy/1"], "score": 0.5, "n_windows": 1, "status": "ok"}
    )
    old.to_parquet(tmp_path / "part-000of001-00000.parquet", index=False)

    def loader(row: pd.Series) -> np.ndarray:
        return np.zeros((1, 4), dtype=np.float32)

    assert score_items(detector, items, loader, tmp_path, prefix=part_prefix(0, 1)) == 2
    scores = read_scores(tmp_path).set_index("item_id")
    assert np.isnan(scores.loc["toy/0", "load_s"]) and scores.loc["toy/3", "load_s"] >= 0


def test_untimed_parts_are_moved_aside_and_rescored(registry, tmp_path):
    detector = load_detector(registry.model("dummy"))
    detector.load("cpu")
    items = pd.DataFrame({"item_id": ["toy/0", "toy/1"]})
    old = pd.DataFrame({"item_id": ["toy/0"], "score": 0.5, "n_windows": 1, "status": "ok"})
    old.to_parquet(tmp_path / "part-000of001-00000.parquet", index=False)
    assert supersede_untimed(tmp_path) == 1 and supersede_untimed(tmp_path) == 0
    assert len(list((tmp_path / "superseded").glob("*/part-*.parquet"))) == 1

    def loader(row: pd.Series) -> np.ndarray:
        return np.zeros((1, 4), dtype=np.float32)

    assert score_items(detector, items, loader, tmp_path, prefix=part_prefix(0, 1)) == 2
    assert read_scores(tmp_path)["load_s"].notna().all()


def test_parts_without_any_telemetry_are_rescored(tmp_path):
    timed = pd.DataFrame(
        {"item_id": ["toy/0"], "score": 0.5, "n_windows": 1, "status": "ok", "load_s": 0.1}
    )
    timed.assign(infer_s=0.1).to_parquet(tmp_path / "part-000of001-00000.parquet", index=False)
    # timed parts but no session telemetry: a session crashed before writing it
    assert supersede_untimed(tmp_path) == 1
    (tmp_path / "telemetry").mkdir()
    (tmp_path / "telemetry" / "s.json").write_text("{}")
    timed.assign(infer_s=0.1).to_parquet(tmp_path / "part-000of001-00001.parquet", index=False)
    assert supersede_untimed(tmp_path) == 0


def test_progress_line_format():
    line = progress_line("aasist__urdu_csalt__1234abcd", 128, 512, 4.0, 3)
    run = "aasist__urdu_csalt__1234abcd"
    assert line == f"[progress] {run} 128/512 items 4.00 it/s eta 1m36s fail 3"


def test_shards_write_their_own_parts(registry, tmp_path):
    detector = load_detector(registry.model("dummy"))
    detector.load("cpu")
    items = pd.DataFrame({"item_id": [f"toy/{i}" for i in range(10)]})

    def loader(row: pd.Series) -> np.ndarray:
        return np.zeros((1, 4), dtype=np.float32)

    # Two shards starting together on an empty run folder must not overwrite each other.
    for index in range(2):
        shard = shard_items(items, index, 2)
        score_items(detector, shard, loader, tmp_path, prefix=part_prefix(index, 2))
    assert sorted(p.name for p in tmp_path.glob("part-*.parquet")) == [
        "part-000of002-00000.parquet",
        "part-001of002-00000.parquet",
    ]
    assert len(read_scores(tmp_path)) == 10


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


def _audio_video_manifest() -> pd.DataFrame:
    # A real clip, a voice-converted one (video untouched) and a face swap whose audio
    # nobody has checked.
    return make_manifest(
        "av",
        [
            {"local": "real", "label": "real", "label_video": "real", "label_audio": "real"},
            {"local": "vc", "label": "fake", "label_video": "real", "label_audio": "fake"},
            {"local": "swap", "label": "fake", "label_video": "fake", "label_audio": None},
        ],
    )


def _track_evalset(label: str, filter: dict | None = None) -> EvalsetConfig:
    return EvalsetConfig(
        id="av_track",
        modality="audio" if label == "from_audio" else "video",
        role="south_asian",
        pairing="same_corpus",
        components=[Component(dataset="av", filter=filter or {}, label=label)],
        status="ready",
    )


def test_a_track_label_scores_the_track_that_was_manipulated():
    audio = _track_evalset("from_audio", {"label_audio": ["real", "fake"]})
    items = load_items(audio, manifests={"av": _audio_video_manifest()})
    assert dict(zip(items["item_id"], items["label"], strict=True)) == {
        "av/real": "real",
        "av/vc": "fake",
    }
    video = load_items(_track_evalset("from_video"), manifests={"av": _audio_video_manifest()})
    assert dict(zip(video["item_id"], video["label"], strict=True))["av/vc"] == "real"


def test_a_track_label_must_be_known_for_every_item():
    with pytest.raises(EvalsetError, match="label_audio"):
        load_items(_track_evalset("from_audio"), manifests={"av": _audio_video_manifest()})


def test_tf32_follows_the_eval_config():
    import torch

    from deeptrace_bench.models.base import set_tf32
    from deeptrace_bench.registry import Registry

    before = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    try:
        set_tf32(True)
        assert torch.backends.cudnn.allow_tf32 and torch.backends.cuda.matmul.allow_tf32
        set_tf32(Registry.load().eval.compute["tf32"])
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = before
