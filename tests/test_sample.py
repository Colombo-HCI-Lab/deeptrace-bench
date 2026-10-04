"""Samples are deterministic, balanced, and rebuilt only when the request changes."""

from __future__ import annotations

import json
import random

import pytest

from deeptrace_bench.sample import SampleError, choose, sample_dataset


def _label(path: str) -> str | None:
    if path.startswith("real/"):
        return "real"
    if path.startswith("fake/"):
        return "fake"
    return None


PATHS = [f"real/{i}.jpg" for i in range(20)] + [f"fake/{i}.jpg" for i in range(30)] + ["x.csv"]


def test_choice_ignores_the_order_of_the_listing():
    shuffled = PATHS[:]
    random.Random(1).shuffle(shuffled)
    assert choose(PATHS, _label, 3, seed=0) == choose(shuffled, _label, 3, seed=0)


def test_choice_takes_n_per_label_and_skips_non_items():
    chosen = choose(PATHS, _label, 3, seed=0)
    assert [_label(p) for p in chosen] == ["real"] * 3 + ["fake"] * 3


def test_the_seed_changes_the_choice():
    assert choose(PATHS, _label, 3, seed=0) != choose(PATHS, _label, 3, seed=1)


def test_a_small_label_gives_what_it_has():
    paths = ["real/0.jpg", *PATHS[20:]]
    assert choose(paths, _label, 3, seed=0).count("real/0.jpg") == 1


def test_a_missing_label_is_an_error():
    with pytest.raises(SampleError, match="no real"):
        choose(PATHS[20:], _label, 3, seed=0)


@pytest.fixture
def local_copy(tmp_path):
    """A tiny copy laid out like the Mendeley archive, with invented names."""
    root = tmp_path / "copy"
    for rel in [
        "deepfake_dataset/real/1.mp4_frame_1_face_0.jpg",
        "deepfake_dataset/real/2.mp4_frame_5_face_0.jpg",
        "deepfake_dataset/deepfake/PersonA_Vid-1_Tech-1_frame_2_face_0.jpg",
        "deepfake_dataset/deepfake/PersonB_Vid-2_Tech-2_frame_7_face_0.jpg",
        "__MACOSX/deepfake_dataset/real/._1.mp4_frame_1_face_0.jpg",
    ]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"jpg")
    return root


def test_a_local_copy_is_sampled_by_symlink(registry, local_copy, tmp_path):
    dest = tmp_path / "smoke" / "mendeley_roop_akool"
    config = registry.dataset("mendeley_roop_akool")
    record = sample_dataset(config, dest, per_label=1, seed=0, local=local_copy)
    assert len(record["chosen"]) == 2
    assert record["candidates"] == {"real": 2, "fake": 2}
    for rel in record["chosen"]:
        assert (dest / rel).is_symlink()
    assert json.loads((dest / ".sample.json").read_text())["spec"]["per_label"] == 1


def test_the_same_request_is_a_no_op_and_a_new_one_rebuilds(registry, local_copy, tmp_path):
    dest = tmp_path / "smoke" / "mendeley_roop_akool"
    config = registry.dataset("mendeley_roop_akool")
    first = sample_dataset(config, dest, per_label=1, seed=0, local=local_copy)
    again = sample_dataset(config, dest, per_label=1, seed=0, local=local_copy)
    assert again["created_utc"] == first["created_utc"]
    bigger = sample_dataset(config, dest, per_label=2, seed=0, local=local_copy)
    assert len(bigger["chosen"]) == 4
    assert sum(1 for p in dest.rglob("*.jpg")) == 4
