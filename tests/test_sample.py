"""Samples are deterministic, balanced, and rebuilt only when the request changes."""

from __future__ import annotations

import json
import random
import sys
import types

import pytest

from deeptrace_bench.registry import DatasetConfig
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


def test_a_single_label_dataset_samples_only_that_label():
    fakes = [f"fake/{i}.jpg" for i in range(5)]
    assert len(choose(fakes, _label, 2, seed=0, labels=["fake"])) == 2
    with pytest.raises(SampleError, match="no fake"):
        choose(PATHS[:20], _label, 2, seed=0, labels=["fake"])


# --- builders registered for these tests only (invented layouts and names) -----------------


def _builder(monkeypatch, name: str, **attrs) -> str:
    module = types.ModuleType(name)
    module.build_manifest = lambda root: None
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return f"{name}:build_manifest"


def _config(builder: str, contains: list[str]) -> DatasetConfig:
    return DatasetConfig(
        id="toy",
        name="toy",
        modality="audio",
        role="south_asian",
        status="open",
        access={"kind": "none"},
        licence="-",
        contains=contains,
        builder=builder,
    )


def test_a_fake_only_dataset_can_be_sampled(monkeypatch, tmp_path):
    root = tmp_path / "copy"
    for rel in ["fake/xx/gen_a/1.wav", "fake/xx/gen_a/2.wav", "fake/xx/gen_b/3.wav"]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"wav")
    builder = _builder(monkeypatch, "dtb_toy_fake_only", label_from_path=lambda p: "fake")
    record = sample_dataset(_config(builder, ["fake"]), tmp_path / "out", per_label=2, local=root)
    assert len(record["chosen"]) == 2
    assert record["candidates"] == {"real": 0, "fake": 3}


def test_labels_can_come_from_a_metadata_file(monkeypatch, tmp_path):
    root = tmp_path / "copy"
    (root / "clips").mkdir(parents=True)
    for n in (1, 2, 3):
        (root / "clips" / f"{n}.wav").write_bytes(b"wav")
    (root / "meta.csv").write_text("file,label\n1.wav,bona-fide\n2.wav,spoof\n3.wav,spoof\n")

    def labels_from_metadata(files: dict[str, bytes]) -> dict[str, str]:
        rows = files["meta.csv"].decode().splitlines()[1:]
        names = dict(row.split(",") for row in rows)
        return {f"clips/{k}": "real" if v == "bona-fide" else "fake" for k, v in names.items()}

    builder = _builder(
        monkeypatch,
        "dtb_toy_meta",
        METADATA_FILES=["meta.csv"],
        labels_from_metadata=labels_from_metadata,
    )
    dest = tmp_path / "out"
    record = sample_dataset(_config(builder, ["real", "fake"]), dest, per_label=1, local=root)
    assert record["candidates"] == {"real": 1, "fake": 2}
    assert sorted(record["chosen"])[0] == "clips/1.wav"
    # The builder reads the same file when it builds the manifest of the sample.
    assert (dest / "meta.csv").exists() and record["metadata"] == ["meta.csv"]


def test_hugging_face_listing_covers_only_the_patterns(monkeypatch):
    import huggingface_hub
    from huggingface_hub.hf_api import RepoFile, RepoFolder

    from deeptrace_bench.sample import _HfSource

    listed = []

    def fake_tree(repo_id, path_in_repo=None, *, recursive=False, revision=None, **kw):
        listed.append(path_in_repo)
        return [
            RepoFolder(path=f"{path_in_repo}/gen", oid="1"),
            RepoFile(path=f"{path_in_repo}/gen/1.wav", size=1, oid="2"),
        ]

    monkeypatch.setattr(huggingface_hub, "list_repo_tree", fake_tree)
    config = _config("dtb_unused:build_manifest", ["fake"]).model_copy(
        update={
            "access": DatasetConfig.model_fields["access"].annotation(
                kind="hf",
                repo="org/big",
                allow_patterns=["README.md"],
                language_patterns={"si": ["fake/si/*"], "hi": ["fake/hi/*"]},
            )
        }
    )
    files = _HfSource(config, ["si"]).candidates()
    # One listing of the Sinhala folder, not of the whole repo; plain names need no listing.
    assert listed == ["fake/si"]
    assert files == ["README.md", "fake/si/gen/1.wav"]


def test_a_gated_download_says_what_to_do(monkeypatch, tmp_path):
    import huggingface_hub
    from huggingface_hub.errors import GatedRepoError

    from deeptrace_bench.fetch import ManualStepRequiredError
    from deeptrace_bench.sample import _HfSource

    def refuse(*args, **kwargs):
        raise GatedRepoError("Cannot access gated repo")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", refuse)
    config = _config("dtb_unused:build_manifest", ["fake"]).model_copy(
        update={
            "access": DatasetConfig.model_fields["access"].annotation(
                kind="hf_gated", repo="org/gated", page="https://example.org/gated"
            )
        }
    )
    with pytest.raises(ManualStepRequiredError, match="accept its terms"):
        _HfSource(config, None).materialize(["fake/1.wav"], tmp_path)


def test_strata_get_their_own_share():
    # Like MAVOS-DD: one real folder and fakes from several methods, one of them rare.
    paths = [f"real/{i}.mp4" for i in range(10)]
    paths += [f"swap/{i}.mp4" for i in range(20)] + ["vc/0.mp4"]
    label = lambda p: "real" if p.startswith("real/") else "fake"  # noqa: E731
    stratum = lambda p: p.split("/")[0]  # noqa: E731
    chosen = choose(paths, label, 2, seed=0, stratum_of=stratum)
    assert sorted(p.split("/")[0] for p in chosen) == ["real", "real", "swap", "swap", "vc"]


def test_asvspoof_samples_both_labels_in_every_split():
    # The eval-only evalset needs real and fake eval items; an unstratified sample of 2 + 2
    # can draw its real items from train and dev only.
    from deeptrace_bench.datasets.asvspoof2019_la import sample_stratum

    paths, labels = [], {}
    for split in ("train", "dev", "eval"):
        for i in range(20):
            path = f"LA/ASVspoof2019_LA_{split}/flac/LA_{split[0].upper()}_{i:07d}.flac"
            paths.append(path)
            labels[path] = "real" if i < 3 else "fake"
    chosen = choose(paths, labels.get, 2, seed=0, stratum_of=sample_stratum)
    eval_labels = {labels[p] for p in chosen if "_LA_eval/" in p}
    assert eval_labels == {"real", "fake"}
