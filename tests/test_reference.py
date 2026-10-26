"""The Markdown reference describes inputs in words and reads features from manifests."""

from __future__ import annotations

import pandas as pd

from deeptrace_bench.reference import held_languages, model_input, recorded_features, render


def test_inputs_in_words(registry):
    assert model_input(registry.model("aasist")) == "4.04 s windows of raw 16 kHz audio"
    expected = "256 px aligned face crops, 32 frames per video"
    assert model_input(registry.model("xception")) == expected
    assert "mouth crops" in model_input(registry.model("lipforensics"))
    assert "with the audio under them" in model_input(registry.model("havic"))


def test_features_come_from_the_full_manifest_first(tmp_path, monkeypatch):
    monkeypatch.setenv("DTB_ROOT", str(tmp_path))
    for folder, gender in ((tmp_path / "manifests", "f"), (tmp_path / "smoke" / "manifests", None)):
        folder.mkdir(parents=True)
        pd.DataFrame({"label": ["real"], "language": ["si"], "g_gender": [gender]}).to_parquet(
            folder / "x.parquet"
        )
    assert recorded_features("x") == ("full", ["language", "gender"])
    assert recorded_features("missing") is None


def test_held_languages_read_the_full_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("DTB_ROOT", str(tmp_path))
    (tmp_path / "manifests").mkdir()
    pd.DataFrame({"label": ["fake", "fake"], "language": ["hi", "hi"]}).to_parquet(
        tmp_path / "manifests" / "x.parquet"
    )
    pd.DataFrame({"label": ["real"]}).to_parquet(tmp_path / "manifests" / "y.parquet")
    assert held_languages("x") == ["hi"]
    assert held_languages("y") is None and held_languages("missing") is None


def test_render_anchors_every_model_and_dataset():
    model = {
        "id": "m", "name": "M", "modality": "audio", "licence": "MIT", "blocked": False,
        "training": [{"name": "D", "level": "train", "split": "train"}], "parity": None,
    }  # fmt: skip
    dataset = {
        "id": "d", "name": "D", "modality": "audio", "role": "south_asian", "status": "open",
        "languages": ["si"], "licence": "CC BY 4.0", "contains": ["real", "fake"],
        "evalsets": ["d"], "recorded": ("sample", ["language"]), "generators": [],
    }  # fmt: skip
    text = render({"built": "2026-10-05", "models": [model], "datasets": [dataset], "runs": []})
    assert text.startswith("---\ntitle: Models and datasets\n")
    assert 'size, cost, what each records"\n---\n' in text and "\n# " not in text
    assert '<a id="model-m"></a>' in text and '<a id="dataset-d"></a>' in text
    assert "trained on D (train split)" in text
    assert "none: real clips only" in text and "Sinhala" in text


def test_render_gives_the_reason_and_marks_cut_candidates():
    model = {
        "id": "m", "name": "M", "modality": "audio", "licence": "MIT", "blocked": False,
        "training": [], "parity": None, "why": "Tests Sinhala.", "cut": True,
    }  # fmt: skip
    dataset = {
        "id": "d", "name": "D", "modality": "audio", "role": "south_asian", "status": "open",
        "languages": ["si"], "licence": "CC BY 4.0", "contains": ["real", "fake"],
        "evalsets": [], "why": "The only Sinhala fakes.",
    }  # fmt: skip
    text = render({"built": "2026-10-26", "models": [model], "datasets": [dataset], "runs": []})
    assert "| Why it's here | Cut candidate. Tests Sinhala. |" in text
    assert "| Why it's here | The only Sinhala fakes. |" in text


def test_render_sets_published_numbers_beside_ours():
    model = {
        "id": "m", "name": "M", "modality": "audio", "licence": "MIT", "blocked": False,
        "training": [], "parity": None,
    }  # fmt: skip
    model["published"] = [
        {"test": "In-the-Wild", "metric": "EER", "value": "1.76%", "source": "paper"}
    ]
    run = {"model": "m", "evalset": "in_the_wild", "auc": 0.9991, "eer": 0.0151}
    text = render({"built": "2026-10-12", "models": [model], "datasets": [], "runs": [run]})
    assert "| Reported by the authors | In-the-Wild: EER 1.76% (paper) |" in text
    assert "| Our runs | `in_the_wild`: AUC 0.999, EER 1.5% |" in text
    assert "scored on 1 evalset" in text
