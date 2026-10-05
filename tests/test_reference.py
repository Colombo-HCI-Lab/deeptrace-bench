"""The Markdown reference describes inputs in words and reads features from manifests."""

from __future__ import annotations

import pandas as pd

from deeptrace_bench.reference import model_input, recorded_features, render


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
    assert '<a id="model-m"></a>' in text and '<a id="dataset-d"></a>' in text
    assert "trained on D (train split)" in text
    assert "none: real clips only" in text and "Sinhala" in text
