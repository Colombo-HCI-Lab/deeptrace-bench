"""The configs load, cross-reference each other, and resolve to real code."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from deeptrace_bench.fetch import read_lock
from deeptrace_bench.registry import Component, EvalsetConfig, WeightSpec, accepts


def test_configs_have_no_problems(registry):
    assert registry.problems(import_adapters=True) == []


def test_wave_one_is_what_the_plan_says(registry):
    wave1 = {m.id for m in registry.models.values() if m.wave == 1}
    assert wave1 == {
        "xception",
        "efficientnet_b4",
        "sbi",
        "gend",
        "aasist",
        "aasist_l",
        "xlsr_aasist",
    }


def test_every_model_lists_training_data(registry):
    for m in registry.models.values():
        if m.status != "test":
            assert m.training_data, f"{m.id} has no training_data; the guard can't check it"


def test_lock_only_names_known_weights(registry):
    for owner_id, files in read_lock().items():
        names = {w.name for w in registry.weight_owner(owner_id).weights}
        # Folder weights are locked per file, as "<folder>/<relative path>".
        assert {key.split("/")[0] for key in files} <= names, f"unknown files for {owner_id}"


def test_the_face_detector_is_a_pinned_tool(registry):
    detector = registry.face_detector()
    assert detector.kind == "face_detector"
    assert all(w.kind != "manual" for w in detector.weights)


def test_video_models_score_images_but_not_audio(registry):
    gend = registry.model("gend")
    assert accepts(gend, registry.evalset("mendeley_roop_akool"))
    assert accepts(gend, registry.evalset("unidatapro_videos"))
    assert not accepts(gend, registry.evalset("urdu_csalt"))
    assert not accepts(registry.model("aasist"), registry.evalset("unidatapro_videos"))


def test_pipeline_test_data_stays_out_of_real_evalsets(registry):
    leaky = EvalsetConfig(
        id="leaky",
        modality="image",
        role="south_asian",
        pairing="same_corpus",
        components=[Component(dataset="mendeley_roop_akool")],
        status="ready",
    )
    registry.evalsets["leaky"] = leaky
    try:
        assert any("pipeline_test" in p for p in registry.problems())
    finally:
        del registry.evalsets["leaky"]


def test_snapshot_weights_need_a_full_revision():
    with pytest.raises(ValidationError, match="40-hex"):
        WeightSpec(name="clip", kind="hf_snapshot", repo="org/model", revision="main")


def test_zip_members_only_come_from_urls():
    with pytest.raises(ValidationError, match="member only applies"):
        WeightSpec(name="w", kind="hf", repo="org/m", filename="w", member="w.onnx")
    with pytest.raises(ValidationError, match="plain relative path"):
        WeightSpec(name="w", kind="url", url="https://example.invalid/a.zip", member="../w")


def test_eval_config_is_still_proposed(registry):
    # Flip this test when the team settles decision 5.
    assert registry.eval.status == "proposed"
