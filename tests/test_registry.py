"""The configs load, cross-reference each other, and resolve to real code."""

from __future__ import annotations

from deeptrace_bench.fetch import read_lock


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
    for model_id, files in read_lock().items():
        names = {w.name for w in registry.model(model_id).weights}
        # Folder weights are locked per file, as "<folder>/<relative path>".
        assert {key.split("/")[0] for key in files} <= names, f"unknown files for {model_id}"


def test_eval_config_is_still_proposed(registry):
    # Flip this test when the team settles decision 5.
    assert registry.eval.status == "proposed"
