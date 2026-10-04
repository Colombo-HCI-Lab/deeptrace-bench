"""The contamination guard on the real configs and on hand-made cases."""

from __future__ import annotations

from deeptrace_bench.eval.contamination import Verdict, check
from deeptrace_bench.registry import Component, EvalsetConfig, ModelConfig, TrainingData


def _evalset(*datasets: str, split: str | None = None) -> EvalsetConfig:
    comps = [Component(dataset=d, filter={"split": split} if split else {}) for d in datasets]
    return EvalsetConfig(
        id="t",
        modality="audio",
        role="south_asian",
        pairing="same_corpus",
        components=comps,
        status="pending_access",
    )


def test_havic_is_refused_on_fakeavceleb(registry):
    result = check(registry.model("havic"), registry.evalset("fakeavceleb"), registry)
    assert result.verdict == Verdict.CONTAMINATED


def test_havic_is_clean_on_deephy(registry):
    result = check(registry.model("havic"), registry.evalset("deephy"), registry)
    assert result.verdict == Verdict.CLEAN


def test_aasist3_is_refused_on_mlaad(registry):
    result = check(registry.model("aasist3"), registry.evalset("mlaad_hi"), registry)
    assert result.verdict == Verdict.CONTAMINATED


def test_reproduction_on_the_test_split_is_clean(registry):
    assert (
        check(registry.model("xception"), registry.evalset("ffpp_test"), registry).verdict
        == Verdict.CLEAN
    )
    assert (
        check(registry.model("aasist"), registry.evalset("asvspoof2019_la_eval"), registry).verdict
        == Verdict.CLEAN
    )


def test_wave_one_is_clean_on_the_open_south_asian_sets(registry):
    for model_id in ("aasist", "aasist_l", "xlsr_aasist"):
        for evalset_id in ("urdu_csalt", "banglafake", "indicsynth_hi", "mlaad_si"):
            verdict = check(
                registry.model(model_id), registry.evalset(evalset_id), registry
            ).verdict
            # BanglaFake's mozilla sub-corpus is Common Voice, which XLS-R was pretrained on.
            expected = (
                Verdict.SOURCE_OVERLAP
                if (model_id, evalset_id) == ("xlsr_aasist", "banglafake")
                else Verdict.CLEAN
            )
            assert verdict == expected, (model_id, evalset_id)


def test_training_on_the_whole_dataset_without_a_list_is_contaminated(registry):
    model = ModelConfig(
        id="m",
        name="m",
        modality="audio",
        status="test",
        licence="-",
        training_data=[TrainingData(dataset="urdu_csalt", level="train")],
    )
    assert check(model, _evalset("urdu_csalt"), registry).verdict == Verdict.CONTAMINATED


def test_an_items_list_turns_refusal_into_exclusion(registry):
    model = ModelConfig(
        id="m",
        name="m",
        modality="audio",
        status="test",
        licence="-",
        training_data=[
            TrainingData(dataset="urdu_csalt", level="finetune", items_file="splits/x.txt")
        ],
    )
    result = check(model, _evalset("urdu_csalt"), registry)
    assert result.verdict == Verdict.OVERLAP_EXCLUDED
    assert result.exclude_items_files == ["splits/x.txt"]


def test_shared_source_corpus_is_a_warning(registry):
    # FakeAVCeleb is derived from VoxCeleb2, so training on VoxCeleb2 overlaps its identities.
    model = ModelConfig(
        id="m",
        name="m",
        modality="video",
        status="test",
        licence="-",
        training_data=[TrainingData(dataset="voxceleb2", level="train")],
    )
    assert check(model, registry.evalset("fakeavceleb"), registry).verdict == Verdict.SOURCE_OVERLAP


def test_pretraining_on_the_evalset_dataset_is_a_warning(registry):
    model = ModelConfig(
        id="m",
        name="m",
        modality="audio",
        status="test",
        licence="-",
        training_data=[TrainingData(dataset="indicsuperb", level="pretrain")],
    )
    assert (
        check(model, registry.evalset("indicsynth_hi"), registry).verdict == Verdict.SOURCE_OVERLAP
    )


def test_sets_that_both_draw_on_youtube_do_not_overlap(registry):
    # "From YouTube" names no particular videos, so FF++ training data says nothing about
    # DeePhy's real videos; only a shared, identifiable corpus counts as overlap.
    verdict = check(registry.model("xception"), registry.evalset("deephy"), registry)
    assert verdict.verdict == Verdict.CLEAN, verdict.reasons
