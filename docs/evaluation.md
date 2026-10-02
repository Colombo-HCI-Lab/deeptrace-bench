# Evaluation

The settings live in one place, [`configs/eval/default.yaml`](../configs/eval/default.yaml). It is marked `proposed` until the team settles decision 5 (due 2026-10-09). Changing it changes every run id, so results from different settings never mix.

## Evalsets, not datasets

A model is never scored on a raw dataset, only on an evalset (`configs/evalsets/<id>.yaml`). An evalset lists one or more components (a dataset, a filter, and optionally a forced label) and says how real and fake are paired:

- **same_corpus**: real and fake come from the same recordings or speakers (Urdu CSALT, BanglaFake, FakeAVCeleb, IndicSynth with IndicSUPERB, its source corpus). These give headline numbers.
- **cross_corpus**: real and fake come from different corpora (MLAAD fakes with real speech from elsewhere). A detector can score well here by telling the corpora apart (microphone, noise, loudness, sample rate) rather than by spotting synthesis. These numbers are never averaged into headline results, and each cross-corpus evalset gets a shortcut probe (below).

Every result row carries its pairing.

## Preprocessing

- **Audio:** 16 kHz mono, cut into 64,600-sample windows (about 4 s). Clips under 1 s get status `too_short` rather than being tiled; shorter-than-a-window clips are tiled up to one window, as AASIST does. A clip's score is the mean of its window scores.
- **Video (planned):** one face detector (RetinaFace via ONNX) runs once per dataset over 32 uniformly sampled frames per video and caches boxes and landmarks in the store. Each model then cuts its own crop from the cache (size, margin, alignment and normalisation from its config), because models trained on different crops degrade on the wrong one. A video with too few face frames gets status `no_face`. A clip's score is the mean of its frame scores.
- **Native mode:** reproducing a paper's numbers uses that model's own preprocessing (`score.py --preprocessing native`). Native and shared runs get different run ids.

## Metrics

- **AUC and EER**, fake as the positive class, overall and per group, each with a 95% stratified bootstrap interval (1,000 resamples).
- **Per-group FPR and FNR at one global threshold** (the overall EER point), plus the fake detection rate at 5% FPR. A model can rank every group well yet be miscalibrated for one, which an AUC gap misses.
- **Small groups** (under 30 real or fake items) are reported and flagged, not dropped, and their gaps aren't interpreted.
- **Failures** (`no_face`, `too_short`, `unreadable`) are counted per group. Face detectors fail more often on darker skin, so silently dropping failures would bias the very gaps we measure.
- **Group gap:** the headline fairness number. Its definition (`eval.metrics.group_gap`) is still open: max minus min, worst group against the rest, or the largest pair with non-overlapping intervals.
- **Score direction:** on each reproduction evalset, AUC must be above 0.5 before the adapter is trusted.

Decision 4 (proposed): an attribute becomes a required data dictionary field if both models in a modality shift by 5 or more AUC points across its values. The rule is recorded under `reporting.attribute_rule` so it is fixed before results arrive.

## Contamination verdicts

Before scoring, `eval.contamination.check` compares the model's `training_data` with the evalset's datasets and everything they were derived from. Every result carries the verdict:

| Verdict            | Meaning                                                                                        | Effect                                   |
| ------------------ | ---------------------------------------------------------------------------------------------- | ---------------------------------------- |
| `clean`            | no known overlap (training on a dataset's train split and testing on its test split is clean)  | normal                                   |
| `source_overlap`   | pretraining data or a shared source corpus overlaps (e.g. Common Voice in XLS-R's pretraining) | reported with a warning                  |
| `overlap_excluded` | the model trained on this dataset, but the items it saw are listed, so they're excluded        | evaluated on the rest                    |
| `contaminated`     | the model trained on this dataset and the items it saw aren't known                            | refused unless `--force`; never headline |

Today the guard refuses HAVIC on FakeAVCeleb and AASIST3 on MLAAD. Fine-tuned checkpoints we make ourselves must list their own training items (`items_file`), so the guard covers our baselines too.

## Shortcut probe (planned)

For every cross-corpus evalset: a logistic regression on trivial features only (duration, loudness, silence ratio, spectral rolloff and bandwidth, original sample rate), cross-validated. Its AUC is reported next to the detectors'. If the probe alone scores high, the evalset measures the corpus, not fakeness.

## Fine-tuning splits

`splits.identity_safe_split` links items that share a `subject_id` or `source_subject_id` and keeps each linked group in one split, so no face or voice appears on both sides. The seed and code live in git; the assignments live in `DTB_ROOT/splits/`.
