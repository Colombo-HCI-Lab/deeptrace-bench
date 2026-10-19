---
title: "Evaluation"
description: "Evalsets, preprocessing, metrics, failure accounting, contamination verdicts, the shortcut probe"
group: benchmark
order: 5
---

How a score becomes a number: what gets scored, how inputs are prepared, which metrics are reported, and which checks a result must pass. Every setting lives in [`configs/eval/default.yaml`](/configs/eval/default.yaml). It is marked `proposed` until the team settles decision 5 (due 2026-10-09, still open as of 2026-10-19), so every number can still move. Changing a setting changes every run id, so results from different settings never mix.

## Evalsets, not datasets

Models are scored only on evalsets (`configs/evalsets/<id>.yaml`), never on raw datasets. An evalset is an exact slice of data: one or more datasets, a filter, optionally a forced label, and a pairing:

- **same_corpus**: real and fake come from the same recordings or speakers (Urdu CSALT, BanglaFake, FakeAVCeleb, IndicSynth with its source IndicSUPERB). Only these give headline numbers.
- **cross_corpus**: real and fake come from different corpora (MLAAD fakes with real speech from elsewhere). A detector can win here by telling the corpora apart (microphone, noise, loudness, sample rate) instead of spotting synthesis, so these are never averaged into headlines, and each gets a shortcut probe (below).

## Preprocessing

- **Audio:** 16 kHz mono, 64,600-sample windows (about 4 s). Clips under 1 s get status `too_short`; clips shorter than a window are tiled up to one, as AASIST does. Clip score is the mean of window scores.
- **Video and images:** one shared face detector (insightface's SCRFD-10GF, `det_10g.onnx`, the one GenD's code uses) runs on 32 evenly spaced frames per video, keeps the largest face per frame, and caches its box and five landmarks. An image is a one-frame video. Each model then cuts its own crop from the cache (size, margin and alignment from its config), because models do worse on crops unlike the ones they trained on. Fewer than 8 frames with a face (video), or none (image), gives `no_face`. Clip score is the mean of frame scores. Code: `preprocess/faces.py`, `preprocess/scrfd.py`.
- **Crop modes** (`input.crop`, per model): `five_point` (GenD's landmark alignment), `none` (a square around the box), `box` (the box widened per side and stretched; SBI's test crop and HAVIC's raw box).
- **Frame-run models** override the shared sampling in their config and get their own detection cache. HAVIC: frames at 5 per second from the start (at most 50) plus the audio under them (`preprocess/av.py`). LipForensics: the first 100 consecutive frames, cut into mouth crops from FAN's 68 landmarks with upstream's warp (`preprocess/mouths.py`). Written 2026-10-05.
- **Tight face crops:** the detector needs some context around a face, so datasets of ready-made face crops defeat it. A frame or image with no face is retried once with a black border (`image_pad_retry`, `video_pad_retry`). On HiDF's 512 px face-crop videos, this took detection from 6 of 32 frames to 32 of 32 on half the sampled videos. The cache records which frames needed it.
- **Native mode:** reproducing a paper uses the model's own preprocessing (`score.py --preprocessing native`), with a different run id. Not built for video and image models; `score.py` refuses it.

Open choices for decision 5, all in `configs/eval/default.yaml`:

- Detection threshold: 0.4 is `detector.py`'s default; GenD's README preprocesses FF++ at 0.1.
- Frame sampling: uniform 32 (ours) vs `detector.py`'s "at least 32 frames with a face, spreading out from the middle".
- Native crop size vs a fixed size per model, and the padding retry for tight crops.
- Video decoding: OpenCV applies the rotation stored in phone videos; PyAV, used by GenD's preprocessing, doesn't. Settle at parity time; decoding is one function (`read_video_frames`).

## Metrics

- **AUC and EER**, fake as positive, overall and per group, each with a 95% stratified bootstrap interval (1,000 resamples).
- **Per-group FPR and FNR at one global threshold** (the overall EER point), plus fake detection rate at 5% FPR. These catch a group that ranks well but is miscalibrated, which an AUC gap misses.
- **Small groups** (under 30 real or fake items) are reported and flagged, not dropped; their gaps aren't interpreted.
- **Failures** (`no_face`, `too_short`, `unreadable`) are counted per group. Face detectors fail more often on darker skin, so dropping failures would bias the gaps we measure.
- **Group gap:** the headline fairness number. Definition (`eval.metrics.group_gap`) still open: max minus min, worst group vs the rest, or the largest pair with non-overlapping intervals.
- **Score direction:** AUC must exceed 0.5 on each reproduction evalset before an adapter is trusted.

Decision 4 (proposed): an attribute becomes a required data dictionary field if both models in a modality shift by 5 or more AUC points across its values. Recorded under `reporting.attribute_rule` so it is fixed before results arrive. The first video runs suggest a second clause is needed: on HiDF, one group's AUC barely moves while its false-positive rate at the global threshold doubles ([Results](results.md#video-hidf)).

## Contamination verdicts

A model scored on data it trained on looks better than it is. Before scoring, the contamination guard (`eval.contamination.check`) compares the model's `training_data` with the evalset's datasets and everything they derive from. Every result carries the verdict:

| Verdict            | Meaning                                                                                        | Effect                                   |
| ------------------ | ---------------------------------------------------------------------------------------------- | ---------------------------------------- |
| `clean`            | no known overlap (training on a dataset's train split and testing on its test split is clean)  | normal                                   |
| `source_overlap`   | pretraining data or a shared source corpus overlaps (e.g. Common Voice in XLS-R's pretraining) | reported with a warning                  |
| `overlap_excluded` | the model trained on this dataset, but the items it saw are listed, so they're excluded        | evaluated on the rest                    |
| `contaminated`     | the model trained on this dataset and the items it saw aren't known                            | refused unless `--force`; never headline |

It currently refuses HAVIC on FakeAVCeleb, AASIST3 and DF Arena on MLAAD, and DF Arena and both AntiDeepfake models on ASVspoof 2019 LA eval.

**Its blind spot:** it compares datasets, not generators. A model that trained (through some other dataset) on the TTS system that made an evalset's fakes still gets `clean`. That may partly explain the multilingual audio models' near-perfect scores on the open South Asian sets ([Results](results.md#three-tiers)). Our own fine-tuned checkpoints list their training items (`items_file`), so the guard covers them too.

## Shortcut probe (planned)

For every cross-corpus evalset: a cross-validated logistic regression on trivial features only (duration, loudness, silence ratio, spectral rolloff and bandwidth, original sample rate), its AUC reported next to the detectors'. If the probe alone scores high, the evalset measures the corpus, not fakeness.

## Fine-tuning splits

`splits.identity_safe_split` links items sharing a `subject_id` or `source_subject_id` and keeps each linked group in one split, so no face or voice is on both sides. Seed and code are in git; assignments in `DTB_ROOT/splits/`.

## Flags on each run

Beside the verdict, the [all-runs table](https://colombo-hci-lab.github.io/deeptrace-bench/results/#all-runs) flags: `cross-corpus`; `below chance`; `no parity check` (the adapter hasn't been shown to score like upstream); `score direction unchecked` (the model hasn't scored above chance on a reproduction evalset, so its scores could be read backwards). Runs with either of the last two are provisional.
