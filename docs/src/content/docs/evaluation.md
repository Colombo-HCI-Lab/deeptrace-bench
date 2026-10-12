---
title: "Evaluation"
description: "Evalsets, preprocessing, metrics, failure accounting, contamination verdicts, the shortcut probe"
group: benchmark
order: 5
---

All settings live in [`configs/eval/default.yaml`](/configs/eval/default.yaml), marked `proposed` until the team settles decision 5 (due 2026-10-09, still open as of 2026-10-12). Changing it changes every run id, so results from different settings never mix.

## Evalsets, not datasets

Models are scored only on evalsets (`configs/evalsets/<id>.yaml`), never on raw datasets. An evalset lists components (a dataset, a filter, optionally a forced label) and a pairing:

- **same_corpus**: real and fake from the same recordings or speakers (Urdu CSALT, BanglaFake, FakeAVCeleb, IndicSynth with its source corpus IndicSUPERB). These give headline numbers.
- **cross_corpus**: real and fake from different corpora (MLAAD fakes with real speech from elsewhere). A detector can score well by separating the corpora (microphone, noise, loudness, sample rate) rather than spotting synthesis. Never averaged into headline results; each gets a shortcut probe (below).

Every result row carries its pairing.

## Preprocessing

- **Audio:** 16 kHz mono, 64,600-sample windows (about 4 s). Clips under 1 s get status `too_short`; clips shorter than a window are tiled up to one, as AASIST does. Clip score is the mean of window scores.
- **Video and images:** one face detector, insightface's SCRFD-10GF (`det_10g.onnx`, the `scrfd_10g` tool; the detector GenD's `detector.py` runs, which calls it RetinaFace), on CPU over 32 uniformly sampled frames per video. It keeps the largest face per frame and caches its box and five landmarks (`faces/<dataset>/<detector>__<hash>/`). An image is a one-frame clip. Each model cuts its own crop from the cache (size, margin, alignment from its config), since models degrade on crops they weren't trained on. GenD aligns the five landmarks to its template at scale 1.3 and keeps native size. Fewer than 8 face frames (video) or none (image) gives `no_face`. Clip score is the mean of frame scores. Written 2026-10-04 in `preprocess/faces.py` and `preprocess/scrfd.py`.
- **Crop modes** (`input.crop`, per model): `five_point` (GenD's landmark alignment), `none` (a square around the box), `box` (the box widened per side and stretched; SBI's test crop and HAVIC's raw box).
- **Frame-run models** override the shared sampling in their config and get their own detection cache. HAVIC: frames at 5 per second from the start (at most 50) plus the audio under them (`preprocess/av.py`). LipForensics: the first 100 consecutive frames, cut into mouth crops from FAN's 68 landmarks with upstream's warp (`preprocess/mouths.py`). Written 2026-10-05.
- **Tight face crops:** datasets of ready-made face crops (e.g. the Mendeley frames) defeat the detector, which needs context. An image with no face is retried once with a black border of half its size per side (`image_pad_retry`); since 2026-10-05 so is a video frame (`video_pad_retry`). HiDF ships 512 px face-crop videos: without the retry SCRFD found a face in 6 of 32 frames of half its sampled videos (32 of 32 with it). The cache records which frames needed it.
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

Before scoring, `eval.contamination.check` compares the model's `training_data` with the evalset's datasets and everything they derive from. Every result carries the verdict:

| Verdict            | Meaning                                                                                        | Effect                                   |
| ------------------ | ---------------------------------------------------------------------------------------------- | ---------------------------------------- |
| `clean`            | no known overlap (training on a dataset's train split and testing on its test split is clean)  | normal                                   |
| `source_overlap`   | pretraining data or a shared source corpus overlaps (e.g. Common Voice in XLS-R's pretraining) | reported with a warning                  |
| `overlap_excluded` | the model trained on this dataset, but the items it saw are listed, so they're excluded        | evaluated on the rest                    |
| `contaminated`     | the model trained on this dataset and the items it saw aren't known                            | refused unless `--force`; never headline |

The guard currently refuses HAVIC on FakeAVCeleb, AASIST3 and DF Arena on MLAAD, and DF Arena and both AntiDeepfake models on ASVspoof 2019 LA eval. It compares datasets, not generators: a model trained (through another dataset) on the TTS system that made an evalset's fakes still gets `clean`. The multilingual audio models' near-perfect scores on the open South Asian sets may partly reflect that ([Results](results.md#three-tiers)). Our own fine-tuned checkpoints must list their training items (`items_file`), so the guard covers our baselines too.

## Shortcut probe (planned)

For every cross-corpus evalset: a cross-validated logistic regression on trivial features only (duration, loudness, silence ratio, spectral rolloff and bandwidth, original sample rate), its AUC reported next to the detectors'. If the probe alone scores high, the evalset measures the corpus, not fakeness.

## Fine-tuning splits

`splits.identity_safe_split` links items sharing a `subject_id` or `source_subject_id` and keeps each linked group in one split, so no face or voice is on both sides. Seed and code are in git; assignments in `DTB_ROOT/splits/`.

## Flags on each run

Beside the verdict, the [all-runs table](https://colombo-hci-lab.github.io/deeptrace-bench/results/#all-runs) flags: `cross-corpus`; `below chance`; `no parity check` (the adapter hasn't been shown to score like upstream); `score direction unchecked` (the model hasn't scored above chance on a reproduction evalset, so its scores could be read backwards). Runs with either of the last two are provisional.
