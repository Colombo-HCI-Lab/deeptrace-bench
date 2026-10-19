---
title: "Evaluation"
description: "How a score becomes a number: evalsets, preprocessing, the settings still open, metrics, contamination verdicts, the shortcut probe, run flags"
group: benchmark
order: 1
---

Every setting lives in [`configs/eval/default.yaml`](/configs/eval/default.yaml), still `proposed` until the team settles [decision 5](progress.md#open-decisions-that-shape-the-results). A changed setting changes every run id ([naming rules](store_layout.md#naming-rules)), so results from different settings never mix. Terms such as evalset, parity and score direction are defined in [How it works](method.md#key-terms).

## Evalsets, not datasets

An evalset (`configs/evalsets/<id>.yaml`) combines one or more datasets, a filter, optionally a forced label, and a pairing:

| Pairing | Real and fake come from | Examples | In headline numbers |
| --- | --- | --- | --- |
| `same_corpus` | the same recordings or speakers | Urdu CSALT, BanglaFake, FakeAVCeleb, IndicSynth with its source IndicSUPERB | yes |
| `cross_corpus` | different corpora, so microphone, noise, loudness or sample rate can give the label away | MLAAD fakes with real speech from elsewhere | never; each gets a [shortcut probe](#shortcut-probe-planned) |

## Preprocessing

| Input | Preprocessing | Item score | Failure status |
| --- | --- | --- | --- |
| Audio | 16 kHz mono, 64,600-sample windows (about 4 s); a clip shorter than a window is tiled up to one, as AASIST does | mean of window scores | `too_short` under 1 s |
| Video, images | one shared face detector (insightface's SCRFD-10GF, `det_10g.onnx`) on 32 evenly spaced frames, keeping the largest face per frame and caching its box and five landmarks; an image is a one-frame video. Each model then cuts its own crop from the cache | mean of frame scores | `no_face`: fewer than 8 frames with a face (video), or none (image) |
| Frame-run models | their own sampling and detection cache. HAVIC: frames at 5 per second from the start (at most 50) with the audio under them (`preprocess/av.py`). LipForensics: the first 100 consecutive frames, cut into mouth crops from FAN's 68 landmarks with upstream's warp (`preprocess/mouths.py`). Written 2026-10-05 | per model ([Models](models.md#per-model)) | as for video |

Each model crops the cached faces its own way (size, margin and alignment from its config), because detectors do worse on crops unlike the ones they trained on. Code: `preprocess/faces.py`, `preprocess/scrfd.py`.

| `input.crop.align` | Crop | Used by |
| --- | --- | --- |
| `five_point` | landmark alignment | GenD, the DeepfakeBench detectors, Effort |
| `none` | a square around the box | no model yet |
| `box` | the box widened per side and stretched | SBI's test crop, HAVIC's raw box |

- **Tight face crops.** The detector needs some context around a face, so datasets of ready-made face crops defeat it. A frame or image with no face is retried once with a black border (`image_pad_retry`, `video_pad_retry`). On HiDF's 512 px face-crop videos this took detection from 6 of 32 frames to 32 of 32 on half the sampled videos. The cache records which frames needed it.
- **Native mode.** Reproducing a paper with the model's own preprocessing (`score.py --preprocessing native`) gets a different run id. Not built for video and image models; `score.py` refuses it.

## Settings still open

Decision 5 settles these:

| Setting | Ours now | Alternative |
| --- | --- | --- |
| Audio clip score | the mean over every 4 s window, refusing clips under 1 s | the first 4 s only, or the whole utterance, as some upstreams do ([Models](models.md#per-model)) |
| Face detection threshold | 0.4, `detector.py`'s default | 0.1, as GenD's README preprocesses FF++ |
| Frame sampling | 32 evenly spaced frames | `detector.py`'s "at least 32 frames with a face, spreading out from the middle" |
| Crop size | each model's config; `size: null` keeps native size (GenD, HAVIC) | one rule for every model |
| Padding retry for tight crops | on | off |
| Video decoding | OpenCV, which applies the rotation stored in phone videos | PyAV, used by GenD's preprocessing, which doesn't. Settle at parity time; decoding is one function, `read_video_frames` |

## Metrics

| Metric | How |
| --- | --- |
| AUC and EER | fake as positive, overall and per group, each with a 95% stratified bootstrap interval (1,000 resamples) |
| Per-group FPR and FNR | at one global threshold, the overall EER point; they catch a group that ranks well but is miscalibrated, which an AUC gap misses |
| Fake detection rate at 5% FPR | overall and per group |
| Failures | `no_face`, `too_short` and `unreadable` counted per group. Face detectors fail more often on darker skin, so dropping failures would bias the gaps |
| Small groups | under 30 real or fake items: reported and flagged, their gaps not interpreted |
| Group gap | the headline fairness number. Its definition (`eval.metrics.group_gap`) is still open: max minus min, worst group against the rest, or the largest pair with non-overlapping intervals |

Decision 4 (proposed, `reporting.attribute_rule`): an attribute becomes a required field of the data dictionary if both models in a modality shift by 5 or more AUC points across its values. It is recorded before results arrive so it can't be fitted to them.

## Contamination verdicts

Before scoring, the contamination guard (`eval.contamination.check`) compares the model's `training_data` with the evalset's datasets and everything they derive from. Every result carries the verdict:

| Verdict            | Meaning                                                                                        | Effect                                   |
| ------------------ | ---------------------------------------------------------------------------------------------- | ---------------------------------------- |
| `clean`            | no known overlap (training on a dataset's train split and testing on its test split is clean)  | normal                                   |
| `source_overlap`   | pretraining data or a shared source corpus overlaps (e.g. Common Voice in XLS-R's pretraining) | reported with a warning                  |
| `overlap_excluded` | the model trained on this dataset, but the items it saw are listed, so they're excluded        | evaluated on the rest                    |
| `contaminated`     | the model trained on this dataset and the items it saw aren't known                            | refused unless `--force`; never headline |

It currently refuses HAVIC on FakeAVCeleb, AASIST3 and DF Arena on MLAAD, and DF Arena and both AntiDeepfake models on ASVspoof 2019 LA eval.

Its blind spot: it compares datasets, not generators, so a model that trained (through some other dataset) on the TTS system behind an evalset's fakes still gets `clean`. Our own fine-tuned checkpoints list their training items (`items_file`), so the guard covers them exactly.

## Shortcut probe (planned)

For every cross-corpus evalset: a cross-validated logistic regression on trivial features only (duration, loudness, silence ratio, spectral rolloff and bandwidth, original sample rate), its AUC reported next to the detectors'. If the probe alone scores high, the evalset measures the corpus, not fakeness.

## Fine-tuning splits

`splits.identity_safe_split` links items sharing a `subject_id` or `source_subject_id` and keeps each linked group in one split, so no face or voice is on both sides. Seed and code are in git; assignments in `DTB_ROOT/splits/`.

## Flags on each run

Beside the verdict, the [all-runs table](https://colombo-hci-lab.github.io/deeptrace-bench/results/#all-runs) flags:

| Flag | Means |
| --- | --- |
| `cross-corpus` | the evalset pairs different corpora |
| `below chance` | AUC under 0.5 |
| `no parity check` | the adapter hasn't been shown to score like upstream; the run is provisional |
| `score direction unchecked` | the model hasn't scored above chance on a reproduction evalset, so its scores could be read backwards; the run is provisional |
