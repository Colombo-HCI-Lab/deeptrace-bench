# Evaluation

The settings live in one place, [`configs/eval/default.yaml`](../configs/eval/default.yaml). It is marked `proposed` until the team settles decision 5 (due 2026-10-09). Changing it changes every run id, so results from different settings never mix.

## Evalsets, not datasets

A model is never scored on a raw dataset, only on an evalset (`configs/evalsets/<id>.yaml`). An evalset lists one or more components (a dataset, a filter, and optionally a forced label) and says how real and fake are paired:

- **same_corpus**: real and fake come from the same recordings or speakers (Urdu CSALT, BanglaFake, FakeAVCeleb, IndicSynth with IndicSUPERB, its source corpus). These give headline numbers.
- **cross_corpus**: real and fake come from different corpora (MLAAD fakes with real speech from elsewhere). A detector can score well here by telling the corpora apart (microphone, noise, loudness, sample rate) rather than by spotting synthesis. These numbers are never averaged into headline results, and each cross-corpus evalset gets a shortcut probe (below).

Every result row carries its pairing.

## Preprocessing

- **Audio:** 16 kHz mono, cut into 64,600-sample windows (about 4 s). Clips under 1 s get status `too_short` rather than being tiled; shorter-than-a-window clips are tiled up to one window, as AASIST does. A clip's score is the mean of its window scores.
- **Video and images:** one face detector, insightface's SCRFD-10GF (`det_10g.onnx`, the `scrfd_10g` tool; the detector GenD's own `detector.py` runs, which calls it RetinaFace), runs on CPU over 32 uniformly sampled frames per video, keeps the largest face per frame, and caches its box and five landmarks in the store (`faces/<dataset>/<detector>__<hash>/`). An image is a one-frame clip. Each model then cuts its own crop from the cache (size, margin, alignment from its config), because models trained on different crops degrade on the wrong one. GenD aligns the five landmarks to its template at scale 1.3 and keeps the native size. A video with fewer than 8 face frames, or an image with none, gets status `no_face`. A clip's score is the mean of its frame scores. Written 2026-10-04 in `preprocess/faces.py` and `preprocess/scrfd.py`.
- **Crop modes:** `five_point` (GenD's landmark alignment), `none` (a square around the box) and `box` (the box itself widened per side and stretched, SBI's test crop and HAVIC's raw box), set per model in `input.crop`.
- **Models that read frame runs** override the shared sampling with their own, in their config, and get their own detection cache: HAVIC takes frames at 5 per second from the start (at most 50) and the audio under them (`preprocess/av.py`), LipForensics the first 100 consecutive frames, cut into mouth crops from FAN's 68 landmarks with upstream's own warp (`preprocess/mouths.py`). Written 2026-10-05.
- **Tight face crops:** datasets that ship ready-made face crops (a face filling the image, like the Mendeley frames) defeat the detector, which needs context around a face. An image with no face is tried once more with a black border of half its size on each side (`image_pad_retry`), and since 2026-10-05 so is a video frame (`video_pad_retry`): HiDF ships 512 px videos of face crops, and without the retry SCRFD found a face in 6 of 32 frames of half its sampled videos (32 of 32 with it). The cache records which frames needed it.
- **Native mode:** reproducing a paper's numbers uses that model's own preprocessing (`score.py --preprocessing native`). Native and shared runs get different run ids. Not built yet for video and image models; `score.py` refuses it.

Open choices for decision 5, all in `configs/eval/default.yaml`:

- The detection threshold: 0.4 is `detector.py`'s default; GenD's README preprocesses FF++ at 0.1.
- Frame sampling: uniform 32 frames (ours) against `detector.py`'s "at least 32 frames with a face, spreading out from the middle".
- Native crop size against a fixed size per model, and the padding retry for tight crops.
- Video decoding: OpenCV applies the rotation stored in phone videos; PyAV, which GenD's preprocessing uses, doesn't. Settle this at parity time; decoding is one function (`read_video_frames`).

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
