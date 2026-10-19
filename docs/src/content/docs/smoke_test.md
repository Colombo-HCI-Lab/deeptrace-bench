---
title: "Smoke test"
description: "One command that runs the whole pipeline on a few items per dataset, and how to read its report"
group: working
order: 1
---

One command runs the whole pipeline (setup, sampling, face detection, scoring, evaluation) on a handful of items per dataset and writes a readable report. Run it after any pipeline change, and before a real run of a new model or dataset. It proves the code works, not that the model is good: nothing it produces is ever reported.

```bash
uv run scripts/smoke.py                         # GenD on every pipeline-test evalset
uv run scripts/smoke.py --model gend --sample 3 # 3 real + 3 fake per dataset
uv run scripts/smoke.py --device cpu            # force a device (default: cuda, mps, cpu)
uv run scripts/smoke.py --evalsets unidatapro_videos --keep
```

Runs on a laptop: GenD on two four-item evalsets takes about a minute on Apple Silicon, after a first-time download of about 3 GB (GenD, its CLIP backbone, the face detector). Pin weights once with `uv run scripts/setup_models.py gend --record`; the smoke test refuses unpinned weights.

## What it does

Everything runs in the smoke namespace, `$DTB_ROOT/smoke/` ([store_layout.md](store_layout.md)), with the same scripts as a real run:

1. `setup_models.py <model>`: upstream checkout and weights (plus the face detector for video models), checked against `configs/weights.lock.yaml`.
2. `setup_datasets.py <datasets> --sample 2 --manifest`: two real and two fake items per dataset, chosen deterministically from the remote file list (a Hugging Face listing, or a zip's table of contents read by HTTP range), then the dataset's builder makes the manifest.
3. `score.py --smoke ... --save-crops 4` per evalset: face detection, cropping, scoring; up to four crops per item kept for the report.
4. `evaluate.py --smoke ...` per evalset: metrics and summary.

Earlier smoke faces, scores and results are deleted first so a fixed bug can't pass on stale scores (`--keep` keeps them). Sampled datasets are kept; they don't change.

## The pipeline-test datasets

| Evalset               | Modality | Items                                  | Exercises                                                          |
| --------------------- | -------- | -------------------------------------- | ------------------------------------------------------------------ |
| `mendeley_roop_akool` | image    | face-crop frames, Bangladeshi/Indian   | the image path, the padded detection retry, remote-zip sampling    |
| `unidatapro_videos`   | video    | 5 real phone videos, 5 face-swap fakes | video decoding, frame sampling, frame-score averaging, HF sampling |
| `unidatapro_av`       | audio-video | the same ten videos, with sound     | the audio-visual path (frames at a fixed rate plus the audio track) |

All three are `role: pipeline_test`: the registry keeps them out of real evalsets and `evaluate.py --publish` refuses them. Of the two UniDataPro evalsets a video model runs only `unidatapro_videos`; frame-run models (LipForensics) skip the image set. Sources and licences: [datasets.md](datasets.md).

No open audio set is a pipeline test, so audio models smoke-test on two real and two fake `urdu_csalt` clips by default (`AUDIO_FALLBACK` in `smoke.py`): `uv run scripts/smoke.py --model aasist`. Any evalset works with `--evalsets`, e.g. `--evalsets in_the_wild banglafake`. Still the smoke namespace: nothing there can be published.

## Reading the report

`$DTB_ROOT/smoke/reports/latest.md` is a copy of the newest report (not a link, so a store shared between machines can always replace it). It holds:

- **PASS or FAIL**, with reasons on failure.
- **Provenance**: repo commit (and whether the tree had uncommitted changes), sample size and seed, the model's upstream commit and weight hashes, the face detector's hash, library versions, and a link to the full log.
- **One table per evalset**: each item's label, method, score, status, faces found out of frames sampled, and a thumbnail of its first face crop; links to the run's scores and results folders.

Look at the crops. Faces should be centred and upright with level eyes, with black edges only where the 1.3 margin runs past the frame. An off-centre, rotated or mostly-background crop means detection or alignment broke, whatever the scores say.

## Pass criteria

- every step exits 0;
- every item has a score row, with a known status (`ok`, `no_face`, `unreadable`, `too_short`, `empty_audio`, `no_audio`);
- each evalset has at least one `ok` item;
- every score is a probability in [0, 1].

With four items, AUC and EER are noise: they only show the metrics code ran.

## Checks worth repeating by hand

- `uv run scripts/smoke.py --device cpu` after an MPS or CUDA run: scores should agree to about 1e-3 (GenD agreed to 2e-6 between MPS and CPU on 2026-10-04). Parity checks against upstream always run on CPU.
- `uv run scripts/score.py --model gend --evalset urdu_csalt` must exit 2 (a video model can't score audio) without creating a run folder.
- `uv run scripts/evaluate.py --smoke --model gend --evalset unidatapro_videos --publish` must exit 2.
