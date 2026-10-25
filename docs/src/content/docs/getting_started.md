---
title: "Getting started"
description: "Install the harness, run the smoke test, and read its report"
---

## What you need

[uv](https://docs.astral.sh/uv/), and one of:

| Machine | Runs |
| --- | --- |
| Apple Silicon Mac | the tests and the smoke test |
| Linux x86_64, NVIDIA GPU, CUDA 12 drivers | everything |
| ARM Linux | nothing yet: `uv.lock` has no ARM wheels |

The team's GPU cluster needs none of this setup: see [Where to run](spark.md).

## Install

```bash
git clone https://github.com/Colombo-HCI-Lab/deeptrace-bench.git && cd deeptrace-bench
git config core.hooksPath scripts/hooks   # refuse commits that contain data
cp .env.example .env                      # set DTB_ROOT (the store), and DTB_CACHE if /tmp is small
uv sync --frozen                          # Python 3.12, torch 2.8 (CUDA 12.8 wheels on Linux)
uv run pytest
```

## First run: the smoke test

```bash
uv run scripts/setup_models.py gend --record   # GenD, its CLIP backbone, the face detector (~3 GB)
uv run scripts/smoke.py                        # GenD on every pipeline-test evalset
uv run scripts/smoke.py --model aasist         # an audio model, on a few Urdu clips
```

The smoke test runs the [pipeline](method.md#the-pipeline) from set-up to evaluation on a few items per evalset, with the same scripts as a real run, in the smoke namespace. It proves the code works, not that a model is good. Run it after any pipeline change and before a real run of a new model or dataset. GenD on two four-item evalsets takes about a minute on Apple Silicon. It refuses weights not pinned with `--record`.

| Flag | Default | Effect |
| --- | --- | --- |
| `--model` | `gend` | the model to test |
| `--sample N` | 2 | N real and N fake items per dataset |
| `--evalsets` | every pipeline-test evalset the model accepts; `urdu_csalt` for audio models | any evalset works, still in the smoke namespace |
| `--device` | `auto` (cuda, then mps, then cpu) | force a device |
| `--keep` | off | keep earlier smoke faces and scores, which are otherwise deleted so a fixed bug can't pass on stale ones |

`setup_models.py --list` and `setup_datasets.py --list` print every model and dataset with its status.

## The pipeline-test datasets

No open audio set is a pipeline test, so audio models fall back to `urdu_csalt` (`AUDIO_FALLBACK` in `smoke.py`). Of the two UniDataPro evalsets a video model runs only `unidatapro_videos`; frame-run models (LipForensics) skip the image set. Checked 2026-10-04.

| Evalset | Modality | Items | Exercises | Source, licence |
| --- | --- | --- | --- | --- |
| `mendeley_roop_akool` | image | face-crop frames, Bangladeshi and Indian subjects | the image path, the padded detection retry, remote-zip sampling | [Mendeley Data](https://data.mendeley.com/datasets/pdcp9mjy3z/3), one 4.1 GB zip; CC BY 4.0 |
| `unidatapro_videos` | video | 5 real phone videos, 5 face-swap fakes made from them | video decoding, frame sampling, frame-score averaging, Hugging Face sampling | [Hugging Face](https://huggingface.co/datasets/UniDataPro/deepfake-videos-dataset); CC BY-NC-ND 4.0 |
| `unidatapro_av` | audio-video | the same ten videos, with sound | the audio-visual path | as above |

## Reading the report

`$DTB_ROOT/smoke/reports/latest.md` is a copy of the newest report (a copy, not a link, so a store shared between machines can always replace it).

| Part | Shows |
| --- | --- |
| Verdict | PASS or FAIL, with reasons |
| Provenance | repo commit and whether the tree was dirty, sample size and seed, upstream commit, weight and face-detector hashes, library versions, a link to the full log |
| One table per evalset | each item's label, method, score, status, faces found out of frames sampled, a thumbnail of its first crop |

Look at the crops. Faces should be centred and upright with level eyes, with black edges only where the 1.3 margin runs past the frame. An off-centre, rotated or mostly-background crop means detection or alignment broke, whatever the scores say.

It passes when every step exits 0, every item has a score row with a known status (`ok`, `no_face`, `unreadable`, `too_short`, `empty_audio`, `no_audio`), each evalset has at least one `ok` item, and every score is in [0, 1]. With four items, AUC and EER are noise.

## Checks worth repeating by hand

| Command | Expect |
| --- | --- |
| `uv run scripts/smoke.py --device cpu`, after an MPS or CUDA run | scores agree to about 1e-3 (GenD agreed to 2e-6 between MPS and CPU on 2026-10-04) |
| `uv run scripts/score.py --model gend --evalset urdu_csalt` | exit 2 (a video model can't score audio), no run folder created |
| `uv run scripts/evaluate.py --smoke --model gend --evalset unidatapro_videos --publish` | exit 2 |
