# Smoke test

One command runs the whole pipeline (setup, sampling, face detection, scoring, evaluation) on a handful of items per dataset and writes a report a person can read. It is how you check that the harness works after a change, and how a new model or dataset proves it is wired up, before any real run.

```bash
uv run scripts/smoke.py                         # GenD on every pipeline-test evalset
uv run scripts/smoke.py --model gend --sample 3 # 3 real + 3 fake per dataset
uv run scripts/smoke.py --device cpu            # force a device (default: cuda, mps, cpu)
uv run scripts/smoke.py --evalsets unidatapro_videos --keep
```

It runs on a laptop: GenD on two four-item evalsets takes about a minute on Apple Silicon, after a first-time download of about 3 GB (GenD, its CLIP backbone and the face detector). Weights must be pinned once with `uv run scripts/setup_models.py gend --record`; the smoke test refuses to run on unpinned weights.

## What it does

Everything happens in the smoke namespace, `$DTB_ROOT/smoke/` (see [store_layout.md](store_layout.md)), with the same scripts a real run uses:

1. `setup_models.py <model>`: upstream checkout and weights, plus the face detector for a video model, checked against `configs/weights.lock.yaml`.
2. `setup_datasets.py <datasets> --sample 2 --manifest`: two real and two fake items per dataset, chosen deterministically from the remote file list (a Hugging Face listing, or a zip's table of contents read with HTTP range requests), then the dataset's own builder makes the manifest.
3. `score.py --smoke ... --save-crops 4` per evalset: face detection, cropping, scoring; up to four crops per item are kept for the report.
4. `evaluate.py --smoke ...` per evalset: metrics and summary.

Earlier smoke faces, scores and results are deleted first, so a fixed bug can't pass on stale scores (`--keep` keeps them). Sampled datasets are kept, since they don't change.

## The pipeline-test datasets

| Evalset               | Modality | Items                                  | Exercises                                                          |
| --------------------- | -------- | -------------------------------------- | ------------------------------------------------------------------ |
| `mendeley_roop_akool` | image    | face-crop frames, Bangladeshi/Indian   | the image path, the padded detection retry, remote-zip sampling    |
| `unidatapro_videos`   | video    | 5 real phone videos, 5 face-swap fakes | video decoding, frame sampling, frame-score averaging, HF sampling |
| `unidatapro_av`       | audio-video | the same ten videos, with sound     | the audio-visual path (frames at a fixed rate plus the audio track) |

All three are `role: pipeline_test`; a video model runs only `unidatapro_videos` of the two UniDataPro evalsets, and a model that reads frame runs (LipForensics) skips the image set: the registry won't let them into a real evalset, and `evaluate.py --publish` refuses them. See [datasets.md](datasets.md) for their sources and licences.

No open audio set is a pipeline test, so an audio model smoke-tests on two real and two fake clips of `urdu_csalt` by default (`AUDIO_FALLBACK` in `smoke.py`): `uv run scripts/smoke.py --model aasist`. Any other evalset works with `--evalsets`, e.g. `--evalsets in_the_wild banglafake`. It is still the smoke namespace: nothing there can be published.

## Reading the report

`$DTB_ROOT/smoke/reports/latest.md` is a copy of the newest report (a copy, not a link, so a store shared between machines can always replace it). It holds:

- **PASS or FAIL** and, on failure, the reasons.
- **Provenance**: repo commit (and whether the tree had uncommitted changes), sample size and seed, the model's upstream commit and weight hashes, the face detector's hash, library versions, and a link to the full log.
- **One table per evalset**: every item with its label, method, score, status, faces found out of frames sampled, and a thumbnail of its first face crop. The run's scores and results folders are linked.

Open the crops and look. Faces should be centred and upright with level eyes, with black edges only where the 1.3 margin runs past the frame. A crop that is off-centre, rotated or mostly background means detection or alignment broke, whatever the scores say.

## Pass criteria

- every step exits 0;
- every item has a score row, with a known status (`ok`, `no_face`, `unreadable`, `too_short`, `empty_audio`, `no_audio`);
- each evalset has at least one `ok` item;
- every score is a probability in [0, 1].

That proves the plumbing, not the model. With four items, AUC and EER are noise: they show the metrics code ran, nothing more. The numbers are never reported.

## Checks worth repeating by hand

- `uv run scripts/smoke.py --device cpu` after an MPS or CUDA run: scores should agree to about 1e-3 (on 2026-10-04 GenD agreed to 2e-6 between MPS and CPU). Parity checks against upstream always run on CPU.
- `uv run scripts/score.py --model gend --evalset urdu_csalt` must exit 2 (a video model can't score audio) without creating a run folder.
- `uv run scripts/evaluate.py --smoke --model gend --evalset unidatapro_videos --publish` must exit 2.
