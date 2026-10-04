# deeptrace-bench

Benchmark harness for deepfake detection in South Asia. It runs open video and audio detectors on South Asian deepfake datasets under one evaluation setup and reports results per group (gender, skin tone, language, generator), so numbers from different datasets and models can be compared, and the gaps between groups can be measured.

It starts with existing datasets and models: reproduce each model's published numbers, test it zero-shot on South Asian data, then fine-tune baselines. Later it evaluates the data collected through [deeptrace](https://github.com/Colombo-HCI-Lab/deeptrace), and becomes the benchmark released with that dataset.

**Status (2026-10-04):** the pipeline runs end to end. One command (`scripts/smoke.py`) fetches GenD and a few items of two open datasets, detects and aligns faces, scores, evaluates and writes a readable report, on a laptop in about a minute. The GenD adapter, the shared face pipeline and dataset sampling are written; GenD's parity with upstream isn't checked yet. Other adapters and dataset builders are documented stubs, written as each model and dataset arrives. The evaluation setup is proposed, not yet settled.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/). Linux x86_64 with CUDA 12 drivers for real runs; Apple Silicon Macs for tests.

```bash
git config core.hooksPath scripts/hooks   # refuse commits that contain data
cp .env.example .env                      # point DTB_ROOT at the store
uv sync
uv run pytest

uv run scripts/setup_models.py gend --record   # GenD, its CLIP backbone, the face detector (~3 GB)
uv run scripts/smoke.py                        # the whole pipeline on a few items; report in DTB_ROOT/smoke/reports/

uv run scripts/setup_models.py --list     # what's available, and its status
uv run scripts/setup_datasets.py --list
uv run scripts/setup_datasets.py urdu_csalt asvspoof2019_la
```

[docs/smoke_test.md](docs/smoke_test.md) explains the smoke test and its report; [docs/store_layout.md](docs/store_layout.md) maps everything the scripts write.

## Workflow

1. **Set up** models (`scripts/setup_models.py`: pinned upstream code, weights, hash check; video models also get the shared face detector) and datasets (`scripts/setup_datasets.py`: download, `--from` a copy obtained by request, or `--sample N` for a few items).
2. **Build manifests** (`setup_datasets.py <id> --manifest`): every dataset becomes one table, one row per item, with labels, generator, identities and group attributes.
3. **Score** (`scripts/score.py --model <id> --evalset <id>`): the model must read the evalset's modality and the contamination guard checks the pair first; video and image items go through the shared face pipeline; scoring is sharded and resumes after a kill.
4. **Evaluate** (`scripts/evaluate.py`): AUC and EER overall and per group, with bootstrap intervals, failure counts and the contamination verdict. `--publish` copies the aggregates into `results/`.
5. **Smoke test** (`scripts/smoke.py`): steps 1 to 4 on a few items per dataset, in `DTB_ROOT/smoke/`, with a report per run. Run it after any change, and to prove a new model or dataset is wired up.

Adding a model or a dataset: [docs/adding_a_model.md](docs/adding_a_model.md), [docs/adding_a_dataset.md](docs/adding_a_dataset.md).

On the cluster the same steps run as SLURM jobs from `slurm/`. See [docs/cluster.md](docs/cluster.md).

## What's in scope

| Wave | Video                                | Audio                            |
| ---- | ------------------------------------ | -------------------------------- |
| 1    | Xception, EfficientNet-B4, SBI, GenD | AASIST, AASIST-L, XLS-R + AASIST |
| 2    | Effort, HAVIC, LipForensics          |                                  |

| Data available now                                                                         | Data on request                                                       |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| Urdu (CSALT), BanglaFake, IndicSynth + IndicSUPERB, MLAAD (accept terms), ASVspoof 2019 LA | FF++, FakeAVCeleb, InDeepFake, DeePhy, DF-Platter, Deepfake-Eval-2024 |

No South Asian video dataset is downloadable without approval yet. Two open sets, Mendeley Roop/Akool frames and the UniDataPro video preview, serve as pipeline tests only: they prove the code works and are never reported. Details, access routes and licence terms: [docs/models.md](docs/models.md), [docs/datasets.md](docs/datasets.md).

## Layout

```
configs/
  models/        one YAML per detector: upstream commit, weights, training data
  datasets/      one YAML per dataset: access, licence terms, provenance
  evalsets/      what gets scored: components, pairing, groups
  tools/         shared preprocessing models (the face detector), pinned like detectors
  eval/          the one evaluation setup
  corpora.yaml   corpora seen only in training (for the contamination guard)
  weights.lock.yaml   sha256 of every weight file
src/deeptrace_bench/
  registry.py    config schemas and cross-checks
  paths.py       the store layout and the smoke namespace
  fetch.py       downloaders and hash checks
  remote_zip.py  reading single members of a remote zip by HTTP range
  sample.py      a few items per dataset, for pipeline tests
  upstream.py    pinned upstream checkouts
  manifest.py    the per-item schema
  datasets/      one manifest builder per dataset
  splits.py      identity-safe fine-tuning splits
  evalset.py     evalset to item table
  preprocess/    audio windows; face detection (SCRFD), alignment and crops
  models/        detector adapters behind one interface
  score.py       resumable, sharded scoring
  runs.py        run records (provenance)
  eval/          metrics, contamination guard, publish guard, shortcut probe
scripts/         setup_models, setup_datasets, score, evaluate, smoke; hooks/pre-commit
slurm/           job templates for Curnagl
patches/         compatibility patches for upstream code
results/         published aggregate results (no item-level data)
docs/            smoke test, store layout, adding a model or dataset, models, datasets,
                 evaluation, cluster, data policy
tests/           synthetic fixtures only
```

## Data stays out of git

Datasets, weights, manifests and per-item scores live under `DTB_ROOT`, never in the repo: most licences are research only, and several count item names as derived data. Only code, configs and aggregate tables are committed, and the pre-commit hook enforces it. See [docs/data_policy.md](docs/data_policy.md).

## Licence

Not yet chosen; this repo is private. Upstream detector code keeps its own licence and is never copied in unless it is MIT.
