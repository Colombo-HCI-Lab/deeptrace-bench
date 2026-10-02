# deeptrace-bench

Benchmark harness for deepfake detection in South Asia. It runs open video and audio detectors on South Asian deepfake datasets under one evaluation setup and reports results per group (gender, skin tone, language, generator), so numbers from different datasets and models can be compared, and the gaps between groups can be measured.

It starts with existing datasets and models: reproduce each model's published numbers, test it zero-shot on South Asian data, then fine-tune baselines. Later it evaluates the data collected through [deeptrace](https://github.com/Colombo-HCI-Lab/deeptrace), and becomes the benchmark released with that dataset.

**Status (2026-10-03):** scaffolded. The configs, setup scripts, contamination guard, metrics, resumable scoring and tests work. Model adapters, dataset manifest builders and face preprocessing are documented stubs, written as each model and dataset arrives. The evaluation setup is proposed, not yet settled.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/). Linux x86_64 with CUDA 12 drivers for real runs; Apple Silicon Macs for tests.

```bash
git config core.hooksPath scripts/hooks   # refuse commits that contain data
cp .env.example .env                      # point DTB_ROOT at the shared store
uv sync
uv run pytest

uv run scripts/setup_models.py --list     # what's available, and its status
uv run scripts/setup_datasets.py --list
uv run scripts/setup_models.py --wave 1 --record
uv run scripts/setup_datasets.py urdu_csalt asvspoof2019_la
```

## Workflow

1. **Set up** models (`scripts/setup_models.py`: pinned upstream code, weights, hash check) and datasets (`scripts/setup_datasets.py`: download, or `--from` a copy obtained by request).
2. **Build manifests** (`setup_datasets.py <id> --manifest`): every dataset becomes one table, one row per item, with labels, generator, identities and group attributes.
3. **Score** (`scripts/score.py --model <id> --evalset <id>`): the contamination guard checks the pair first; scoring is sharded and resumes after a kill.
4. **Evaluate** (`scripts/evaluate.py`): AUC and EER overall and per group, with bootstrap intervals, failure counts and the contamination verdict. `--publish` copies the aggregates into `results/`.

On the cluster the same steps run as SLURM jobs from `slurm/`. See [docs/cluster.md](docs/cluster.md).

## What's in scope

| Wave | Video                                | Audio                            |
| ---- | ------------------------------------ | -------------------------------- |
| 1    | Xception, EfficientNet-B4, SBI, GenD | AASIST, AASIST-L, XLS-R + AASIST |
| 2    | Effort, HAVIC, LipForensics          |                                  |

| Data available now                                                                         | Data on request                                                       |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| Urdu (CSALT), BanglaFake, IndicSynth + IndicSUPERB, MLAAD (accept terms), ASVspoof 2019 LA | FF++, FakeAVCeleb, InDeepFake, DeePhy, DF-Platter, Deepfake-Eval-2024 |

No South Asian video dataset is downloadable without approval yet. Details, access routes and licence terms: [docs/models.md](docs/models.md), [docs/datasets.md](docs/datasets.md).

## Layout

```
configs/
  models/        one YAML per detector: upstream commit, weights, training data
  datasets/      one YAML per dataset: access, licence terms, provenance
  evalsets/      what gets scored: components, pairing, groups
  eval/          the one evaluation setup
  corpora.yaml   corpora seen only in training (for the contamination guard)
  weights.lock.yaml   sha256 of every weight file
src/deeptrace_bench/
  registry.py    config schemas and cross-checks
  fetch.py       downloaders and hash checks
  upstream.py    pinned upstream checkouts
  manifest.py    the per-item schema
  splits.py      identity-safe fine-tuning splits
  evalset.py     evalset to item table
  preprocess/    audio windows; face crops (planned)
  models/        detector adapters behind one interface
  score.py       resumable, sharded scoring
  runs.py        run records (provenance)
  eval/          metrics, contamination guard, shortcut probe
scripts/         setup_models, setup_datasets, score, evaluate; hooks/pre-commit
slurm/           job templates for Curnagl
patches/         compatibility patches for upstream code
results/         published aggregate results (no item-level data)
docs/            models, datasets, evaluation, cluster, data policy
tests/           synthetic fixtures only
```

## Data stays out of git

Datasets, weights, manifests and per-item scores live under `DTB_ROOT`, never in the repo: most licences are research only, and several count item names as derived data. Only code, configs and aggregate tables are committed, and the pre-commit hook enforces it. See [docs/data_policy.md](docs/data_policy.md).

## Licence

Not yet chosen; this repo is private. Upstream detector code keeps its own licence and is never copied in unless it is MIT.
