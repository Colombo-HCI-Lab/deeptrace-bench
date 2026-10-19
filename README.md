# deeptrace-bench

Benchmark harness for deepfake detection in South Asia. It runs open video and audio detectors on South Asian deepfake datasets under one evaluation setup and reports results per group (gender, skin tone, language, generator), so numbers from different datasets and models can be compared, and the gaps between groups can be measured.

It starts with existing datasets and models: reproduce each model's published numbers, test it zero-shot on South Asian data, then fine-tune baselines. Later it evaluates the data collected through [deeptrace](https://github.com/Colombo-HCI-Lab/deeptrace), and becomes the benchmark released with that dataset.

**Status:** 25 detectors have adapters and 17 pass a parity check (their scores match the authors' own code). The first zero-shot round (2026-10-12) scored 48 model and evalset pairs, on Urdu, Bengali, Hindi and the 16 languages of the IndicTTS challenge set, plus the first video runs on HiDF. Detectors trained only on English spoofing data fail on South Asian speech; those trained on large multilingual spoof collections sit near the ceiling. Since 2026-10-16 runs also go to the DESI lab's SPARK cluster. The evaluation setup is still proposed, so every number can move. [Progress](https://colombo-hci-lab.github.io/deeptrace-bench/progress/) says what's next; [Results](https://colombo-hci-lab.github.io/deeptrace-bench/results/) has the numbers and caveats.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/). Linux x86_64 with CUDA 12 drivers for real runs; Apple Silicon Macs for tests and the smoke test.

```bash
git config core.hooksPath scripts/hooks   # refuse commits that contain data
cp .env.example .env                      # point DTB_ROOT at the store
uv sync
uv run pytest

uv run scripts/setup_models.py gend --record   # GenD, its CLIP backbone, the face detector (~3 GB)
uv run scripts/smoke.py                        # the whole pipeline on a few items; report in DTB_ROOT/smoke/reports/
uv run scripts/smoke.py --model aasist         # an audio model, on a few Urdu clips

uv run scripts/setup_models.py --list          # every model and its status
uv run scripts/setup_datasets.py --list        # every dataset and how to get it
```

## How a run works

| Step | Script | What it does |
| --- | --- | --- |
| Set up | `setup_models.py`, `setup_datasets.py` | fetch pinned upstream code, weights (hash-checked) and data; `--from` a copy obtained by request, `--sample N` for a few items |
| Manifest | `setup_datasets.py <id> --manifest` | one table per dataset, one row per item: label, generator, identity, group attributes |
| Score | `score.py --model <id> --evalset <id>` | checks the model fits the evalset and wasn't trained on it, then scores; sharded, resumes after a kill |
| Evaluate | `evaluate.py` | AUC and EER overall and per group, with intervals and failure counts; `--publish` copies the aggregates into `results/` |
| Smoke test | `smoke.py` | all of the above on a few items, with a readable report; run it after any change |
| Parity | `parity.py` | adapter and upstream code score the same ~200 items within 1e-3 before a model's numbers count |
| Profile | `profile_models.py --device cuda` | parameters, FLOPs, GPU time and memory per model, into `results/profile/` |
| Site | `build_site.py` | rewrites the site's data from the store; commit `docs/src/reference/status.json` and `catalog.md`, and a push to `main` redeploys |

Guides: [Smoke test](https://colombo-hci-lab.github.io/deeptrace-bench/docs/smoke_test/), [The store](https://colombo-hci-lab.github.io/deeptrace-bench/docs/store_layout/), [Adding a model](https://colombo-hci-lab.github.io/deeptrace-bench/docs/adding_a_model/), [Adding a dataset](https://colombo-hci-lab.github.io/deeptrace-bench/docs/adding_a_dataset/), [Running on SPARK](https://colombo-hci-lab.github.io/deeptrace-bench/docs/spark/) (the DESI lab's GPU cluster, through the private [UNIL-DESI/deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments)), and [Curnagl or any GPU machine](https://colombo-hci-lab.github.io/deeptrace-bench/docs/cluster/).

## What's in scope

| Wave | Video | Audio | Audio-visual |
| --- | --- | --- | --- |
| 1 | Xception, EfficientNet-B4, SBI, GenD | AASIST, AASIST-L, XLS-R + AASIST | |
| 2 (planned) | Effort, LipForensics | | HAVIC |
| added, no wave yet | UCF, F3Net, SPSL, RECCE, SRM, CORE, FFD, GenD (PE-L) | DF Arena 500M and 1B, XLS-R + SLS, XLSR-Mamba, AntiDeepfake (2), AASIST3 | |

| Data | Datasets |
| --- | --- |
| Open now | Urdu (CSALT), BanglaFake, BD-GRF6, Bangla voices (Mendeley), IndicTTS challenge, IndicSynth + IndicSUPERB, SpeechFake, OpenSLR Sinhala, HiDF, ASVspoof 2019 LA, In-the-Wild |
| Accept terms first | MLAAD, MAVOS-DD Hindi (the only South Asian video open without an institutional request), Svarah |
| On request | FF++, Celeb-DF v2, FakeAVCeleb, InDeepFake, DeePhy, DF-Platter, Deepfake-Eval-2024, Casual Conversations v2, IndieFake, SEA-Spoof |

Two open sets, the Mendeley Roop/Akool frames and the UniDataPro video preview, are pipeline tests: they prove the code works and are never reported. Access routes and licence terms: [Models](https://colombo-hci-lab.github.io/deeptrace-bench/docs/models/), [Datasets](https://colombo-hci-lab.github.io/deeptrace-bench/docs/datasets/).

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
  preprocess/    audio windows; face detection (SCRFD), alignment and crops; audio-visual
                 inputs; mouth crops from FAN landmarks
  models/        detector adapters behind one interface; _upstream.py imports upstream code,
                 _fairseq.py and _mamba.py stand in for fairseq and mamba-ssm
  parity.py      adapter-versus-upstream checks
  score.py       resumable, sharded scoring
  runs.py        run records (provenance)
  eval/          metrics, contamination guard, publish guard, shortcut probe
scripts/         setup_models, setup_datasets, score, evaluate, smoke, parity; hooks/pre-commit
  parity_reference/  upstream-side scorers for parity checks, one per upstream
slurm/           job templates for Curnagl
patches/         compatibility patches for upstream code
results/         published aggregate results and parity summaries (no item-level data)
docs/            the public site (Astro): the guided pages and every doc, in
                 src/content/docs/; npm run dev to preview
tests/           synthetic fixtures only
```

## Data stays out of git

Datasets, weights, manifests and per-item scores live under `DTB_ROOT`, never in the repo: most licences are research only, and several count item names as derived data. Only code, configs and aggregate tables are committed, and the pre-commit hook enforces it. See [Data policy](https://colombo-hci-lab.github.io/deeptrace-bench/docs/data_policy/).

## Licence

Not yet chosen. The repo has been public since 2026-10-05, so GitHub Pages can serve the site at <https://colombo-hci-lab.github.io/deeptrace-bench/>. Upstream detector code keeps its own licence and is never copied in unless it is MIT.
