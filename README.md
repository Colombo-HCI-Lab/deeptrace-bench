# deeptrace-bench

Benchmark harness for deepfake detection in South Asia. It runs open video and audio detectors on South Asian deepfake datasets under one evaluation setup and reports results per group (gender, skin tone, language, generator), so numbers from different datasets and models can be compared, and the gaps between groups can be measured.

It starts with existing datasets and models: reproduce each model's published numbers, test it zero-shot on South Asian data, then fine-tune baselines. Later it evaluates the data collected through [deeptrace](https://github.com/Colombo-HCI-Lab/deeptrace), and becomes the benchmark released with that dataset.

**Status (2026-10-12):** 25 detectors have adapters and 17 pass a parity check against their upstream code. 48 model and evalset pairs are scored, including the first zero-shot round on South Asian speech (Urdu, Bengali, Hindi and the 16 languages of the IndicTTS challenge set, Nepali among them) and the first video runs, on HiDF. Detectors trained only on English spoofing data fail on South Asian speech, while those post-trained on large multilingual spoof collections sit near the ceiling. The evaluation setup is still proposed. [Progress](https://colombo-hci-lab.github.io/deeptrace-bench/progress/) has the round-by-round record and what's next; [Results](https://colombo-hci-lab.github.io/deeptrace-bench/results/) has the numbers and their caveats.

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
uv run scripts/smoke.py --model aasist         # an audio model, on a few Urdu clips
```

[Smoke test](https://colombo-hci-lab.github.io/deeptrace-bench/smoke_test/) explains the smoke test and its report; [The store](https://colombo-hci-lab.github.io/deeptrace-bench/store_layout/) maps everything the scripts write.

## Workflow

1. **Set up** models (`scripts/setup_models.py`: pinned upstream code, weights, hash check; video models also get the shared face detector) and datasets (`scripts/setup_datasets.py`: download, `--from` a copy obtained by request, or `--sample N` for a few items).
2. **Build manifests** (`setup_datasets.py <id> --manifest`): every dataset becomes one table, one row per item, with labels, generator, identities and group attributes.
3. **Score** (`scripts/score.py --model <id> --evalset <id>`): the model must read the evalset's modality and the contamination guard checks the pair first; video and image items go through the shared face pipeline; scoring is sharded and resumes after a kill.
4. **Evaluate** (`scripts/evaluate.py`): AUC and EER overall and per group, with bootstrap intervals, failure counts and the contamination verdict. `--publish` copies the aggregates into `results/`.
5. **Smoke test** (`scripts/smoke.py`): steps 1 to 4 on a few items per dataset, in `DTB_ROOT/smoke/`, with a report per run. Run it after any change, and to prove a new model or dataset is wired up.
6. **Parity** (`scripts/parity.py`): before a model's numbers count, its adapter and the upstream code score the same ~200 items within 1e-3; the summary goes to `results/parity/<model>.json`.
7. **Profile** (`scripts/profile_models.py --device cuda`): parameters, FLOPs per input, GPU time per input and peak memory for every model, on sampled smoke items, into `results/profile/<model>.json`.
8. **Site** (`scripts/build_site.py`): rewrites the data behind the status page of the public site at <https://colombo-hci-lab.github.io/deeptrace-bench/>, which also carries every doc: a primer for newcomers, the pipeline, which model was scored on which evalset, every model and dataset with its paper and citation count, every countable run, and compute costs with time estimates. It reads run summaries from the store, so rebuild it where `DTB_ROOT` is set, and commit `docs/src/reference/status.json` and the regenerated `catalog.md`; a push to `main` redeploys the site. Plain-language summaries and paper ids live in `docs/src/reference/catalog.yaml`; `--citations` refreshes the counts from Semantic Scholar into `results/citations.json`.

Adding a model or a dataset: [Adding a model](https://colombo-hci-lab.github.io/deeptrace-bench/adding_a_model/), [Adding a dataset](https://colombo-hci-lab.github.io/deeptrace-bench/adding_a_dataset/).

On the cluster the same steps run as SLURM jobs from `slurm/`. See [Running on Curnagl](https://colombo-hci-lab.github.io/deeptrace-bench/cluster/).

## What's in scope

| Wave               | Video                                                                 | Audio                                                                  | Audio-visual |
| ------------------ | --------------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------ |
| 1                  | Xception, EfficientNet-B4, SBI, GenD                                  | AASIST, AASIST-L, XLS-R + AASIST                                       |              |
| 2                  | Effort, LipForensics                                                  |                                                                        | HAVIC        |
| added, no wave yet | UCF, F3Net, SPSL, RECCE, SRM, CORE, FFD, GenD (PE-L)                  | DF Arena 500M and 1B, XLS-R + SLS, XLSR-Mamba, AntiDeepfake (2), AASIST3 |              |

| Data available now                                                                                                                                                                                                  | Accept terms first                                    | Data on request                                                                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Urdu (CSALT), BanglaFake, BD-GRF6, Bangla voices (Mendeley), IndicTTS challenge, IndicSynth + IndicSUPERB, SpeechFake, OpenSLR Sinhala, HiDF, ASVspoof 2019 LA, In-the-Wild                                        | MLAAD, MAVOS-DD Hindi, Svarah                          | FF++, Celeb-DF v2, FakeAVCeleb, InDeepFake, DeePhy, DF-Platter, Deepfake-Eval-2024, Casual Conversations v2, IndieFake, SEA-Spoof |

MAVOS-DD's Hindi subset is the only South Asian video available without an institutional request (accept its terms on Hugging Face). Two open sets, Mendeley Roop/Akool frames and the UniDataPro video preview (with its audio, for the audio-visual path), serve as pipeline tests only: they prove the code works and are never reported. Details, access routes and licence terms: [Models](https://colombo-hci-lab.github.io/deeptrace-bench/models/), [Datasets](https://colombo-hci-lab.github.io/deeptrace-bench/datasets/).

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
docs/            the public site (Astro): the status page and every doc, in
                 src/content/docs/; npm run dev to preview
tests/           synthetic fixtures only
```

## Data stays out of git

Datasets, weights, manifests and per-item scores live under `DTB_ROOT`, never in the repo: most licences are research only, and several count item names as derived data. Only code, configs and aggregate tables are committed, and the pre-commit hook enforces it. See [Data policy](https://colombo-hci-lab.github.io/deeptrace-bench/data_policy/).

## Licence

Not yet chosen. The repo has been public since 2026-10-05, so GitHub Pages can serve the status page at <https://colombo-hci-lab.github.io/deeptrace-bench/>. Upstream detector code keeps its own licence and is never copied in unless it is MIT.
