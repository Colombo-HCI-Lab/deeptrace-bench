---
title: "Where to run"
description: "Every machine the harness runs on, and how: your first experiment on the DESI lab's SPARK cluster step by step, any Linux GPU machine, and UNIL's Curnagl"
---

The scripts are the same everywhere; only storage and job submission differ. As of 2026-10-19:

| Machine | Status | How |
| --- | --- | --- |
| The DESI lab's SPARK cluster | works for any pair; its scoring runs reproduce published numbers | [SPARK](#spark), with [your first experiment](#your-first-experiment-step-by-step) step by step |
| Any Linux machine with an NVIDIA GPU and CUDA 12 drivers | works | [below](#any-linux-gpu-machine) |
| UNIL's Curnagl cluster | set up, never run: the team has no compute project yet | [Curnagl](#curnagl) |
| An Apple Silicon laptop | tests and the smoke test only | [Getting started](getting_started.md) |

## SPARK

SPARK is the DESI lab's GPU cluster at HEC Lausanne. Checked on 2026-10-19.

### How it works

Think of SPARK as a restaurant kitchen you never walk into.

| In the restaurant | In our setup |
| --- | --- |
| The cookbook | **deeptrace-bench** (this repo, public): all the benchmark code. Code changes always land here first |
| The order slip | **[UNIL-DESI/deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments)** (private): two small files saying which stage to run, with which settings, on which machine, against a pinned deeptrace-bench commit. Its helper `scripts/exp.py` fills them in for you. It lives in the [UNIL-DESI](https://github.com/UNIL-DESI) org because SPARK only runs repos there |
| Ringing the bell | pushing a git tag called `cluster-run` |
| The kitchen | a GPU machine (a worker) that checks out deeptrace-bench at the pinned commit, builds its locked environment and runs the stage in a Docker container |
| The waiter bringing the plate | a bot that commits `reports/` (aggregates only) back to deeptrace-experiments' `main` |

You never log in or queue jobs by hand: you fill in the slip (`scripts/exp.py score …`), ring the bell (`scripts/exp.py go`) and wait for the plate. The kitchen keeps its pantry: datasets, weights and per-item scores stay on the worker's disk between runs, so the second run on a dataset skips the download.

### One-time setup

| You need | Check with | If it's missing |
| --- | --- | --- |
| Membership of the UNIL-DESI org | <https://github.com/UNIL-DESI> lists you under People | ask Oshan |
| Write access to deeptrace-experiments | `gh repo view UNIL-DESI/deeptrace-experiments` | ask Oshan |
| git and the GitHub CLI, signed in | `gh auth status` | install [gh](https://cli.github.com/), then `gh auth login` |
| uv, which runs the helper | `uv --version` | install [uv](https://docs.astral.sh/uv/) |

```bash
gh repo clone UNIL-DESI/deeptrace-experiments && cd deeptrace-experiments
```

That's all. You don't need SSH, a UNIL account, a GPU or a Hugging Face token (the repo's `HF_TOKEN` secret covers downloads). The UNIL VPN only matters for Cluster-CI's dashboard; logs reach you through GitHub. Cluster-CI's own guide tells you to run its `install.sh`: skip it, because it overwrites files this repo has already fixed.

### Your first experiment, step by step

An experiment here is one model scored on one evalset: "how well does AASIST-L tell real from fake Urdu speech?" You get back AUC, EER, the item and failure counts, and a check that they make sense.

#### 1. Pick a model and an evalset

Any model with an adapter on any evalset whose datasets are open or click-through ([catalog](catalog.md#models), [Datasets](datasets.md)). The 35 audio pairs below already have a published run (scored on another machine on 2026-10-12), so their runs are also checked against it:

| Models | Evalsets |
| --- | --- |
| `aasist`, `aasist_l`, `xlsr_aasist`, `antideepfake_w2v_small`, `antideepfake_mms_300m`, `df_arena_500m`, `df_arena_1b` | `urdu_csalt`, `bangla_voices`, `banglafake`, `in_the_wild`, `indictts_challenge` |

Start with one of those: it teaches the loop and shows the cluster gets the published numbers. A pair with no published run reports its metrics alone, and a number counts only once the model has passed parity ([Models](models.md)).

#### 2. Fill in the order

```bash
scripts/exp.py score aasist_l urdu_csalt
```

This checks that both names exist at the pinned deeptrace-bench commit and looks up the pair's published run. It then writes the order into the two files, `params.yaml` (what to score, and the reference to check against) and `.cluster-ci` (`STAGES=score` on the worker that runs it), and prints what changed. Add `--no-reference` to skip the check. Leave `bench.commit`, the pinned deeptrace-bench version, alone unless you mean to [run newer code](#run-code-you-just-changed).

**A stage only reruns when its inputs change.** DVC skips a stage whose settings and pinned commit match the last run, so ordering the same thing twice brings nothing new back. (`gpu_check` is the exception: it reruns every time.)

#### 3. Start it

```bash
scripts/exp.py go
```

It pulls the bot's latest commit, refuses if someone else's run is queued or running, commits your order, pushes it, and pushes the `cluster-run` tag that starts the job. Then it follows the run in your terminal and prints the result when it ends. Ctrl-C stops the watching, not the run; `scripts/exp.py watch` picks it up again.

| Dataset | How long |
| --- | --- |
| already on the worker | minutes: AntiDeepfake w2v2 small on Urdu CSALT took 3 min 44 s from tag push to the bot's commit, 169 s of it the stage (scoring 120 s, evaluation 32 s) |
| first use | Hugging Face lets the cluster fetch about 1,000 files per 5 minutes, and one run tries 12 rounds, so at most about 12,000 files. Urdu CSALT (6,800 clips) took about 20 minutes. BanglaFake, In-the-Wild and IndicTTS (25,000 to 31,000 clips) need two or three runs: each ends with the dataset `incomplete` until the download is complete, and the next resumes it |

#### 4. Read the result

```bash
scripts/exp.py show      # pull and summarise the newest report
scripts/exp.py history   # every score run so far, one line each
```

`show` summarises `reports/score.json`:

| Field | Want | Means |
| --- | --- | --- |
| `passed` | `true` | scoring and evaluation both finished |
| `datasets` | each `matches_repo` or `stable` | the download is complete: the same files as the manifest committed in deeptrace-bench (`matches_repo`), or, where none is committed, two fetches a rate window apart agree (`stable`). `incomplete` means run again; `not_fetchable` means the dataset needs an access request |
| `matches_reference` | `true` | same item and failure counts, and AUC and EER within `tolerance`; `null` when there was no reference |
| `overall` | | the run's `auc`, `eer`, `n` and `n_failed`, with intervals (`_lo`, `_hi`) |
| `diff` | close to 0 | the distance from the reference |
| `steps` | every `ok: true` | each phase (environment, evalset, weights, dataset, score, evaluate) with its time; the first `ok: false` shows where it broke |
| `hf_token` | `true` | the Hugging Face secret reached the job (never its value) |

`reports/score_by_method.csv` breaks the result down by how the fakes were made.

#### 5. If it failed

See [When a run fails](#when-a-run-fails).

#### 6. Share what you found

Post the run link and the `overall` numbers to the team. Only aggregates leave the cluster ([Rules](#rules)). Numbers reach the [Results](results.md) page when they are added after a round of runs.

### Other stages

| Stage | Order it with | Worker | Settings in `params.yaml` | Report |
| --- | --- | --- | --- | --- |
| `score` | `scripts/exp.py score <model> <evalset>` | `isipol09` | `bench.commit`, `score.*` | `reports/score.json`, `reports/score_by_method.csv` |
| `bench_smoke` | `scripts/exp.py smoke [--models ..] [--sample N]` | `isipol09` | `bench.commit`, `bench.smoke_models`, `bench.sample` | `reports/bench_smoke.json` |
| `gpu_check` | `scripts/exp.py gpu` | `HEC45801`, `HEC45803` | none | `reports/gpu_check.json` |

Each one then runs with `scripts/exp.py go` and reads with `scripts/exp.py show`, as in steps 3 and 4. The helper sets `STAGES` and `ALLOWED_WORKERS` in `.cluster-ci` together. To do it by hand, take the workers from the stage's `meta` in `dvc.yaml`.

- **Check that a run works at all.** `gpu_check` is a few minutes when a worker is free. `reports/gpu_check.json` should name an NVIDIA GB10 at about 90 TFLOP/s in bf16.
- **Run the tests and the smoke test at a commit.** `bench_smoke` proves a deeptrace-bench commit works on the cluster's GPU: `pytest`, then the [smoke test](getting_started.md#first-run-the-smoke-test) for each model in `bench.smoke_models` (one video and one audio model by default). `reports/bench_smoke.json` has `passed`, the GPU the environment saw, and each step's time.

#### Run code you just changed

SPARK runs deeptrace-bench at the commit pinned in `params.yaml`, never your working copy. So:

1. Land the change in deeptrace-bench: tests, smoke test, push to `main`.
2. Pin it: `scripts/exp.py pin main` (or `pin <commit hash>`) sets `bench.commit`.
3. Run `scripts/exp.py smoke` and `go` first. Every stage that uses the pin reruns.

The worker builds the environment from `uv.lock`: about a minute when the lock changed, seconds when it didn't.

### Bigger experiments

Some of these need deeptrace-bench changed before an order can name them, so talk to Oshan first. The order is always: code into deeptrace-bench, then move `bench.commit` to that commit ([Run code you just changed](#run-code-you-just-changed)).

| You want to | What's needed |
| --- | --- |
| score a pair with no published run | nothing: `scripts/exp.py score <model> <evalset> --no-reference`, or any pair the site has no run for. Proved on 2026-10-19 with XLS-R + SLS on Urdu CSALT; that model's number doesn't count yet (no parity) |
| use an evalset outside the five above (IndicSynth, HiDF, MAVOS-DD) | the stage reads an evalset's datasets from deeptrace-bench's registry and fetches each, only the languages it needs. Not run on SPARK yet, and some datasets need access requests first ([Datasets](datasets.md)) |
| run a video model (GenD, SBI, Xception) | it runs on the cluster (the smoke test runs GenD); a full video evalset hasn't been scored there yet |
| add a model or a dataset | an adapter or builder in deeptrace-bench, and a parity check before any number counts ([Adding a model](adding_a_model.md), [Adding a dataset](adding_a_dataset.md)) |
| fine-tune a model | not built yet; planned after the evaluation setup is settled ([Progress](progress.md#next)) |
| use the ARM workers' larger memory | not yet: `uv.lock` has no ARM Linux wheels ([Workers and limits](#workers-and-limits)) |

### Rules

- **Aggregates only.** `reports/` gets pass or fail, counts, timings and metric tables, never per-item scores, crops or anything from a licensed dataset ([data_policy.md](data_policy.md)). Don't declare those as DVC `outs` either: Cluster-CI backs outputs up to Google Drive.
- **Pull with `--rebase` before every edit.** The bot commits after each run and rewrites `dvc.yaml`.
- **One run at a time.** Runs share `main`, and a worker runs one job at a time. Tell the team before a long one.
- **This repo's rules still hold:** score through evalsets, keep the contamination guard on, publish nothing from the smoke namespace.
- **Secrets** live in the repo's GitHub secrets and reach stages as environment variables. Never write one to a file or print it.
- **Don't edit Cluster-CI's files:** `.github/workflows/cluster-ci.yml`, `.cluster-ci-tools/`, `.dvc-viewer/`, and the managed block in `AGENTS.md`.

### When a run fails

Get the full log first:

```bash
gh run view <run id> --log -R UNIL-DESI/deeptrace-experiments > run.log
```

| What you see | What to do |
| --- | --- |
| Passes in seconds, `reports/` unchanged | nothing changed, so DVC skipped the stage; change the setting you meant to |
| You pushed the tag and no run starts | you tagged one of the bot's commits (they carry `[skip ci]`); make a commit of your own and tag that |
| Queued for hours | the worker is running someone else's job (up to 24 hours); wait, or `gh run cancel <run id> -R UNIL-DESI/deeptrace-experiments` |
| `CUDA driver initialization failed`, though Cluster-CI's GPU check passed | a one-off GPU glitch (once, on 2026-10-19; the identical retry passed and matched exactly). Fix the next row, then run again |
| "Workspace Sanitizer" aborts over `dvc.lock` hashes | the bot committed a failed run's reports without the lock. `scripts/exp.py fix-reports` puts back the versions the lock expects; then `scripts/exp.py go` again |
| `dvc-viewer: command not found` before any stage | the worker's job volume lost a launcher (a Cluster-CI bug); ask Oshan |
| "The NVIDIA driver on your system is too old" | keep `DOCKER_IMAGE_AMD64=nvcr.io/nvidia/pytorch:25.06-py3`; it's the only image that fits `isipol09`'s driver |
| A dataset `incomplete` (`manifest_matches_repo: false`) | the download didn't finish in one run's 12 rounds ([step 3](#3-start-it)), or `hf_token` is false; check the token, then run again (it resumes) |
| Killed by the memory guard, or out of GPU memory | keep `REQUIRED_RAM` at 16GB or more (it also covers Cluster-CI's own install); `REQUIRED_VRAM` can go up to the 3090's 24 GB |
| Stopped at the time limit | raise `MAX_RUNTIME_HOURS` (24 at most) and run again; scoring resumes from the saved parts |

### Workers and limits

| Worker | GPU | Runs deeptrace-bench |
| --- | --- | --- |
| `isipol09` (x86) | RTX 3090, 24 GB | yes |
| `HEC45801`, `HEC45803` (ARM) | NVIDIA GB10, about 120 GB shared with the CPU | not yet: `uv.lock` has no ARM Linux wheels |

- **Every job stops at 24 hours.** `score.py` resumes, so a long run can be restarted.
- **The GB10s have more memory, not more speed.** They matter for models that don't fit in 24 GB, once the lock supports ARM.

deeptrace-experiments' own `CLAUDE.md` covers how a run works inside and every trap from the first runs. Cluster-CI's documentation, with every `.cluster-ci` key, is at <https://unil-desi.github.io/cluster-ci/>.

## Any Linux GPU machine

Install as in [Getting started](getting_started.md#install), then score on the GPU. Sharding and throughput are in [Results](results.md#experiments).

```bash
uv run scripts/score.py --model aasist --evalset urdu_csalt --device cuda
```

## Curnagl

**Access.** A DCSR compute project must be requested at <https://requests.dcsr.unil.ch>, and each member needs a UNIL account added to it. Then connect through the UNIL VPN, `ssh <unil-user>@curnagl.dcsr.unil.ch`, and put the project's account name (from `Sproject`) in the `--account` line of the files in [`slurm/`](/slurm).

**Partitions.** Jobs run at most 3 days. Ask for one GPU and a realistic `--time`: small, short jobs start sooner.

| Partition | GPUs | Use for |
| --- | --- | --- |
| `gpu-l40` | 8 x L40S 48 GB | inference; the default |
| `gpu` | 14 x A100 40 GB | inference, fine-tuning |
| `gpu-h100` | 8 x H100 94 GB | anything bigger |
| `interactive` | 20 GB A100 slices, 8 h | setup and debugging only |
| `gpu-gh` | GH200 (ARM) | not supported: `uv.lock` has no ARM wheels |

**Storage.** Put `DTB_ROOT` at `/work/<project>/deeptrace-bench/` (no backup) and use `$TMPDIR/dtb-cache/` (fast node-local disk, wiped after each job) as `DTB_CACHE`. The tree is in [store_layout.md](store_layout.md).

- `/users` holds only 50 GB: the repo clone fits there, the venv and uv cache don't. Set `UV_CACHE_DIR` and `UV_PROJECT_ENVIRONMENT` to folders under `/work`, and `UV_LINK_MODE=copy`.
- `/scratch` is purged when it fills, so nothing that must survive goes there.
- Don't write millions of small files (frames, crops) to `/work`; keep them in `$TMPDIR` during a job.

**Setup.** As for [any Linux GPU machine](#any-linux-gpu-machine), plus `mkdir -p logs` for SLURM's logs. Downloads need internet, which compute nodes may not have, so run `setup_models.py` and `setup_datasets.py` on the login node; jobs set `HF_HUB_OFFLINE=1`.

**Jobs.** Scoring is sharded and resumable: resubmit a job that times out.

```bash
sbatch slurm/prepare.sbatch urdu_csalt banglafake           # build manifests (CPU)
sbatch --array=0-7 slurm/score.sbatch aasist urdu_csalt     # 8 shards on gpu-l40
sbatch --dependency=afterok:<jobid> slurm/evaluate.sbatch aasist urdu_csalt
squeue -u $USER                                             # your jobs
squeue --start -j <jobid>                                   # a job's estimated start
```
