---
title: "Where to run"
description: "Every machine the harness runs on, and how: the DESI lab's SPARK cluster step by step, any Linux GPU machine, and UNIL's Curnagl"
group: working
order: 4
---

The scripts are the same everywhere; only storage and job submission differ. As of 2026-10-16:

| Machine | Status | How |
| --- | --- | --- |
| The DESI lab's SPARK cluster | works; its first scoring run reproduced a published number | [SPARK](#spark) |
| Any Linux machine with an NVIDIA GPU and CUDA 12 drivers | works | [below](#any-linux-gpu-machine) |
| UNIL's Curnagl cluster | set up, never run: the team has no compute project yet | [Curnagl](#curnagl) |
| An Apple Silicon laptop | tests and the smoke test only | [Getting started](getting_started.md) |

## SPARK


SPARK is the DESI lab's GPU cluster at HEC Lausanne. You don't log in to it or queue jobs by hand. You edit two files in a git repo and push a tag; a worker runs the job in a Docker container and commits a summary of the results back. Checked on 2026-10-16, the day the first runs passed.

### How it works

Two repos are involved:

- **deeptrace-bench** (this repo, public) holds the benchmark code. Code changes always land here first.
- **[UNIL-DESI/deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments)** (private) holds what SPARK runs. SPARK only runs repos in the [UNIL-DESI](https://github.com/UNIL-DESI) org, so the runs live there. It holds no benchmark code: it pins a deeptrace-bench commit and names a stage to run.

```
you      edit .cluster-ci and params.yaml in deeptrace-experiments, push, push the cluster-run tag
worker   checks out deeptrace-bench at the pinned commit, builds its locked environment, runs the stage
bot      commits reports/ (aggregates only) back to deeptrace-experiments' main
```

Datasets, weights and per-item scores stay on the worker's disk (the job volume) between runs, so a second run on the same dataset skips the download.

### One-time setup

| You need | Check with | If it's missing |
| --- | --- | --- |
| Membership of the UNIL-DESI org | <https://github.com/UNIL-DESI> lists you under People | ask Oshan |
| Write access to deeptrace-experiments | `gh repo view UNIL-DESI/deeptrace-experiments` | ask Oshan |
| git and the GitHub CLI, signed in | `gh auth status` | install [gh](https://cli.github.com/), then `gh auth login` |

```bash
gh repo clone UNIL-DESI/deeptrace-experiments
```

That's all. You don't need SSH, a UNIL account, a GPU or a Hugging Face token (the repo's `HF_TOKEN` secret covers downloads). The UNIL VPN only matters for Cluster-CI's dashboard; logs reach you through GitHub. Cluster-CI's own guide tells you to run its `install.sh`: skip it, because it overwrites files this repo has already fixed.

### The run loop

Every run is the same five steps, from inside `deeptrace-experiments`:

```bash
git pull --rebase                                           # 1. the bot commits after every run
# 2. edit .cluster-ci (where and what) and params.yaml (settings); see the recipes below
git commit -am "score aasist on urdu csalt" && git push     # 3. commit your change
git tag -f cluster-run && git push -f origin cluster-run    # 4. pushing the tag starts the run
gh run list -R UNIL-DESI/deeptrace-experiments --limit 3    # 5. find the run, follow it,
gh run watch <run id> -R UNIL-DESI/deeptrace-experiments    #    then git pull --rebase and read reports/
```

In `.cluster-ci`, two lines change from run to run, and always together: `ALLOWED_WORKERS` (which machine) and `STAGES` (what to run). `MAX_RUNTIME_HOURS` sets the time limit (24 at most). Leave the rest alone. The file on `main` holds whatever the last person ran, so read it before you edit.

| Stage | `ALLOWED_WORKERS` | Settings in `params.yaml` | Report |
| --- | --- | --- | --- |
| `gpu_check` | `HEC45801,HEC45803` | none | `reports/gpu_check.json` |
| `bench_smoke` | `isipol09` | `bench.commit`, `bench.smoke_models`, `bench.sample` | `reports/bench_smoke.json` |
| `score` | `isipol09` | `bench.commit`, `score.*` | `reports/score.json`, `reports/score_by_method.csv` |
| `hello` | any | none | `reports/hello.json` |

**A stage only reruns when its inputs change.** DVC skips a stage whose settings and pinned commit are the same as last time, so the run passes in seconds and `reports/` doesn't move. Change what you meant to change. `gpu_check` is the exception: it reruns every time.

### Recipes

#### Check that a run works at all

Your first run. It takes a few minutes when a worker is free.

```ini
ALLOWED_WORKERS=HEC45801,HEC45803
STAGES=gpu_check
```

`reports/gpu_check.json` should name an NVIDIA GB10 at about 90 TFLOP/s in bf16.

#### Run the tests and the smoke test at a commit

Proves a deeptrace-bench commit works on the cluster's GPU: `pytest`, then the [smoke test](getting_started.md#first-run-the-smoke-test) for each model in `bench.smoke_models`.

```ini
ALLOWED_WORKERS=isipol09
STAGES=bench_smoke
```

```yaml
bench:
  commit: <full deeptrace-bench commit hash>
  smoke_models: gend aasist     # one video and one audio model by default
  sample: 2                     # real and fake items per evalset
```

`reports/bench_smoke.json` has `passed`, the GPU the environment saw, and each step's time.

#### Score a model on an evalset

Runs the same scripts as any other machine (`setup_models.py`, `setup_datasets.py --manifest`, `score.py --device cuda`, `evaluate.py`), then compares the result with a run already published.

```ini
ALLOWED_WORKERS=isipol09
STAGES=score
MAX_RUNTIME_HOURS=6
```

```yaml
score:
  model: aasist
  evalset: urdu_csalt
  reference:                    # the published run to compare with (from results.md or its summary)
    auc: 0.8632242084258313
    eer: 0.22218952324896998
    n: 6794
    n_failed: 19
  tolerance: 0.001
```

Read `reports/score.json`:

| Field | Means |
| --- | --- |
| `passed` | scoring and evaluation both finished |
| `matches_reference` | same item and failure counts, and AUC and EER within `tolerance` |
| `overall`, `diff` | the run's metrics, and their distance from the reference |
| `manifest_matches_repo` | the downloaded dataset is exactly the one committed in deeptrace-bench |
| `hf_token` | whether the Hugging Face secret reached the job (never its value) |

**What it can score today.** A pair that already has a published run, on an evalset made from one dataset with a manifest committed under [`results/manifests/`](/results/manifests): `urdu_csalt`, `banglafake`, `bangla_voices`, `in_the_wild` and `indictts_challenge`. A new pair, or an evalset like `indicsynth_bn`, needs the stage extended first; ask Oshan.

**How long it takes.** The first download of a dataset is the slow part: Hugging Face lets the cluster fetch about 1,000 files per 5 minutes, so the stage downloads in rounds. Urdu CSALT (6,800 clips) took about 20 minutes to download and 88 seconds to score on the 3090. Later runs on the same worker skip the download.

#### Run code you just changed

SPARK runs deeptrace-bench at the commit pinned in `params.yaml`, never your working copy. So:

1. Land the change in deeptrace-bench: tests, smoke test, push to `main`.
2. Copy the commit hash: `git -C path/to/deeptrace-bench rev-parse HEAD`.
3. Set `bench.commit` in deeptrace-experiments' `params.yaml` to it, then run `bench_smoke` first. Every stage that uses the pin reruns.

The worker builds the environment from `uv.lock`: about a minute when the lock changed, seconds when it didn't.

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
| "Workspace Sanitizer" aborts over `dvc.lock` hashes | a failed run's reports were committed without the lock; `git checkout <last good commit> -- reports/<file>`, commit, run again |
| `dvc-viewer: command not found` before any stage | the worker's job volume lost a launcher (a Cluster-CI bug); ask Oshan |
| "The NVIDIA driver on your system is too old" | keep `DOCKER_IMAGE_AMD64=nvcr.io/nvidia/pytorch:25.06-py3`; it's the only image that fits `isipol09`'s driver |
| `manifest_matches_repo: false` | the download didn't finish in its 12 rounds, or `hf_token` is false; check the token, then run again (it resumes) |
| Killed by the memory guard | keep `REQUIRED_RAM` at 16GB or more; it also covers Cluster-CI's own install |
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
